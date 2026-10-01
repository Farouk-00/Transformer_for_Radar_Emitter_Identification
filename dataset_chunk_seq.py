"""
Last updated on 26/08/2026

@author : Foucault BERNARD

Ce fichier contient les classes de traitement des données (chunking des séquences de PDW à la bonne taille, normalisation...)
et de création des augmentations pour l'apprentissage contrastif (noising, cropping, dropping) et des augmentations par
biais (pour data augmentation).
"""

import torch
from torch.utils.data.dataset import Dataset
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence, pack_sequence, pad_sequence
import numpy as np
from pathlib import Path
import logging
import random
from collections import Counter
from utils import normalize_batch, load_database

def chunk_sequences(data_list, label_list, max_time_us=x*50000, max_len=500):
    """
    Découpe les séquences de PDW de manière hybride pour coller au format d'entrée du traitement actuel (x*50ms).
    La découpe se fait si on dépasse `max_time_us` OU si on atteint `max_len` éléments.
    """
    chunked_data = []
    chunked_labels = []

    nb_initial = len(data_list)

    for seq, label in zip(data_list, label_list):
        start_idx = 0
        cumulative_dtoa = 0.0
        seq_len = len(seq)

        for i in range(seq_len):
            dtoa = seq[i][0]
            current_chunk_len = i - start_idx

            # Découpe si on dépasse le temps cumulé OU la taille limite max_len
            if (cumulative_dtoa + dtoa > max_time_us or current_chunk_len >= max_len) and current_chunk_len > 0:
                chunk = seq[start_idx:i]
                chunked_data.append(chunk)
                chunked_labels.append(label)

                # Réinitialisation
                start_idx = i
                cumulative_dtoa = dtoa
            else:
                cumulative_dtoa += dtoa

        if start_idx < seq_len:
            chunk = seq[start_idx:]
            chunked_data.append(chunk)
            chunked_labels.append(label)

    nb_final = len(chunked_data)
    log_msg = f"Chunking hybride : {nb_initial} -> {nb_final} sous-séquences (max {max_time_us} µs ou {max_len} PDW)."
    print(log_msg)
    logging.info(log_msg)

    return chunked_data, chunked_labels


class Dataset_pulse(Dataset):
    """classe de traitement initial des fichiers .npz
        - chunk les séquences
        - crée un dictionnaire index to label (transforme les label en index entier pour l'entraînement
        - retiens le max et le min des données pour la normalisation
        """

    def __init__(self, root, dirname, mode=None, min_nb_data=10, label_to_index=None, maxi=None, mini=None,
                 mode_library=None, classif_mode='open_set'):
        """
        - root et dirname : dossier contenant les données
        - mode : train ou valid
        - min_nb_data : nombre min de data dans une classe d'entraînement nécessaire pour conserver la classe dans le
        dataloader (les autres classes sont ignorées)
        - label_to_index : dictionnaire de traduction des label en indice entier (pour avoir les même entre data train et valid)
        - maxi, mini : valeur max et min des paramètres des data de train pour normalisation des data de validation
        - mode_library : biblihotèque des sous-modes associés à chaque mode pour intégrer leur max et leur min dans la normalisation
        - classif_mode : open_set ou close_set (si 'open_set' alors les classes non retenues car possédant moins de min_nb_data
        données deviennent des classes à rejeter
        """
        self.root = Path(root)
        self.dirname = dirname
        self.mode = mode
        self.classif_mode = classif_mode

        target_dir = self.root / self.dirname
        raw_data, raw_labels = load_database(target_dir)
        chunked_data, chunked_labels = chunk_sequences(raw_data, raw_labels, max_time_us=6*50000, max_len=500)

        label_counts = Counter(chunked_labels)
        valid_labels = {label for label, count in label_counts.items() if count >= min_nb_data}

        filtered_data = []
        filtered_labels = []

        for data, label in zip(chunked_data, chunked_labels):
            if label in valid_labels:
                filtered_data.append(data)
                filtered_labels.append(label)

        nb_removed = len(chunked_labels) - len(filtered_labels)
        log_msg = f"Filtrage min_nb_data={min_nb_data} : {len(valid_labels)} classes conservées. {nb_removed} sous-séquences supprimées."
        print(log_msg)
        logging.info(log_msg)

        if len(filtered_data) == 0:
            raise ValueError(f"Le dataset est vide après le filtrage min_nb_data={min_nb_data}.")

        # Extract des log min et log max pour préparer la normalisation
        data_2d = np.concatenate(filtered_data, axis=0).astype(np.float32)
        epsilon = 1e-9
        data_2d[:, [0, 2]] = np.log10(data_2d[:, [0, 2]] + epsilon)

        # Si on est dans le train_set (pas de mini/maxi fournis par un dataset précédent)
        if maxi is None or mini is None:
            emp_mini = data_2d.min(axis=0)
            emp_maxi = data_2d.max(axis=0)

            # INTEGRATION DES BORNES THEORIQUES DE LA BIBLIOTHEQUE
            if mode_library is not None:
                # Initialisation avec des valeurs de l'infini
                lib_mini = np.array([np.inf, np.inf, np.inf])
                lib_maxi = np.array([-np.inf, -np.inf, -np.inf])

                # On ne cherche que pour les classes présentes dans les données (valid_labels)
                for label in valid_labels:
                    modes = mode_library.get(label, [])
                    for submode in modes:
                        # DTOA/PRI (colonne 0)
                        if 'pri' in submode:
                            if submode['pri'][0] is not None: lib_mini[0] = min(lib_mini[0], submode['pri'][0])
                            if submode['pri'][1] is not None: lib_maxi[0] = max(lib_maxi[0], submode['pri'][1])

                        # Frequence (colonne 1)
                        if 'fn' in submode:
                            if submode['fn'][0] is not None: lib_mini[1] = min(lib_mini[1], submode['fn'][0])
                            if submode['fn'][1] is not None: lib_maxi[1] = max(lib_maxi[1], submode['fn'][1])

                        # Largeur d'impulsion / LI (colonne 2) - Conversion ns -> us
                        if 'li' in submode:
                            if submode['li'][0] is not None: lib_mini[2] = min(lib_mini[2], submode['li'][0] / 1000.0)
                            if submode['li'][1] is not None: lib_maxi[2] = max(lib_maxi[2], submode['li'][1] / 1000.0)
                # Conversion Log10 pour la DTOA et la LI (car data_2d est en log)
                for col in [0, 2]:
                    if lib_mini[col] != np.inf:
                        lib_mini[col] = np.log10(max(lib_mini[col], 0.0) + epsilon)
                    if lib_maxi[col] != -np.inf:
                        lib_maxi[col] = np.log10(max(lib_maxi[col], 0.0) + epsilon)

                # combine l'empirique et la bibliothèque en prenant les valeurs les plus extrêmes
                self.mini = np.minimum(emp_mini, lib_mini)
                self.maxi = np.maximum(emp_maxi, lib_maxi)
            else:
                self.mini = emp_mini
                self.maxi = emp_maxi

            self.denom = self.maxi - self.mini
            self.denom[self.denom == 0] = 1e-9

        else:
            self.mini = mini
            self.maxi = maxi
            self.denom = self.maxi - self.mini

        # Création du dictionnaire de classes après le filtrage
        if label_to_index is None:
            # Si c'est le set d'entraînement, on crée le dictionnaire avec les classes restantes
            classes_uniques = sorted(list(set(filtered_labels)))
            self.num_class = len(classes_uniques)
            self.label_to_index = {label: index for index, label in enumerate(classes_uniques)}
            self.index_to_label = {index: label for index, label in enumerate(classes_uniques)}
        else:
            # Si c'est le set de validation, on utilise le dico fourni par le train
            self.label_to_index = label_to_index
            self.num_class = len(label_to_index)
            self.index_to_label = {v: k for k, v in label_to_index.items()}

        # Transform to tensor et application du label_to_index
        clean_data_info = []
        final_labels_idx = []

        for data, label in zip(filtered_data, filtered_labels):
            if label in self.label_to_index:
                # classe connue
                clean_data_info.append(torch.tensor(data, dtype=torch.float32))
                final_labels_idx.append(self.label_to_index[label])
            else:
                if self.classif_mode == 'open_set':
                    # la class ne fait pas partie du train : inconnue
                    clean_data_info.append(torch.tensor(data, dtype=torch.float32))
                    final_labels_idx.append(-1)
                elif self.classif_mode == 'close_set':
                    pass
                else:
                    raise ValueError(f"classif_mode '{self.classif_mode}' non authorized when calling Dataset_pulse. Change to 'open_set' or 'close_set'")

        self.data_info = clean_data_info

        # Vérification et transformation des labels en Tensor
        if len(final_labels_idx) > 0:
            self.label_info = torch.tensor(final_labels_idx, dtype=torch.long)
        else:
            logging.warning("Attention : la liste finale de labels est vide.")
            self.label_info = torch.tensor([], dtype=torch.long)

        print("\n" + "=" * 30)
        logmsg = f"Dataset finalisé : {len(self.data_info)} séquences réparties sur {self.num_class} classes."
        print(logmsg)
        logging.info(logmsg)
        if len(self.data_info) > 0:
            msg = f"Shape du premier morceau : {self.data_info[0].shape}, 3 premières impulsions : {self.data_info[0][0:3]}"
            print(msg)
            logging.info(msg)
        print("=" * 30 + "\n")

    def __getitem__(self, idx):
        return self.data_info[idx], self.label_info[idx]

    def __len__(self):
        return len(self.label_info)


class PDWAugmenter:
    """
    Générateur de vues pour séquences PDW [batch_size, seq_len, 3] pour l'apprentissage contrastif.
    Applique les transformations de manière probabiliste.
    """
    def __init__(self,col_min_tensor, denom_tensor, use_log=True, noise_std=0.1, drop_prob=0.1, crop_min=0.01, crop_max=0.2):
        #paramètres pour la création des vues
        self.noise_std = noise_std   # Écart-type du bruit
        self.drop_prob = drop_prob   # Probabilité de perdre une impulsion
        self.crop_min = crop_min     # Rognage minimum
        self.crop_max = crop_max     # Rognage maximum

        #paramètres pour la normalisation
        self.col_min = col_min_tensor
        self.denom = denom_tensor
        self.use_log = use_log

    def add_noise(self, x):
        """Ajoute un bruit gaussien"""
        if self.noise_std == 0.0:
            return x

        is_packed = isinstance(x, PackedSequence)
        
        # On extrait les données brutes si c'est une PackedSequence
        base_data = x.data if is_packed else x

        noise = torch.randn_like(base_data) * self.noise_std
        x_noisy = base_data + noise
        
        # Oubli réparé : On garantit que la DTOA (colonne 0) ne devient pas négative
        x_noisy[..., 0] = torch.clamp(x_noisy[..., 0], min=0.0)
        
        if is_packed:
            return PackedSequence(x_noisy, x.batch_sizes, x.sorted_indices, x.unsorted_indices)
        else:
            return x_noisy

    def drop_pulses(self, x):
        """
        Suppression physique de certaines impulsions de la séquence.
        """
        is_packed = isinstance(x, PackedSequence)

        # Déballage et gestion des longueurs d'origine
        if is_packed:
            padded_x, lengths = pad_packed_sequence(x, batch_first=True)
            lengths = lengths.to(padded_x.device)
        else:
            padded_x = x
            lengths = torch.full((padded_x.size(0),), padded_x.size(1), dtype=torch.long, device=x.device)

        batch_size, seq_len, channels = padded_x.size()

        # Masque de validité
        seq_range = torch.arange(seq_len, device=padded_x.device)
        valid_mask = seq_range.unsqueeze(0) < lengths.unsqueeze(1)
        valid_mask = valid_mask.unsqueeze(-1)

        # Génération du masque de drop stochastique
        rand_mask = torch.rand(batch_size, seq_len, 1, device=padded_x.device) > self.drop_prob
        
        mask = valid_mask & rand_mask

        # --- CORRECTION ANTI-CRASH (Séquence vide) ---
        mask_flat = mask.squeeze(-1) 
        empty_seqs = (mask_flat.sum(dim=1) == 0)
        
        # Si une séquence s'est faite intégralement effacer, on garde la 1ère impulsion
        if empty_seqs.any():
            mask_flat[empty_seqs, 0] = True
            mask = mask_flat.unsqueeze(-1)

        # Logique d'accumulation de la DTOA
        DTOA = padded_x[:, :, 0:1]
        S = torch.cumsum(DTOA, dim=1)
        
        idx = torch.arange(seq_len, device=padded_x.device).view(1, seq_len, 1).expand(batch_size, seq_len, 1)
        idx_masked = torch.where(mask, idx, torch.full_like(idx, -1))
        
        last_kept_idx, _ = torch.cummax(idx_masked, dim=1)

        prev_kept_idx = torch.cat([
            torch.full((batch_size, 1, 1), -1, dtype=torch.long, device=padded_x.device),
            last_kept_idx[:, :-1, :]
        ], dim=1)

        gather_idx = torch.clamp(prev_kept_idx, min=0)
        S_prev = torch.gather(S, 1, gather_idx)
        S_prev = torch.where(prev_kept_idx >= 0, S_prev, torch.zeros_like(S_prev))
        
        DTOA_new = S - S_prev

        # Application de la nouvelle DTOA
        x_updated = padded_x.clone()
        x_updated[:, :, 0:1] = DTOA_new

        # SUPPRESSION PHYSIQUE DES IMPULSIONS
        flat_kept_pulses = x_updated[mask_flat]
        new_lengths = mask_flat.sum(dim=1)

        kept_sequences = torch.split(flat_kept_pulses, new_lengths.tolist())

        # Reconditionnement
        if is_packed:
            return pack_sequence(kept_sequences, enforce_sorted=False)
        else:
            return pad_sequence(kept_sequences, batch_first=True)
        
    def crop_pulses(self, x):
        """
        Rogne (crop) aléatoirement le début et la fin de chaque séquence.
        La DTOA de la première impulsion de chaque séquence rognée est réinitialisée à 0.
        """
        is_packed = isinstance(x, PackedSequence)

        # Déballage et récupération des vraies longueurs
        if is_packed:
            padded_x, lengths = pad_packed_sequence(x, batch_first=True)
            lengths = lengths.to(padded_x.device)
        else:
            padded_x = x
            lengths = torch.full((padded_x.size(0),), padded_x.size(1), dtype=torch.long, device=x.device)

        batch_size, seq_len, channels = padded_x.size()

        # Génération des ratios de crop aléatoires
        p_left = torch.rand(batch_size, device=padded_x.device) * (self.crop_max - self.crop_min) + self.crop_min
        p_right = torch.rand(batch_size, device=padded_x.device) * (self.crop_max - self.crop_min) + self.crop_min

        c_left = (p_left * lengths).long()
        c_right = (p_right * lengths).long()

        # Sécurité : on garde au moins 1 impulsion
        max_c_right = torch.clamp(lengths - c_left - 1, min=0)
        c_right = torch.minimum(c_right, max_c_right)

        # Construction du masque de Crop
        seq_range = torch.arange(seq_len, device=padded_x.device).unsqueeze(0)
        start_mask = seq_range >= c_left.unsqueeze(1)
        end_mask = seq_range < (lengths - c_right).unsqueeze(1)
        
        mask = (start_mask & end_mask).unsqueeze(-1)
        mask_flat = mask.squeeze(-1)

        # On clone pour ne pas altérer la séquence originale des autres vues
        x_updated = padded_x.clone()

        # Extraction physique et reconditionnement
        flat_kept_pulses = x_updated[mask_flat]
        new_lengths = mask_flat.sum(dim=1)
        kept_sequences = torch.split(flat_kept_pulses, new_lengths.tolist())

        if is_packed:
            return pack_sequence(kept_sequences, enforce_sorted=False)
        else:
            return pad_sequence(kept_sequences, batch_first=True)

    def generate_views(self, x):
        """
        Génère 3 vues distinctes de la séquence originale :
        1. Vue avec ajout de bruit uniquement
        2. Vue avec impulsions masquées (drop) uniquement
        3. Vue avec rognage (crop) début/fin uniquement
        """

        # --- VUE 1 : BRUIT ---
        view_1_norm = normalize_batch(x, self.col_min, self.denom, self.use_log)
        view_1_final = self.add_noise(view_1_norm)
        
        # --- VUE 2 : DROP ---
        view_2_raw = self.drop_pulses(x)
        view_2_final = normalize_batch(view_2_raw, self.col_min, self.denom, self.use_log)

        # --- VUE 3 : CROP ---
        view_3_raw = self.crop_pulses(x)
        view_3_final = normalize_batch(view_3_raw, self.col_min, self.denom, self.use_log)

        return view_1_final, view_2_final#, view_3_final


class LibraryBiasAugmenter:
    """
    Générateur de n_views vues par ajout de biais probabiliste basé sur la bibliothèque.
    L'objectif est de faire de la DATA AUGMENTATION en permettant au réseau d'apprendre des sous-modes non présents dans les données
    """

    def __init__(self, mode_library, n_views=2):
        self.n_views = n_views
        # dico json
        self.mode_library = mode_library

    def _get_random_bias(self, data_min, data_max, lib_min, lib_max):
        if lib_min is None or lib_max is None:
            return 0.0

        bias_min = lib_min - data_min
        bias_max = lib_max - data_max

        if bias_min > bias_max:
            return (bias_min + bias_max) / 2.0

        return random.uniform(bias_min, bias_max)

    def generate_views(self, x, labels, prob_augm=0.99):

        if self.n_views == 0 or self.mode_library is None:
            return [x]

        views_list = [[] for _ in range(self.n_views)]
        is_packed = isinstance(x, PackedSequence)

        if is_packed:
            padded_x, lengths = pad_packed_sequence(x, batch_first=True)
            sequences = [padded_x[i, :lengths[i]] for i in range(len(lengths))]
        else:
            sequences = [x[i] for i in range(x.size(0))]

        for i, seq in enumerate(sequences):
            label = labels[i]
            mode_submodes = self.mode_library.get(label, None)

            for v in range(self.n_views):
                if mode_submodes is None or len(mode_submodes) == 0 or random.random() > prob_augm:
                    views_list[v].append(seq.clone())
                    continue

                seq_aug = seq.clone()
                chosen_submode = random.choice(mode_submodes)

                epsilon = 1e-6

                # BIAIS DTOA
                dtoas = seq[:, 0]

                # certaines DTOA prennent des valeurs aberrantes (impulsion de début de rafale) + les DTOA sont souvent très bruitées donc il ne faut pas lisser
                # ce bruit à cause de l'augmentation par biais
                median_dtoa = torch.median(dtoas).item()
                abs_deviation = torch.abs(dtoas - median_dtoa)
                mad = torch.median(abs_deviation)
                radius_dtoa_lib = (chosen_submode['pri'][1] - chosen_submode['pri'][0]) / 2.0
                valid_dtoas = dtoas[(dtoas < median_dtoa + 2.0*mad + radius_dtoa_lib + epsilon) & (dtoas > median_dtoa - 2.0*mad - radius_dtoa_lib - epsilon)]
                if len(valid_dtoas) > 0:
                    d_max = valid_dtoas.max().item()
                    d_min = valid_dtoas.min().item()
                else:
                    d_max = dtoas.max().item()  # au cas où il n'y ait pas de DTOA aberrantes
                    d_min = dtoas.min().item()

                b_dtoa = self._get_random_bias(d_min, d_max, chosen_submode['pri'][0], chosen_submode['pri'][1])
                seq_aug[:, 0] += b_dtoa
                seq_aug[:, 0] = torch.clamp(seq_aug[:, 0], min=0.0)

                # BIAIS FN
                freq = seq[:, 1]

                # sécurité contre les données aberrantes
                median_freq = torch.median(freq).item()
                valid_freq = freq[(freq < 3.0 * median_freq + epsilon) & (freq > 0.33 * median_freq - epsilon)]
                if len(valid_freq) > 0:
                    f_max = valid_freq.max().item()
                    f_min = valid_freq.min().item()
                else:
                    f_max = freq.max().item()
                    f_min = freq.min().item()

                b_freq = self._get_random_bias(f_min, f_max, chosen_submode['fn'][0], chosen_submode['fn'][1])
                seq_aug[:, 1] += b_freq
                seq_aug[:, 1] = torch.clamp(seq_aug[:, 1], min=0.0)

                # BIAIS LI
                li = seq[:, 2]

                # sécurité contre les données aberrantes
                median_li = torch.median(li).item()
                valid_li = li[(li < 3.0 * median_li + epsilon) & (li > 0.33 * median_li - epsilon)]
                if len(valid_li) > 0:
                    l_max = valid_li.max().item()
                    l_min = valid_li.min().item()
                else:
                    l_max = li.max().item()
                    l_min = li.min().item()

                # conversion des nano s (unité de la LI en lib) en micro s (unité de la LI en bib scénario SR)
                lib_li_min = chosen_submode['li'][0] / 1000.0 if chosen_submode['li'][0] is not None else None
                lib_li_max = chosen_submode['li'][1] / 1000.0 if chosen_submode['li'][1] is not None else None
                b_li = self._get_random_bias(l_min, l_max, lib_li_min, lib_li_max)
                seq_aug[:, 2] += b_li
                seq_aug[:, 2] = torch.clamp(seq_aug[:, 2], min=0.0)

                views_list[v].append(seq_aug)

        final_views = []
        for v in range(self.n_views):
            if is_packed:
                final_views.append(pack_sequence(views_list[v], enforce_sorted=False))
            else:
                final_views.append(pad_sequence(views_list[v], batch_first=True))

        return final_views
