"""
Last updated on 26/08/2026

@author : Foucault BERNARD

Ce fichier contient les fonction de calcul des métriques d'évaluation de classification (recall, precision et F1-score).
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.metrics import classification_report, precision_recall_curve, auc, average_precision_score
from torch.utils.data import DataLoader
from dataset_chunk_seq import LibraryBiasAugmenter
import logging
from dataset_chunk_seq import Dataset_pulse
from utils import collate_fn, normalize_batch
import transformer

def main():
    parser = argparse.ArgumentParser(description='Évaluation des métriques Open Set depuis des poids')
    parser.add_argument('--weights', default="", type=str, help='poids du NN à évaluer')
    parser.add_argument('--dir', default="", type=str, help='Dossier contenant les données')
    parser.add_argument('--rejection_rate', default=0.01, type=float, help='Taux de rejet pour les inconnus (ex: 0.05 = 5%) pour la classif open set')
    args = parser.parse_args()

    print('loading data...')
    logging.info("loading data...")
    min_length = 4  # nb minimal de data dans une classe pour être acceptée pour l'entraînement
    n_views = 10  # nombre de vues par biais par data

    # fichier Json contenant la biblihotèque des sous-modes par mode pour la data augmentation par biais
    with open(f"{args.dir}/JsonBIB_scenario_SR.json", "r") as f:
        loaded_library = json.load(f)
    lib_augmenter = LibraryBiasAugmenter(mode_library=loaded_library, n_views=n_views)

    # Initialisation et Chargement du Modèle
    print(f"Chargement des poids depuis {args.weights}...")
    d_model, nhead, num_layers = 256, 8, 4 # paramètres du modèle
    model = transformer.PDWSupConOnlyTransformer(inputs=3, d_model=d_model, nhead=nhead, num_layers=num_layers)

    model_name = f'{model.__class__.__name__}_Rej{args.rejection_rate}_MinLength={min_length}'

    model.load_state_dict(torch.load(args.weights))
    model.cuda()
    model.eval()


    unknown_label = -1 # label à donner aux classes inconnues pour la classification open set
    classif_mode = 'close_set'  # 'open_set' gives label "-1" to unused classes, 'close_set' throws away unused classes

    signals_train = Dataset_pulse(args.dir, dirname='train', mode='train', min_nb_data=min_length,
                                  classif_mode=classif_mode)  # , mode_library=loaded_library)
    signals_valid = Dataset_pulse(args.dir, dirname=f'valid', mode='test', min_nb_data=0,
                                  label_to_index=signals_train.label_to_index, mini=signals_train.mini,
                                  maxi=signals_train.maxi, classif_mode=classif_mode)
    num_classes = signals_train.num_class
    col_min_tensor = torch.tensor(signals_train.mini, dtype=torch.float32).cuda()
    denom_tensor = torch.tensor(signals_train.denom, dtype=torch.float32).cuda()

    dataloader_train_eval = DataLoader(signals_train, batch_size=1, shuffle=False, collate_fn=collate_fn)
    dataloader_valid = DataLoader(signals_valid, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # ---------------------------------------------------------
    # Calcul des prototypes (Train Set)
    # ---------------------------------------------------------
    print("Extraction des features d'entraînement et calcul de Mahalanobis...")
    all_train_features, all_train_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader_train_eval:
            inputs, labels = inputs.cuda(), labels.squeeze().long().cuda()
            inputs_norm = normalize_batch(inputs, col_min_tensor, denom_tensor, use_log=True)
            features = model(inputs_norm, mode="extract")
            all_train_features.append(features)
            all_train_labels.append(labels)

            # augmentation par biais
            if lib_augmenter is not None:
                labels_list = labels.cpu().tolist()
                augmented_views = lib_augmenter.generate_views(inputs, labels_list, prob_augm=1.0)

                for aug_inputs in augmented_views:
                    aug_inputs = aug_inputs.cuda()
                    aug_inputs_norm = normalize_batch(aug_inputs, col_min_tensor, denom_tensor, use_log=True)
                    aug_features = model(aug_inputs_norm, mode='extract')

                    all_train_features.append(aug_features)
                    all_train_labels.append(labels)

    all_train_features = torch.cat(all_train_features, dim=0)
    all_train_labels = torch.cat(all_train_labels, dim=0)
    all_train_features = F.normalize(all_train_features, p=2, dim=1)

    d_model_feat = all_train_features.size(1)
    known_classes = torch.unique(all_train_labels).tolist()

    means = torch.zeros(num_classes, d_model_feat, device=all_train_features.device)
    inv_covs = torch.zeros(num_classes, d_model_feat, d_model_feat, device=all_train_features.device)
    eye = torch.eye(d_model_feat, device=all_train_features.device)

    for c in known_classes:
        mask = (all_train_labels == c)
        class_feats = all_train_features[mask]
        if class_feats.size(0) > 1:
            mu = class_feats.mean(dim=0)
            means[c] = mu
            centered = class_feats - mu
            cov = (centered.T @ centered) / (class_feats.size(0) - 1)
            inv_covs[c] = torch.linalg.pinv(cov + 1e-4 * eye)
        elif class_feats.size(0) == 1:
            means[c] = class_feats[0]
            inv_covs[c] = eye
        else:
            inv_covs[c] = eye

    def compute_dists(feats):
        N = feats.size(0)
        dists = torch.zeros(N, num_classes, device=feats.device)
        for c in known_classes:
            diff = feats - means[c]
            left_term = torch.matmul(diff, inv_covs[c])
            dists[:, c] = (left_term * diff).sum(dim=1)
        for c in range(num_classes):
            if c not in known_classes:
                dists[:, c] = float('inf')
        return dists

    distances_train = compute_dists(all_train_features)
    thresholds = torch.zeros(num_classes, device=all_train_features.device)
    for c in known_classes:
        mask = (all_train_labels == c)
        if mask.sum() > 0:
            thresholds[c] = torch.quantile(distances_train[mask, c], 1.0 - args.rejection_rate)
        else:
            thresholds[c] = float('inf')

    # ---------------------------------------------------------
    # Inférence sur le Valid Set
    # ---------------------------------------------------------
    print("Inférence sur le Set de Validation...")
    all_preds, all_true, all_min_distances = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloader_valid:
            inputs, labels = inputs.cuda(), labels.squeeze().long().cuda()
            inputs_norm = normalize_batch(inputs, col_min_tensor, denom_tensor, use_log=True)

            features = model(inputs_norm, mode="extract")
            features = F.normalize(features, p=2, dim=1)

            distances = compute_dists(features)

            normalized_distances = distances / (thresholds + 1e-9)
            min_norm_dists, preds = torch.min(normalized_distances, dim=1)
            preds[min_norm_dists > 1.0] = unknown_label

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
            all_min_distances.extend(min_norm_dists.cpu().numpy())

    all_true = np.array(all_true)
    all_preds = np.array(all_preds)
    all_min_distances = np.array(all_min_distances)

    # ---------------------------------------------------------
    # Calcul des Métriques (Precision, Recall, F1, Macro)
    # ---------------------------------------------------------
    print("\nGénération et sauvegarde du Rapport de Classification en image...")

    # Génération des noms de classes pour l'affichage
    target_names_dict = signals_train.index_to_label
    labels_present = np.unique(np.concatenate([all_true, all_preds]))

    target_names = []
    for lbl in labels_present:
        if lbl == unknown_label:
            target_names.append("Inconnu (-1)")
        else:
            target_names.append(f"Classe {target_names_dict.get(lbl, lbl)}")

    # On demande le rapport sous forme de dictionnaire (output_dict=True)
    report_dict = classification_report(all_true, all_preds, labels=labels_present, target_names=target_names,
                                        zero_division=0, output_dict=True)

    # Conversion en tableau (DataFrame) Pandas
    df_report = pd.DataFrame(report_dict).transpose()

    # Formatage : 3 décimales pour les scores, et entiers pour le 'support' (nombre de signaux)
    df_formatted = df_report.copy()
    for col in ['precision', 'recall', 'f1-score']:
        df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.3f}" if not pd.isna(x) else "")
    df_formatted['support'] = df_formatted['support'].apply(lambda x: str(int(x)) if not pd.isna(x) else "")

    # Création de l'image du tableau
    # La hauteur (figsize) s'adapte automatiquement au nombre de classes
    fig, ax = plt.subplots(figsize=(10, len(df_formatted) * 0.3 + 1.5))
    ax.axis('off')  # On cache les axes du graphique

    # Dessin du tableau
    table = ax.table(cellText=df_formatted.values,
                     rowLabels=df_formatted.index,
                     colLabels=df_formatted.columns,
                     cellLoc='center',
                     loc='center',
                     colColours=['#f2f2f2'] * 4,
                     rowColours=['#f2f2f2'] * len(df_formatted))

    # Style du tableau
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Sauvegarde
    save_path_report = f"SupCon_Mahalanobis_OpenSet/results/classification_report_{model_name}.png"
    plt.title("Rapport de Classification (Précision, Rappel, F1)", fontweight="bold", fontsize=14, pad=20)
    plt.savefig(save_path_report, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Rapport de classification sauvegardé sous : {save_path_report}")

if __name__ == '__main__':
    main()