"""
Last updated on 26/08/2026

@author : Foucault BERNARD

Ce fichier contient les fonction et classes utiles pour le reste du projet
"""

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence, pack_sequence
import pandas as pd
import logging
import copy
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def generate_closeset_final_TEST_confusion_matrix_mahalanobis(
        model,
        dataloader_train_eval,
        dataloader_test,
        num_train_classes,
        num_test_classes,
        col_min_tensor,
        denom_tensor,
        train_class_names,
        test_class_names,
        save_path,
        bias_augmenter=None
):
    """
    fonction de création d'une matrice de confusion sur les données de test en mode close set, adaptée pour des classes
    aux labels différents entre data de test et data d'entraînement.
    """
    print("--- Génération de la Matrice de Confusion Close Set TEST Rectangulaire (Mahalanobis) ---")
    logging.info("--- Génération de la Matrice de Confusion Close Set TEST Rectangulaire (Mahalanobis) ---")

    model.eval()

    # ---------------------------------------------------------
    # ÉTAPE 1 : Extraction d'entraînement et calcul Moyennes/Covariances (Train)
    # ---------------------------------------------------------
    all_train_features = []
    all_train_labels = []

    print("Extraction des features du Train (avec augmentation si fournie)...")
    with torch.no_grad():
        for inputs, labels in dataloader_train_eval:
            inputs, labels = inputs.cuda(), labels.squeeze().long().cuda()

            # --- A. Features des données originales ---
            inputs_norm = normalize_batch(inputs, col_min_tensor, denom_tensor, use_log=True)
            features = model(inputs_norm, mode="train")
            all_train_features.append(features)
            all_train_labels.append(labels)

            # --- B. Features des données augmentées ---
            if bias_augmenter is not None:
                labels_list = labels.cpu().tolist()
                augmented_views = bias_augmenter.generate_views(inputs, labels_list)

                for aug_inputs in augmented_views:
                    aug_inputs = aug_inputs.cuda()
                    aug_inputs_norm = normalize_batch(aug_inputs, col_min_tensor, denom_tensor, use_log=True)
                    aug_features = model(aug_inputs_norm, mode="train")

                    all_train_features.append(aug_features)
                    all_train_labels.append(labels)

    # Concaténation et Normalisation L2
    all_train_features = torch.cat(all_train_features, dim=0)
    all_train_labels = torch.cat(all_train_labels, dim=0)
    all_train_features = F.normalize(all_train_features, p=2, dim=1)

    d_model_feat = all_train_features.size(1)

    means = torch.zeros(num_train_classes, d_model_feat, device=all_train_features.device)
    inv_covs = torch.zeros(num_train_classes, d_model_feat, d_model_feat, device=all_train_features.device)

    epsilon = 1e-4
    eye = torch.eye(d_model_feat, device=all_train_features.device)

    print("Calcul des matrices de covariance Mahalanobis sur le Train...")
    for c in range(num_train_classes):
        mask = (all_train_labels == c)
        class_features = all_train_features[mask]

        if class_features.size(0) > 1:
            mu = class_features.mean(dim=0)
            means[c] = mu
            centered = class_features - mu
            cov = (centered.T @ centered) / (class_features.size(0) - 1)
            inv_covs[c] = torch.linalg.pinv(cov + epsilon * eye)
        elif class_features.size(0) == 1:
            means[c] = class_features[0]
            inv_covs[c] = eye
        else:
            inv_covs[c] = eye

    # ---------------------------------------------------------
    # ÉTAPE 2 : Prédiction sur le set de Test (Assignation forcée)
    # ---------------------------------------------------------
    all_preds = []
    all_true = []

    # On indexe les erreurs selon les noms des classes Test
    misclassified_sizes = {name: [] for name in test_class_names}
    correct_predictions_count = 0
    total_samples = 0

    print("Évaluation sur le Test set (Close Set avec classes différentes)...")
    with torch.no_grad():
        for inputs, labels in dataloader_test:
            inputs, labels = inputs.cuda(), labels.squeeze().long().cuda()
            inputs_norm = normalize_batch(inputs, col_min_tensor, denom_tensor, use_log=True)

            features = model(inputs_norm, mode="train")
            features = F.normalize(features, p=2, dim=1)

            N = features.size(0)
            distances = torch.zeros(N, num_train_classes, device=features.device)

            # Calcul des distances par rapport aux classes TRAIN
            for c in range(num_train_classes):
                diff = features - means[c]
                left_term = torch.matmul(diff, inv_covs[c])
                dist_sq = (left_term * diff).sum(dim=1)
                distances[:, c] = dist_sq

            # Prédiction : Indice de la classe Train la plus proche
            preds = torch.argmin(distances, dim=1)

            preds_cpu = preds.cpu().numpy()
            true_cpu = labels.cpu().numpy()

            try:
                _, lengths_tensor = pad_packed_sequence(inputs, batch_first=True)
                sizes = lengths_tensor.cpu().numpy()
            except Exception:
                sizes = [1.0] * N

            for i in range(N):
                t_idx = int(true_cpu[i])
                p_idx = int(preds_cpu[i])

                t_name = test_class_names[t_idx]
                p_name = train_class_names[p_idx]

                # L'erreur est définie si le nom de la classe prédite != nom de la vraie classe
                if t_name != p_name:
                    misclassified_sizes[t_name].append(float(sizes[i]))
                else:
                    correct_predictions_count += 1

                total_samples += 1

            all_preds.extend(preds_cpu)
            all_true.extend(true_cpu)

    # ---------------------------------------------------------
    # ÉTAPE 3 : Création de la matrice de confusion Rectangulaire
    # ---------------------------------------------------------
    # Lignes = Prédictions (Train), Colonnes = Vérité Terrain (Test)
    cm_counts = np.zeros((num_train_classes, num_test_classes), dtype=float)

    for true_lbl, pred_lbl in zip(all_true, all_preds):
        row = int(pred_lbl)  # Espace Train
        col = int(true_lbl)  # Espace Test
        cm_counts[row, col] += 1

    # Normalisation par colonnes (Vérité Terrain) pour avoir des %
    # Attention aux colonnes vides (division par zéro)
    col_sums = cm_counts.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    cm_normalized = cm_counts / col_sums

    final_acc = correct_predictions_count / total_samples if total_samples > 0 else 0.0

    print(f"Accuracy sur l'intersection des classes (Test Set) : {final_acc:.4f}")
    logging.info(f"Accuracy sur l'intersection des classes (Test Set) : {final_acc:.4f}")

    annot_labels = np.empty_like(cm_counts, dtype=object)
    for i in range(num_train_classes):
        for j in range(num_test_classes):
            count = int(cm_counts[i, j])
            percent = cm_normalized[i, j]

            annot_labels[i, j] = f"{count} / {percent:.1%}"

    # Plot
    plt.figure(figsize=(20, 16))

    # On affiche les pourcentages, mais on peut remettre fmt='g' sur cm_counts si on préfère les entiers bruts
    sns.heatmap(cm_normalized, annot=annot_labels, fmt='', cmap='Blues',
                xticklabels=test_class_names, yticklabels=train_class_names)

    plt.xlabel('Vérité Terrain (Classes du Test set)', fontsize=14)
    plt.ylabel('Prédictions du Modèle (Classes d\'Entraînement)', fontsize=14)
    plt.title(f'Matrice de Confusion Rectangulaire Test Set', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # ÉTAPE 4 : Génération du dictionnaire des erreurs
    # ---------------------------------------------------------
    error_stats = {}

    for t_name in test_class_names:
        sizes_list = misclassified_sizes[t_name]
        count = len(sizes_list)

        if count > 0:
            error_stats[t_name] = {
                "nombre_mal_classifies": count,
                "taille_moyenne": float(np.mean(sizes_list)),
                "taille_min": float(np.min(sizes_list)),
                "taille_max": float(np.max(sizes_list))
            }
        else:
            error_stats[t_name] = {
                "nombre_mal_classifies": 0,
                "taille_moyenne": None,
                "taille_min": None,
                "taille_max": None
            }

    logging.info("\n--- Résumé des erreurs par classe (Test Set) ---")
    for nom, stats in error_stats.items():
        if stats["nombre_mal_classifies"] > 0:
            logging.info(f"Classe Test '{nom}': {stats['nombre_mal_classifies']} erreurs "
                         f"(Taille Moy: {stats['taille_moyenne']:.1f}, "
                         f"Min: {stats['taille_min']}, Max: {stats['taille_max']})")

    return error_stats

def generate_openset_TEST_final_confusion_matrix_mahalanobis(
        model,
        dataloader_train_eval,
        dataloader_valid,
        num_classes,
        col_min_tensor,
        denom_tensor,
        train_class_names,
        test_class_names,
        save_path,
        rejection_rate=0.01,
        unknown_label=-1,
        bias_augmenter=None,
        threshold_coef=1.0 # artificially extend the Mahalanobis frontier of classes
):
    """
        fonction de création d'une matrice de confusion sur les données de test en mode open set , adaptée pour des classes
    aux labels différents entre data de test et data d'entraînement.
    """
    print("--- Génération de la Matrice de Confusion finale Open Set (Mahalanobis) ---")
    logging.info("--- Génération de la Matrice de Confusion finale Open Set (Mahalanobis) ---")

    model.eval()

    # ---------------------------------------------------------
    # ÉTAPE 1 : Extraction d'entraînement et calcul Moyennes/Covariances/Seuils
    # ---------------------------------------------------------
    all_train_features = []
    all_train_labels = []

    with torch.no_grad():
        logging.info("inférence et augmentations par biais pour tracé des frontières de classes")
        for inputs, labels in dataloader_train_eval:
            inputs, labels = inputs.cuda(), labels.squeeze().long().cuda()

            inputs_norm = normalize_batch(inputs, col_min_tensor, denom_tensor, use_log=True)
            features = model(inputs_norm, mode='train')
            all_train_features.append(features)
            all_train_labels.append(labels)

            if bias_augmenter is not None:
                labels_list = labels.cpu().tolist()
                augmented_views = bias_augmenter.generate_views(inputs, labels_list, prob_augm=1.0)

                for aug_inputs in augmented_views:
                    aug_inputs = aug_inputs.cuda()
                    aug_inputs_norm = normalize_batch(aug_inputs, col_min_tensor, denom_tensor, use_log=True)
                    aug_features = model(aug_inputs_norm, mode='train')

                    all_train_features.append(aug_features)
                    all_train_labels.append(labels)

    logging.info("tracé des frontières de classe")
    all_train_features = torch.cat(all_train_features, dim=0)
    all_train_labels = torch.cat(all_train_labels, dim=0)

    # Normalisation L2 vitale pour SupCon
    all_train_features = F.normalize(all_train_features, p=2, dim=1)

    d_model_feat = all_train_features.size(1)

    # Identification des classes connues
    known_classes = torch.unique(all_train_labels).tolist()

    means = torch.zeros(num_classes, d_model_feat, device=all_train_features.device)
    inv_covs = torch.zeros(num_classes, d_model_feat, d_model_feat, device=all_train_features.device)

    epsilon = 1e-4
    eye = torch.eye(d_model_feat, device=all_train_features.device)

    for c in known_classes:
        mask = (all_train_labels == c)
        class_features = all_train_features[mask]

        if class_features.size(0) > 1:
            mu = class_features.mean(dim=0)
            means[c] = mu
            centered = class_features - mu
            cov = (centered.T @ centered) / (class_features.size(0) - 1)
            inv_covs[c] = torch.linalg.pinv(cov + epsilon * eye)
        elif class_features.size(0) == 1:
            means[c] = class_features[0]
            inv_covs[c] = eye
        else:
            inv_covs[c] = eye

    # Fonction utilitaire pour calculer les distances
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

    # Calcul des seuils d'acceptation sur le Train Set
    distances_train = compute_dists(all_train_features)
    thresholds = torch.zeros(num_classes, device=all_train_features.device)

    for c in known_classes:
        mask = (all_train_labels == c)
        if mask.sum() > 0:
            thresholds[c] = torch.quantile(distances_train[mask, c], 1.0 - rejection_rate)
        else:
            thresholds[c] = float('inf')
        # print(f'threshold of class {c} is {thresholds[c]}')

    # ---------------------------------------------------------
    # ÉTAPE 2 : Prédiction sur le set de Test (Avec Rejet)
    # ---------------------------------------------------------
    all_preds = []
    all_true = []

    with torch.no_grad():
        for inputs, labels in dataloader_valid:  # Correspond ici à ton set de Test
            inputs, labels = inputs.cuda(), labels.squeeze().long().cuda()
            inputs_norm = normalize_batch(inputs, col_min_tensor, denom_tensor, use_log=True)

            features = model(inputs_norm, mode="train")
            features = F.normalize(features, p=2, dim=1)

            distances = compute_dists(features)

            valid_distances = distances.clone()
            valid_distances[distances > threshold_coef*thresholds] = float('inf')

            # 1. On prend la distance minimum PARMI LES CLASSES VALIDES
            min_dists, preds = torch.min(valid_distances, dim=1)

            # 2. Si la distance minimum est l'infini, c'est qu'AUCUN rayon ne contenait ce point
            preds[min_dists == float('inf')] = unknown_label

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    # ---------------------------------------------------------
    # ÉTAPE 3 : Création et sauvegarde de la matrice de confusion RECTANGULAIRE
    # ---------------------------------------------------------
    extended_train_class_names = [train_class_names[c] for c in known_classes] + ["Inconnu"]

    num_train_classes = len(known_classes)
    num_test_classes = len(test_class_names)

    # Matrice rectangulaire : Lignes = Train + Inconnu, Colonnes = Test
    cm_counts = np.zeros((num_train_classes + 1, num_test_classes), dtype=int)

    for true_lbl, pred_lbl in zip(all_true, all_preds):
        col = int(true_lbl)  # Colonne = Vérité terrain (Test)

        # Ligne = Prédiction (Train ou Inconnu)
        if pred_lbl == unknown_label:
            row = num_train_classes  # L'index de la dernière ligne ("Inconnu")
        else:
            row = known_classes.index(int(pred_lbl))

        cm_counts[row, col] += 1

    # Normalisation par colonnes (Vérité Terrain) pour avoir des %
    # Attention aux colonnes vides (division par zéro)
    col_sums = cm_counts.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    cm_normalized = cm_counts / col_sums

    annot_labels = np.empty_like(cm_counts, dtype=object)
    for i in range(num_train_classes + 1):
        for j in range(num_test_classes):
            count = int(cm_counts[i, j])
            percent = cm_normalized[i, j]

            annot_labels[i, j] = f"{count} / {percent:.1%}"

    plt.figure(figsize=(20, 16))

    # Affichage avec Seaborn du compte exact (fmt='d')
    sns.heatmap(cm_normalized, annot=annot_labels, fmt='', cmap='Blues',
                xticklabels=test_class_names, yticklabels=extended_train_class_names)

    plt.xlabel('Vérité Terrain (Classes du Test set)', fontsize=14)
    plt.ylabel('Prédictions du Modèle (Classes Train + Inconnu)', fontsize=14)
    plt.title('Matrice de Confusion Open Set Rectangulaire', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Matrice de confusion rectangulaire sauvegardée sous : {save_path}")
    logging.info(f"Matrice de confusion rectangulaire sauvegardée sous : {save_path}")

def generate_openset_final_confusion_matrix_mahalanobis(
        model,
        dataloader_train_eval,
        dataloader_valid,
        num_classes,
        col_min_tensor,
        denom_tensor,
        class_names,
        save_path,
        rejection_rate=0.01,
        unknown_label=-1
):
    """
    Matrice de confusion sur les données de validation en mode open set
    """
    print("--- Génération de la Matrice de Confusion finale Open Set (Mahalanobis) ---")
    logging.info("--- Génération de la Matrice de Confusion finale Open Set (Mahalanobis) ---")

    model.eval()

    # ---------------------------------------------------------
    # ÉTAPE 1 : Extraction d'entraînement et calcul Moyennes/Covariances/Seuils
    # ---------------------------------------------------------
    all_train_features = []
    all_train_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader_train_eval:
            inputs, labels = inputs.cuda(), labels.squeeze().long().cuda()
            inputs_norm = normalize_batch(inputs, col_min_tensor, denom_tensor, use_log=True)

            features = model(inputs_norm, mode="train")
            all_train_features.append(features)
            all_train_labels.append(labels)

    all_train_features = torch.cat(all_train_features, dim=0)
    all_train_labels = torch.cat(all_train_labels, dim=0)

    # --- MODIF : Normalisation L2 vitale pour SupCon ---
    all_train_features = F.normalize(all_train_features, p=2, dim=1)

    d_model_feat = all_train_features.size(1)

    # --- MODIF : Identification des classes connues ---
    known_classes = torch.unique(all_train_labels).tolist()

    means = torch.zeros(num_classes, d_model_feat, device=all_train_features.device)
    inv_covs = torch.zeros(num_classes, d_model_feat, d_model_feat, device=all_train_features.device)

    epsilon = 1e-4
    eye = torch.eye(d_model_feat, device=all_train_features.device)

    for c in known_classes:
        mask = (all_train_labels == c)
        class_features = all_train_features[mask]

        if class_features.size(0) > 1:
            mu = class_features.mean(dim=0)
            means[c] = mu
            centered = class_features - mu
            cov = (centered.T @ centered) / (class_features.size(0) - 1)
            inv_covs[c] = torch.linalg.pinv(cov + epsilon * eye)
        elif class_features.size(0) == 1:
            means[c] = class_features[0]
            inv_covs[c] = eye
        else:
            inv_covs[c] = eye

    # Fonction utilitaire pour calculer les distances
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

    # --- MODIF : Calcul des seuils d'acceptation sur le Train Set ---
    distances_train = compute_dists(all_train_features)
    thresholds = torch.zeros(num_classes, device=all_train_features.device)

    for c in known_classes:
        mask = (all_train_labels == c)
        if mask.sum() > 0:
            thresholds[c] = torch.quantile(distances_train[mask, c], 1.0 - rejection_rate)
        else:
            thresholds[c] = float('inf')

    # ---------------------------------------------------------
    # ÉTAPE 2 : Prédiction sur le set de validation (Avec Rejet)
    # ---------------------------------------------------------
    all_preds = []
    all_true = []

    # --- MODIF : Initialiser les erreurs avec la classe Inconnue (-1) ---
    all_eval_labels = known_classes + [unknown_label]
    misclassified_sizes = {lbl: [] for lbl in all_eval_labels}

    with torch.no_grad():
        for inputs, labels in dataloader_valid:
            inputs, labels = inputs.cuda(), labels.squeeze().long().cuda()
            inputs_norm = normalize_batch(inputs, col_min_tensor, denom_tensor, use_log=True)

            features = model(inputs_norm, mode="train")
            # --- MODIF : Normalisation L2 ---
            features = F.normalize(features, p=2, dim=1)

            distances = compute_dists(features)

            valid_distances = distances.clone()
            valid_distances[distances > thresholds] = float('inf')

            min_dists, preds = torch.min(valid_distances, dim=1)
            preds[min_dists == float('inf')] = unknown_label

            preds_cpu = preds.cpu().numpy()
            true_cpu = labels.cpu().numpy()

            # Récupération des tailles (Ton code exact)
            _, lengths_tensor = pad_packed_sequence(inputs, batch_first=True)
            sizes = lengths_tensor.cpu().numpy()

            N = features.size(0)
            for i in range(N):
                t_label = true_cpu[i]
                p_label = preds_cpu[i]
                if t_label != p_label:
                    misclassified_sizes[t_label].append(float(sizes[i]))

            all_preds.extend(preds_cpu)
            all_true.extend(true_cpu)

    # ---------------------------------------------------------
    # ÉTAPE 3 : Création et sauvegarde de la matrice de confusion
    # ---------------------------------------------------------
    # --- MODIF : Ajout de la classe Inconnue pour l'affichage ---
    extended_class_names = [class_names[c] for c in known_classes] + ["Inconnu"]

    cm = confusion_matrix(all_true, all_preds, labels=all_eval_labels, normalize='true')
    final_acc = np.mean(np.array(all_true) == np.array(all_preds))

    print(f"Accuracy finale sur la Validation (Open Set Mahalanobis) : {final_acc:.4f}")
    logging.info(f"Accuracy finale sur la Validation (Open Set Mahalanobis) : {final_acc:.4f}")

    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='.0%', cmap='Blues', xticklabels=extended_class_names,
                yticklabels=extended_class_names)
    plt.xlabel('Prédictions du Modèle (Avec Rejet)', fontsize=12)
    plt.ylabel('Vrais Labels (Inclus classes rares)', fontsize=12)
    plt.title(f'Matrice de Confusion Open Set (Acc: {final_acc:.2%})', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Matrice de confusion sauvegardée sous : {save_path}")
    logging.info(f"Matrice de confusion sauvegardée sous : {save_path}")

    # ---------------------------------------------------------
    # ÉTAPE 4 : Génération du dictionnaire des erreurs
    # ---------------------------------------------------------
    error_stats = {}

    # --- MODIF : On itère sur toutes les classes + l'Inconnu ---
    for lbl, name in zip(all_eval_labels, extended_class_names):
        sizes_list = misclassified_sizes[lbl]
        count = len(sizes_list)

        if count > 0:
            error_stats[name] = {
                "nombre_mal_classifies": count,
                "taille_moyenne": float(np.mean(sizes_list)),
                "taille_min": float(np.min(sizes_list)),
                "taille_max": float(np.max(sizes_list))
            }
        else:
            error_stats[name] = {
                "nombre_mal_classifies": 0,
                "taille_moyenne": None,
                "taille_min": None,
                "taille_max": None
            }

    logging.info("\n--- Résumé des erreurs par classe ---")
    for nom, stats in error_stats.items():
        if stats["nombre_mal_classifies"] > 0:
            logging.info(f"Classe '{nom}': {stats['nombre_mal_classifies']} erreurs "
                         f"(Taille Moy: {stats['taille_moyenne']:.1f}, "
                         f"Min: {stats['taille_min']}, Max: {stats['taille_max']})")


def generate_final_confusion_matrix_mahalanobis(
    model, 
    dataloader_train_eval, 
    dataloader_valid, 
    num_classes, 
    col_min_tensor, 
    denom_tensor,
    class_names,
    save_path
):
    """
    Matrice de confusion sur les données de validation en mode close set
    """
    print("--- Génération de la Matrice de Confusion finale (Mahalanobis) ---")
    logging.info("--- Génération de la Matrice de Confusion finale (Mahalanobis) ---")

    model.eval()

    # ---------------------------------------------------------
    # ÉTAPE 1 : Extraction d'entraînement et calcul Moyennes/Covariances
    # ---------------------------------------------------------
    all_train_features = []
    all_train_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader_train_eval:
            inputs, labels = inputs.cuda(), labels.squeeze().long().cuda()
            inputs_norm = normalize_batch(inputs, col_min_tensor, denom_tensor, use_log=True)
            
            # Mode extract pour avoir les features brutes du backbone
            features = model(inputs_norm, mode="train") 
            
            all_train_features.append(features)
            all_train_labels.append(labels)

    all_train_features = torch.cat(all_train_features, dim=0)
    all_train_labels = torch.cat(all_train_labels, dim=0)

    d_model_feat = all_train_features.size(1)

    means = torch.zeros(num_classes, d_model_feat, device=all_train_features.device)
    inv_covs = torch.zeros(num_classes, d_model_feat, d_model_feat, device=all_train_features.device)
    
    epsilon = 1e-4
    eye = torch.eye(d_model_feat, device=all_train_features.device)

    for c in range(num_classes):
        mask = (all_train_labels == c)
        class_features = all_train_features[mask]
        
        if class_features.size(0) > 1:
            mu = class_features.mean(dim=0)
            means[c] = mu
            centered = class_features - mu
            cov = (centered.T @ centered) / (class_features.size(0) - 1)
            inv_cov = torch.linalg.pinv(cov + epsilon * eye)
            inv_covs[c] = inv_cov
        elif class_features.size(0) == 1:
            means[c] = class_features[0]
            inv_covs[c] = eye
        else:
            inv_covs[c] = eye

    # ---------------------------------------------------------
    # ÉTAPE 2 : Prédiction sur le set de validation
    # ---------------------------------------------------------
    all_preds = []
    all_true = []

    misclassified_sizes = {c: [] for c in range(num_classes)}

    with torch.no_grad():
        for inputs, labels in dataloader_valid:
            inputs, labels = inputs.cuda(), labels.squeeze().long().cuda()
            inputs_norm = normalize_batch(inputs, col_min_tensor, denom_tensor, use_log=True)
            
            features = model(inputs_norm, mode="train")
            
            N = features.size(0)
            distances = torch.zeros(N, num_classes, device=features.device)
            
            for c in range(num_classes):
                diff = features - means[c]
                left_term = torch.matmul(diff, inv_covs[c])
                dist_sq = (left_term * diff).sum(dim=1)
                distances[:, c] = dist_sq
                
            preds = torch.argmin(distances, dim=1)

            # analyse des erreurs et récupération des tailles
            preds_cpu = preds.cpu().numpy()
            true_cpu = labels.cpu().numpy()

            _, lengths_tensor = pad_packed_sequence(inputs, batch_first=True)
            sizes = lengths_tensor.cpu().numpy()

            for i in range(N):
                t_label = true_cpu[i]
                p_label = preds_cpu[i]
                if t_label != p_label:
                    misclassified_sizes[t_label].append(float(sizes[i]))

            all_preds.extend(preds.cpu().numpy())
            all_true.extend(labels.cpu().numpy())

    # ---------------------------------------------------------
    # ÉTAPE 3 : Création et sauvegarde de la matrice de confusion
    # ---------------------------------------------------------
    cm = confusion_matrix(all_true, all_preds, normalize='true')
    final_acc = np.mean(np.array(all_true) == np.array(all_preds))
    #class_accuracies = cm.diagonal() / (cm.sum(axis=1) + 1e-9)
    #final_acc = np.mean(class_accuracies)
    
    print(f"Accuracy finale sur la Validation (Mahalanobis) : {final_acc:.4f}")
    logging.info(f"Accuracy finale sur la Validation (Mahalanobis) : {final_acc:.4f}")

    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='.0%', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prédictions du Modèle (Mahalanobis)', fontsize=12)
    plt.ylabel('Vrais Labels', fontsize=12)
    plt.title(f'Matrice de Confusion sur Validation (Acc: {final_acc:.2%})', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    print(f"Matrice de confusion sauvegardée sous : {save_path}")
    logging.info(f"Matrice de confusion sauvegardée sous : {save_path}")

    # ---------------------------------------------------------
    # NOUVEAU : ÉTAPE 4 : Génération du dictionnaire des erreurs
    # ---------------------------------------------------------
    error_stats = {}

    for c in range(num_classes):
        c_name = class_names[c]
        sizes_list = misclassified_sizes[c]
        count = len(sizes_list)

        if count > 0:
            error_stats[c_name] = {
                "nombre_mal_classifies": count,
                "taille_moyenne": float(np.mean(sizes_list)),
                "taille_min": float(np.min(sizes_list)),
                "taille_max": float(np.max(sizes_list))
            }
        else:
            # Si aucune erreur pour cette classe
            error_stats[c_name] = {
                "nombre_mal_classifies": 0,
                "taille_moyenne": None,
                "taille_min": None,
                "taille_max": None
            }

    # Optionnel : Afficher un petit résumé dans la console
    logging.info("\n--- Résumé des erreurs par classe ---")
    for nom, stats in error_stats.items():
        if stats["nombre_mal_classifies"] > 0:
            logging.info(f"Classe '{nom}': {stats['nombre_mal_classifies']} erreurs "
                  f"(Taille Moy: {stats['taille_moyenne']:.1f}, "
                  f"Min: {stats['taille_min']}, Max: {stats['taille_max']})")


# Fonction d'extraction des features après la phase 1 de l'entrainement contrastif
def extract_features(model, dataloader, col_min_tensor, denom_tensor):
    model.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.cuda()
            labels = labels.squeeze().long().cuda()
            
            inputs_norm = normalize_batch(inputs, col_min_tensor, denom_tensor, use_log=True)

            # On récupère les vecteurs de taille 256
            features = model(inputs_norm, mode="extract")
            
            # On renvoie sur le CPU pour le stockage
            features_list.append(features.cpu())
            labels_list.append(labels.squeeze().long().cpu())
            
    return torch.cat(features_list), torch.cat(labels_list)

def evaluate_prototypes(model, dataloader_train_p2, dataloader_valid, num_classes, col_min_tensor, denom_tensor, mode="extract"):
    """
    Calcule les prototypes sur le Train Set
    et évalue l'Accuracy selon la Distance de Mahalanobis (qui prend en compte la covariance).
    """
    model.eval()
    
    # ---------------------------------------------------------
    # Collecte des features d'entraînement
    # ---------------------------------------------------------
    all_train_features = []
    all_train_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader_train_p2: 
            inputs, labels = inputs.cuda(), labels.squeeze().long().cuda()
            
            # Si tu utilises ta fonction normalize_batch
            inputs_norm = normalize_batch(inputs, col_min_tensor, denom_tensor, use_log=True)
            features = model(inputs_norm, mode=mode) 
            
            all_train_features.append(features)
            all_train_labels.append(labels)
            
    all_train_features = torch.cat(all_train_features, dim=0)
    all_train_labels = torch.cat(all_train_labels, dim=0)
    
    d_model = all_train_features.size(1)
    
    # ---------------------------------------------------------
    # Calcul des centres (Moyennes) et Matrices de Covariance Inverse
    # ---------------------------------------------------------
    means = torch.zeros(num_classes, d_model, device=all_train_features.device)
    inv_covs = torch.zeros(num_classes, d_model, d_model, device=all_train_features.device)
    
    # Terme de régularisation pour éviter les matrices singulières (Division par zéro / Instabilité)
    epsilon = 1e-4 
    eye = torch.eye(d_model, device=all_train_features.device)

    for c in range(num_classes):
        mask = (all_train_labels == c)
        class_features = all_train_features[mask]
        
        if class_features.size(0) > 1:
            # Vecteur moyen
            mu = class_features.mean(dim=0)
            means[c] = mu
            
            # Centrer les données
            centered = class_features - mu
            
            # Calcul de la matrice de covariance: (X^T * X) / (N - 1)
            cov = (centered.T @ centered) / (class_features.size(0) - 1)
            
            # Inverse de covariance robuste avec régularisation
            inv_cov = torch.linalg.pinv(cov + epsilon * eye)
            inv_covs[c] = inv_cov
            
        elif class_features.size(0) == 1:
            # Sécurité si une seule donnée dans la classe (pas de covariance possible)
            means[c] = class_features[0]
            inv_covs[c] = eye
        else:
            # Sécurité si la classe est vide
            inv_covs[c] = eye

    # Fonction locale pour calculer la distance pour un batch
    def compute_mahalanobis(features):
        N = features.size(0)
        distances = torch.zeros(N, num_classes, device=features.device)
        
        for c in range(num_classes):
            # Différence au centre : (x - mu)
            diff = features - means[c] # [N, D]
            
            # Produit matriciel : (x - mu) @ Sigma^-1
            left_term = torch.matmul(diff, inv_covs[c]) # [N, D]
            
            # Produit scalaire élément par élément avec (x - mu)^T, puis somme sur D
            # Cela équivaut à la diagonale de (diff @ inv_cov @ diff^T) mais en beaucoup plus rapide (O(N) au lieu de O(N^2))
            dist_sq = (left_term * diff).sum(dim=1) 
            distances[:, c] = dist_sq
            
        return distances

    # ---------------------------------------------------------
    # Évaluation sur le Train Set
    # ---------------------------------------------------------
    distances_train = compute_mahalanobis(all_train_features)
    preds_train = torch.argmin(distances_train, dim=1) # On prend la classe avec la distance MINIMALE
    train_proto_acc = (preds_train == all_train_labels).float().mean().item()
    
    # ---------------------------------------------------------
    # Évaluation sur le Valid Set
    # ---------------------------------------------------------
    all_valid_features = []
    all_valid_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader_valid:
            inputs, labels = inputs.cuda(), labels.squeeze().long().cuda()
            inputs_norm = normalize_batch(inputs, col_min_tensor, denom_tensor, use_log=True)
            
            features = model(inputs_norm, mode=mode)
            
            all_valid_features.append(features)
            all_valid_labels.append(labels)
            
    all_valid_features = torch.cat(all_valid_features, dim=0)
    all_valid_labels = torch.cat(all_valid_labels, dim=0)
    
    distances_valid = compute_mahalanobis(all_valid_features)
    preds_valid = torch.argmin(distances_valid, dim=1)
    valid_proto_acc = (preds_valid == all_valid_labels).float().mean().item()
    
    return train_proto_acc, valid_proto_acc


def evaluate_openset_prototypes_per_class(model, dataloader_train_p2, dataloader_valid, num_classes, col_min_tensor,
                                          denom_tensor, mode="extract", class_names=None, rejection_rate=0.05,
                                          unknown_label=-1):
    """
    Calcule les prototypes sur le Train Set (uniquement pour les classes connues),
    détermine les seuils d'exclusion, et évalue l'Accuracy (Open Set)
    selon la Distance de Mahalanobis.
    """
    model.eval()

    # ---------------------------------------------------------
    # 1. Collecte des features d'entraînement
    # ---------------------------------------------------------
    all_train_features = []
    all_train_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader_train_p2:
            inputs, labels = inputs.cuda(), labels.squeeze().long().cuda()

            inputs_norm = normalize_batch(inputs, col_min_tensor, denom_tensor, use_log=True)
            features = model(inputs_norm, mode=mode)

            all_train_features.append(features)
            all_train_labels.append(labels)

    all_train_features = torch.cat(all_train_features, dim=0)
    all_train_labels = torch.cat(all_train_labels, dim=0)

    d_model = all_train_features.size(1)

    # --- MODIF OPEN SET : Identifier les classes présentes dans le Train Set ---
    known_classes = torch.unique(all_train_labels).tolist()

    # ---------------------------------------------------------
    # 2. Calcul des centres (Moyennes) et Matrices de Covariance Inverse
    # ---------------------------------------------------------
    means = torch.zeros(num_classes, d_model, device=all_train_features.device)
    inv_covs = torch.zeros(num_classes, d_model, d_model, device=all_train_features.device)

    epsilon = 1e-4
    eye = torch.eye(d_model, device=all_train_features.device)

    # On ne calcule que pour les classes connues
    for c in known_classes:
        mask = (all_train_labels == c)
        class_features = all_train_features[mask]

        if class_features.size(0) > 1:
            mu = class_features.mean(dim=0)
            means[c] = mu

            centered = class_features - mu
            cov = (centered.T @ centered) / (class_features.size(0) - 1)
            inv_covs[c] = torch.linalg.pinv(cov + epsilon * eye)

        elif class_features.size(0) == 1:
            means[c] = class_features[0]
            inv_covs[c] = eye

    def compute_mahalanobis(features):
        N = features.size(0)
        distances = torch.zeros(N, num_classes, device=features.device)

        for c in known_classes:
            diff = features - means[c]  # [N, D]
            left_term = torch.matmul(diff, inv_covs[c])  # [N, D]
            dist_sq = (left_term * diff).sum(dim=1)
            distances[:, c] = dist_sq

        # Les classes non-connues (exclues) reçoivent une distance infinie
        # pour ne jamais être prédites par l'argmin
        for c in range(num_classes):
            if c not in known_classes:
                distances[:, c] = float('inf')

        return distances

    # Fonction pour calculer l'accuracy par classe (Modifiée pour inclure la classe Inconnue)
    def compute_per_class_accuracy(preds, labels):
        stats = {}
        # Évaluation des classes normales (0 à num_classes-1)
        for c in range(num_classes):
            mask = (labels == c)
            total_c = mask.sum().item()
            if total_c > 0:
                correct_c = (preds[mask] == c).sum().item()
                acc = correct_c / total_c
            else:
                acc = None

            nom_classe = class_names[c] if class_names is not None else f"Classe_{c}"
            stats[c] = {'name': nom_classe, 'accuracy': acc, 'count': total_c}

        # Évaluation spécifique de la classe Inconnue (-1)
        mask_unknown = (labels == unknown_label)
        total_unknown = mask_unknown.sum().item()
        if total_unknown > 0:
            correct_unknown = (preds[mask_unknown] == unknown_label).sum().item()
            acc_unknown = correct_unknown / total_unknown
        else:
            acc_unknown = None

        stats[unknown_label] = {
            'name': "Inconnu (Open Set)",
            'accuracy': acc_unknown,
            'count': total_unknown
        }

        return stats

    # ---------------------------------------------------------
    # 3. Évaluation sur le Train Set ET Définition des Seuils
    # ---------------------------------------------------------
    distances_train = compute_mahalanobis(all_train_features)
    preds_train = torch.argmin(distances_train, dim=1)
    train_proto_acc = (preds_train == all_train_labels).float().mean().item()
    train_class_accs = compute_per_class_accuracy(preds_train, all_train_labels)

    # --- MODIF OPEN SET : Calcul des seuils d'acceptation par classe ---
    thresholds = torch.zeros(num_classes, device=all_train_features.device)
    for c in known_classes:
        mask = (all_train_labels == c)
        if mask.sum() > 0:
            dists_to_c = distances_train[mask, c]
            # On prend la distance qui couvre (1 - rejection_rate) des données d'entraînement (ex: 95%)
            thresholds[c] = torch.quantile(dists_to_c, 1.0 - rejection_rate)
        else:
            thresholds[c] = float('inf')

    # ---------------------------------------------------------
    # 4. Évaluation sur le Valid Set (Open Set)
    # ---------------------------------------------------------
    all_valid_features = []
    all_valid_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader_valid:
            inputs, labels = inputs.cuda(), labels.squeeze().long().cuda()
            inputs_norm = normalize_batch(inputs, col_min_tensor, denom_tensor, use_log=True)

            features = model(inputs_norm, mode=mode)

            all_valid_features.append(features)
            all_valid_labels.append(labels)

    all_valid_features = torch.cat(all_valid_features, dim=0)
    all_valid_labels = torch.cat(all_valid_labels, dim=0)

    # Calcul des distances sur le set de validation
    distances_valid = compute_mahalanobis(all_valid_features)

    valid_distances = distances_valid.clone()
    valid_distances[distances_valid > thresholds] = float('inf')

    min_dist_valid, preds_valid = torch.min(valid_distances, dim=1)
    preds_valid[min_dist_valid == float('inf')] = unknown_label

    valid_proto_acc = (preds_valid == all_valid_labels).float().mean().item()
    valid_class_accs = compute_per_class_accuracy(preds_valid, all_valid_labels)

    return train_proto_acc, valid_proto_acc, train_class_accs, valid_class_accs


class BalancedBatchSamplerNoReuse(Sampler):
    """
    Batch sampler équilibré pour le Contrastive Learning (SupCon).
    Garantit `n_classes_per_batch` classes et `n_samples_per_class` exemples par classe.
    """
    def __init__(self, dataset, n_classes_per_batch, n_samples_per_class, drop_last=True):
        self.dataset = dataset
        self.n_classes_per_batch = n_classes_per_batch
        self.n_samples_per_class = n_samples_per_class
        self.drop_last = drop_last
        self.batch_size = n_classes_per_batch * n_samples_per_class

        self.label_to_indices = defaultdict(list)
        
        print("Initialisation du Sampler : Scan des labels en cours...")
        # On itère sur le dataset pour récupérer les labels
        for idx in range(len(dataset)):
            sample = dataset[idx]
            if isinstance(sample, dict):
                label = sample["label"]
            else:
                label = sample[1]
                
            # Gérer le cas où le label est un tenseur PyTorch
            if torch.is_tensor(label):
                label = label.item()
                
            self.label_to_indices[label].append(idx)

        self.labels = sorted(list(self.label_to_indices.keys()))

        if len(self.labels) < self.n_classes_per_batch:
            raise ValueError(f"Pas assez de classes: {len(self.labels)} < {self.n_classes_per_batch}")

        # Calcul du nombre de paquets possibles par classe
        self.class_chunks = {
            label: len(indices) // self.n_samples_per_class
            for label, indices in self.label_to_indices.items()
        }

        # Calcul du nombre total de batchs théoriques
        total_class_contrib = sum(self.class_chunks.values())
        self.n_batches = total_class_contrib // self.n_classes_per_batch
        print(f"Sampler prêt : {self.n_batches} batchs par epoch (Taille du batch: {self.batch_size}).")

    def __iter__(self):
        # 1. On copie et on mélange les index pour chaque classe à chaque nouvel epoch
        class_indices_copy = {}
        for label, indices in self.label_to_indices.items():
            indices_copy = indices.copy()
            random.shuffle(indices_copy)
            
            # Découpage en paquets de taille n_samples_per_class
            chunks = [indices_copy[i : i + self.n_samples_per_class] 
                      for i in range(0, len(indices_copy), self.n_samples_per_class)]
            
            if self.drop_last and len(chunks) > 0 and len(chunks[-1]) < self.n_samples_per_class:
                chunks.pop()
                
            if chunks:
                class_indices_copy[label] = chunks

        # 2. Construction des batchs
        for _ in range(self.n_batches):
            # On filtre les classes qui n'ont plus de paquets disponibles
            available_classes = [label for label, chunks in class_indices_copy.items() if len(chunks) > 0]
            
            # Condition d'arrêt de l'epoch : si on ne peut plus remplir les classes demandées
            if len(available_classes) < self.n_classes_per_batch:
                break
                
            # Tirage aléatoire des classes pour ce batch
            selected_classes = random.sample(available_classes, self.n_classes_per_batch)
            
            batch_indices = []
            for cls in selected_classes:
                # On retire un paquet de cette classe et on l'ajoute au batch
                batch_indices.extend(class_indices_copy[cls].pop(0))
                
            # On mélange les éléments au sein du batch (pour que la SupCon ne reçoive pas les classes dans l'ordre)
            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return self.n_batches


def normalize_batch(batch, col_min, denom, use_log=True):
    """
    Normalise un batch sur le GPU. Compatible avec les PackedSequence.
    """
    epsilon = 1e-9
    
    if isinstance(batch, torch.nn.utils.rnn.PackedSequence):
        # Extraction du tenseur 2D [Total_pulses, 3]
        x = batch.data.clone()
        
        if use_log:
            x[:, 0] = torch.log10(x[:, 0] + epsilon)
            x[:, 2] = torch.log10(x[:, 2] + epsilon)
            
        # Normalisation Min-Max vectorisée
        x_norm = (x - col_min) / denom
        x_norm = (2.0 * x_norm) - 1.0
        x_norm = torch.clamp(x_norm, min=-1.0, max=1.0)
        
        # Reconstruction
        return torch.nn.utils.rnn.PackedSequence(x_norm, batch.batch_sizes, batch.sorted_indices, batch.unsorted_indices)
        
    else:
        # Tenseur 3D standard [Batch, Seq, 3]
        x = batch.clone()
        
        if use_log:
            x[:, :, 0] = torch.log10(x[:, :, 0] + epsilon)
            x[:, :, 2] = torch.log10(x[:, :, 2] + epsilon)
            
        x_norm = (x - col_min) / denom
        x_norm = (2.0 * x_norm) - 1.0
        
        return torch.clamp(x_norm, min=-1.0, max=1.0)

def load_database(chemin_database, min_length=0.0):
    """
    Fonction pour charger la base de données à partir des fichiers .npz.
    """
    liste_fichiers = sorted(chemin_database.rglob('*.npz'))

    dataset = []
    labelset = []

    for fichier in liste_fichiers:
        logging.info(f"Chargement de {fichier.name}")

        # Chargement du fichier .npz
        data = np.load(fichier, allow_pickle=True)

        try:
            if 'pdw' in data:
                pdw_dict = data['pdw'].item()
                data_toa = pdw_dict['liste_dtoa']
                data_li = pdw_dict['liste_li']
                data_freq = pdw_dict['liste_freq']
            else:
                data_toa = data['liste_dtoa']
                data_li = data['liste_li']
                data_freq = data['liste_freq']

        except KeyError as e:
            logging.info(f"Erreur de clé {e} dans {fichier.name}, clés disponibles : {list(data.keys())}")
            continue

        label = fichier.name.removesuffix('_PDW.npz')
        nombre_pdw = len(data_toa)
        if len(data_toa) > min_length:
            logging.info(f"Nombre de PDW : {nombre_pdw}")
            tmp = np.stack([data_toa, data_freq, data_li], axis=1)
            dataset.append(tmp)
            labelset.append(label)
        else:
            logging.info(f"Le fichier {fichier.name} a {len(data_toa)} < {min_length} PDW")

    classes_uniques, comptes = np.unique(labelset, return_counts=True)
    nombre_total_classes = len(classes_uniques)

    logging.info("--- Résumé de la base de données ---")
    logging.info(f"Nombre total de classes uniques : {nombre_total_classes}")

    logging.info("------------------------------------")

    return dataset, labelset