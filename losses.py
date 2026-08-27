"""
Last updated on 26/08/2026

@author : Foucault BERNARD

Ce fichier contient les différentes loss implémentées pour l'apprentissage contrastif : loss SupCon, Decoupled loss SupCon 
et Decoupled loss SupCon Mahalanobis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss (Adaptation de Khosla et al. 2020)
    Formulée mathématiquement pour calculer l'attraction/répulsion sur tout le batch.
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: Tenseur de forme [batch_size, projection_dim] (normalisé L2)
        labels: Tenseur de forme [batch_size]
        """
        device = features.device
        batch_size = features.shape[0]

        # Création de la matrice d'adjacence des labels (batch_size x batch_size)
        # mask[i, j] = 1 si la séquence i et j ont la même classe, 0 sinon.
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Calcul de la matrice de similarité (produit scalaire des vecteurs normalisés = Cosinus)
        # similarity[i, j] = cos_sim(feature_i, feature_j)
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        # Pour la stabilité numérique (trick mathématique du softmax)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # On annule la diagonale du masque (on ne compare pas un élément avec lui-même)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        # Calcul des probabilités exponentielles
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Moyenne sur les éléments positifs uniquement
        counts = mask.sum(1)
        counts = torch.where(counts == 0, torch.ones_like(counts), counts) # si le compte est 0, on le remplace par 1 pour éviter la division par zéro
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / counts

        # La perte finale
        loss = - mean_log_prob_pos
        loss = loss.mean()

        return loss
    

class DecoupledSupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss (Adaptation de Khosla et al. 2020)
    Transformée en retirant les autres positifs du dénominateur (Decoupled)
    ET en appliquant un coefficient (coef_neg) pour accentuer la répulsion 
    des négatifs, tout en préservant la stabilité numérique.
    """
    def __init__(self, temperature=0.07, coef_neg=1.0):
        super(DecoupledSupConLoss, self).__init__()
        self.temperature = temperature
        self.coef_neg = coef_neg

    def forward(self, features, labels):
        """
        features: Tenseur de forme [batch_size, projection_dim] (normalisé L2)
        labels: Tenseur de forme [batch_size]
        """
        device = features.device
        batch_size = features.shape[0]

        # Création de la matrice d'adjacence des labels
        labels = labels.contiguous().view(-1, 1)
        mask_with_diag = torch.eq(labels, labels.T).float().to(device)

        # Création des masques (positifs et négatifs purs)
        logits_mask = torch.scatter(
            torch.ones_like(mask_with_diag), 1, torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask_with_diag * logits_mask  # Positifs sans la diagonale
        neg_mask = logits_mask - mask        # Négatifs uniquement

        # Calcul de la matrice de similarité brute
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        # Application du coefficient pour les négatifs
        logits_adjusted = anchor_dot_contrast * (1 - neg_mask) + (anchor_dot_contrast * self.coef_neg) * neg_mask

        # Stabilité numérique (trick du softmax) basée sur les logits ajustés
        logits_max, _ = torch.max(logits_adjusted, dim=1, keepdim=True)
        logits_stable = logits_adjusted - logits_max.detach()

        # Exponentielles
        exp_logits = torch.exp(logits_stable)

        # Création du dénominateur dynamique
        # exp_pos isole la paire évaluée, sum_neg regroupe toutes les classes différentes
        exp_pos = exp_logits * mask
        sum_neg = (exp_logits * neg_mask).sum(dim=1, keepdim=True)
        
        denominator = exp_pos + sum_neg

        # Calcul de la log-probabilité avec sécurité numérique (+ 1e-9)
        log_prob = logits_stable - torch.log(denominator + 1e-9)
        
        # Moyenne sur les éléments positifs uniquement
        counts = mask.sum(1)
        counts = torch.where(counts == 0, torch.ones_like(counts), counts)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / counts

        # Perte finale
        loss = - mean_log_prob_pos
        loss = loss.mean()

        return loss
    

class DecoupledSupConLossMahalanobis(nn.Module):
    """
    Supervised Contrastive Learning Loss
    - Decoupled : Retrait des autres positifs du dénominateur.
    - Mahalanobis : Remplace la similarité cosinus (convertie via S = -D) par la distance de Mahalanobis
    Cette loss ne donne pas de résultats concluants.
    """
    def __init__(self, temperature=0.07, coef_neg=1.0, eps=1e-4):
        super(DecoupledSupConLossMahalanobis, self).__init__()
        self.temperature = temperature
        self.coef_neg = coef_neg
        self.eps = eps

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        dim = features.shape[1]

        labels = labels.contiguous().view(-1, 1)
        mask_with_diag = torch.eq(labels, labels.T).float().to(device)

        logits_mask = torch.scatter(
            torch.ones_like(mask_with_diag), 1, torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask_with_diag * logits_mask  
        neg_mask = logits_mask - mask        

        P_batch = torch.zeros((batch_size, dim, dim), device=device)
        sqrt_min_eig_batch = torch.zeros((batch_size,), device=device)
        
        unique_classes = torch.unique(labels)
        
        for c in unique_classes:
            mask_c = (labels.view(-1) == c)
            x_c = features[mask_c]
            
            if x_c.shape[0] > 1:
                # Covariance
                cov_c = torch.cov(x_c.T)
                # Régularisation
                cov_c = cov_c + self.eps * torch.eye(dim, device=device)
                # Inversion
                prec_c = torch.linalg.pinv(cov_c)
                # Valeurs propres
                eigvals = torch.linalg.eigvalsh(cov_c)
                min_eig = torch.clamp(eigvals[0], min=0.0)
                
            else:
                prec_c = torch.eye(dim, device=device)
                min_eig = torch.tensor(1.0, device=device) 
            
            # Mapping dans les tenseurs batch
            P_batch[mask_c] = prec_c
            sqrt_min_eig_batch[mask_c] = torch.sqrt(min_eig)

        # Calcul vectorisé de la distance de Mahalanobis au carré
        diff = features.unsqueeze(1) - features.unsqueeze(0)
        dist_sq = torch.einsum('bjd, bdk, bjk -> bj', diff, P_batch, diff)

        # --- BORNER LA DISTANCE ---
        dist = torch.sqrt(torch.clamp(dist_sq, min=1e-9))
        
        # Multiplication par la racine de la VP
        scaled_dist = dist * sqrt_min_eig_batch.unsqueeze(1)
         
        # passage au négatif et ajout de 1 pour forcer l'intervalle [-1, 1] et devenir une similarité
        similarity_score = 1.0 - scaled_dist

        # Conversion en logits : le réseau doit minimiser la distance, donc on l'inverse pour le logit
        anchor_dot_contrast = similarity_score / self.temperature

        # Application du coefficient pour les négatifs
        logits_adjusted = anchor_dot_contrast * (1 - neg_mask) + (anchor_dot_contrast * self.coef_neg) * neg_mask

        # Stabilité numérique
        logits_max, _ = torch.max(logits_adjusted, dim=1, keepdim=True)
        logits_stable = logits_adjusted - logits_max.detach()

        exp_logits = torch.exp(logits_stable)

        exp_pos = exp_logits * mask
        sum_neg = (exp_logits * neg_mask).sum(dim=1, keepdim=True)
        
        denominator = exp_pos + sum_neg

        log_prob = logits_stable - torch.log(denominator + 1e-9)
        
        counts = mask.sum(1)
        counts = torch.where(counts == 0, torch.ones_like(counts), counts)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / counts

        loss = - mean_log_prob_pos
        return loss.mean()