"""
Last updated on 26/08/2026

@author : Foucault BERNARD

Ce fichier contient les classes de construction des modèles de NN.
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence

class PositionalEncoding(nn.Module):
    """
    Injecte l'information sur la position (l'ordre) des impulsions dans la séquence.
    """
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Création de la matrice d'encodage positionnel
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        # register_buffer permet de sauvegarder ce tenseur dans le state_dict sans qu'il soit un paramètre appris
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class PDWTransformer(nn.Module):
    """Transformer classique pour la classification de séquences radar PDW"""
    
    def __init__(self, inputs=3, outputs=30, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1):
        super(PDWTransformer, self).__init__()
        
        self.d_model = d_model
        
        # "Word Embedding" : Projette [DTOA, Freq, LI] -> Vecteur de taille d_model
        self.embedding = nn.Linear(inputs, d_model)
        
        # Encodage positionnel
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Token [CLS] pour agréger l'information de toute la séquence
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Cœur du Transformer (L'Encodeur)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True # Indispensable pour avoir les tenseurs en [batch, seq, features]
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classifieur final
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, outputs)
        )

    def forward(self, x):
        padding_mask = None
        
        # --- GESTION DES LONGUEURS VARIABLES ---
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            # Transforme la séquence packée en tenseur paddé [batch, seq_len, 3]
            x, lengths = pad_packed_sequence(x, batch_first=True)
            
            # Création du masque de padding pour le Transformer.
            batch_size, max_seq_len, _ = x.size()
            padding_mask = torch.arange(max_seq_len)[None, :] >= lengths[:, None]
            padding_mask = padding_mask.to(x.device)

        # On s'assure que x est bien au format [batch_size, seq_len, 3]
        if x.size(-1) != 3:
            x = x.permute(0, 2, 1)

        batch_size = x.size(0)
        
        # --- ÉTAPE 1 : Embedding des PDW ---
        # [batch_size, seq_len, 3] -> [batch_size, seq_len, d_model]
        x = self.embedding(x)
        
        # --- ÉTAPE 2 : Ajout du token [CLS] ---
        # On duplique le token CLS pour chaque élément du batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # On le colle tout au début de la séquence (position 0)
        x = torch.cat((cls_tokens, x), dim=1) # Nouvelle shape: [batch_size, seq_len + 1, d_model]
        
        # On met à jour le padding_mask car on a ajouté 1 élément (le CLS) qui n'est JAMAIS du padding (False)
        if padding_mask is not None:
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat((cls_mask, padding_mask), dim=1)
            
        # --- ÉTAPE 3 : Encodage positionnel ---
        x = self.pos_encoder(x)
        
        # --- ÉTAPE 4 : Passage dans l'Encodeur Transformer ---
        # Le masque indique au Transformer de ne pas prêter d'attention aux zéros de remplissage
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # --- ÉTAPE 5 : Classification ---
        # On extrait uniquement le token [CLS] mis à jour par l'attention (indice 0 de la dimension temporelle)
        cls_output = x[:, 0, :] # Shape: [batch_size, d_model]
        
        # Passage dans le réseau de neurones final
        out = self.classifier(cls_output)
        
        return out

class PDWTwoStageTransformer(nn.Module):
    """
    Transformer optimisé pour l'apprentissage en 2 étapes (Two-Stage) :
    - Phase 1 : Supervised Contrastive Learning (Structuration de l'espace via tete de projection)
    - Phase 2 : Linear Probing (Classification avec Backbone gelé via tete de classification)
    """
    def __init__(self, inputs=3, outputs=10, d_model=64, nhead=4, num_layers=3, proj_dim=64, dim_feedforward=256, dropout=0.1):
        super(PDWTwoStageTransformer, self).__init__()
        
        self.embedding = nn.Linear(inputs, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection SupCon
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim)
        )
        
        # Classification
        self.classifier_head = nn.Linear(d_model, outputs)

    def forward(self, x, mode="phase1"):
        """
        mode: 
          - "phase1" : Retourne la projection normalisée L2 pour la SupConLoss
          - "phase2" : Retourne les logits bruts pour la CrossEntropyLoss
        """
        padding_mask = None
        
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x, lengths = pad_packed_sequence(x, batch_first=True)
            
            batch_size, max_seq_len, _ = x.size()
            padding_mask = torch.arange(max_seq_len)[None, :] >= lengths[:, None]
            padding_mask = padding_mask.to(x.device)

        if x.size(-1) != 3:
            x = x.permute(0, 2, 1)

        batch_size = x.size(0)
        
        # Passage dans l'Embedding
        x = self.embedding(x)
        
        # Ajout du token CLS
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Mise à jour du masque pour inclure le token CLS
        if padding_mask is not None:
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat((cls_mask, padding_mask), dim=1)
            
        # Positional Encoding et Transformer Encodeur
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Extraction de la représentation latente (le token CLS)
        features = x[:, 0, :]
        
        if mode == "phase1":
            proj = self.projection_head(features)
            proj = F.normalize(proj, p=2, dim=1)
            return proj
            
        elif mode == "phase2":
            class_logits = self.classifier_head(features)
            return class_logits
            
        elif mode == "extract":
            return features
            
        else:
            raise ValueError(f"Mode invalide: {mode}. Choisissez 'phase1' ou 'phase2'.")
        

class PDWSupConOnlyTransformer(nn.Module):
    """
    Transformer optimisé uniquement pour l'apprentissage Supervised Contrastive (SupCon).
    L'évaluation se fait sur les features extraites avant la projection via distance de mahalanobis au plus proche prototype de classe.
    """
    def __init__(self, inputs=3, d_model=64, nhead=4, num_layers=3, proj_dim=64, dim_feedforward=256, dropout=0.1):
        super(PDWSupConOnlyTransformer, self).__init__()
        
        self.embedding = nn.Linear(inputs, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection SupCon (utilisée UNIQUEMENT pour la SupConLoss pendant l'entraînement)
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, proj_dim)
        )

    def forward(self, x, mode="extract"):
        """
        mode: 
          - "train" : Retourne la projection normalisée L2 pour la SupConLoss
          - "extract" : (DÉFAUT) Retourne les features brutes du backbone avant projection
        """
        padding_mask = None
        
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x, lengths = pad_packed_sequence(x, batch_first=True)
            batch_size, max_seq_len, _ = x.size()
            padding_mask = torch.arange(max_seq_len)[None, :] >= lengths[:, None]
            padding_mask = padding_mask.to(x.device)

        if x.size(-1) != 3:
            x = x.permute(0, 2, 1)

        batch_size = x.size(0)
        
        # Embedding + CLS Token
        x = self.embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        if padding_mask is not None:
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat((cls_mask, padding_mask), dim=1)
            
        # Positional Encoding + Encodeur
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        # Extraction de la représentation latente (le token CLS) du BACKBONE
        features = x[:, 0, :]
        
        if mode == "train":
            # On passe dans la tête de projection et on normalise (pour la Loss)
            proj = self.projection_head(features)
            proj = F.normalize(proj, p=2, dim=1)
            return proj
            
        elif mode == "extract":
            # On retourne la représentation brute (pour les Prototypes / Inférence)
            return features
            
        else:
            raise ValueError(f"Mode invalide: {mode}. Choisissez 'train' ou 'extract'.")