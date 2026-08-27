"""
Last updated on 26/08/2026

@author : Foucault BERNARD

Ce fichier contient les fonction de projection en 2 dimension des résultats via algorithme t-SNE.
"""

import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from pathlib import Path
from utils import normalize_batch
from dataset_chunk_seq import Dataset_pulse, LibraryBiasAugmenter
from torch.utils.data import DataLoader
import argparse
import torch
from utils import collate_fn
import transformer
import matplotlib.pyplot as plt
import logging


parser = argparse.ArgumentParser(description='projection 2d via t-sne')
parser.add_argument('--bs', default=64, type=int, help='batch_size')
parser.add_argument('--dir', default="", type=str, help='Dossier contenant les données')
parser.add_argument('--weights', default="", type=str, help='poids du NN à évaluer')
args = parser.parse_args()

# Logging setup
logging.basicConfig(level=logging.INFO, filename='SupCon_Mahalanobis_OpenSet/TransformerSupCon_BiasAugm.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# -------------- PREPARATION DES DONNEES --------------
print('loading data...')
logging.info("loading data...")
min_length = 30 # nb minimal de data dans une classe pour être acceptée pour l'entraînement

classif_mode = 'open_set'
signals_train = Dataset_pulse(args.dir, dirname='train', mode='train', min_nb_data=min_length, classif_mode=classif_mode)#, mode_library=loaded_library)
signals_valid = Dataset_pulse(args.dir, dirname='valid', mode='test', min_nb_data=0,
                              label_to_index=signals_train.label_to_index, mini=signals_train.mini, maxi=signals_train.maxi, classif_mode=classif_mode)

dataloader_valid = DataLoader(signals_valid, batch_size=args.bs, shuffle=False, collate_fn=collate_fn)

print("--- Préparation du modèle ---")
d_model = 256
nhead = 8
num_layers = 4
mode = 'train'
model = transformer.PDWSupConOnlyTransformer(inputs=3, d_model=d_model, nhead=nhead, num_layers=num_layers)
model_name = model.__class__.__name__

col_min_tensor = torch.tensor(signals_train.mini, dtype=torch.float32).cuda()
denom_tensor = torch.tensor(signals_train.denom, dtype=torch.float32).cuda()

fichier = args.weigths

model.load_state_dict(torch.load(fichier))
model.cuda()
model.eval()

print("--- Extraction des projections ---")
all_projections = []
all_labels = []

with torch.no_grad():
    for inputs, labels in dataloader_valid:
        inputs = inputs.cuda()
        labels = labels.squeeze().long().cpu().numpy()

        inputs_norm = normalize_batch(inputs, col_min_tensor, denom_tensor, use_log=True)
        projections = model(inputs_norm, mode=mode)

        all_projections.append(projections.cpu().numpy())
        all_labels.extend(labels)

X = np.vstack(all_projections)
y = np.array(all_labels)

print("--- Calcul de la projection t-SNE ---")

# perplexity: L'hyperparamètre clé de t-SNE (typiquement entre 5 et 50)
# init='pca': Permet une convergence plus stable et rapide
tsne = TSNE(
    n_components=2,
    perplexity=30,
    init='pca',
    learning_rate='auto',
    random_state=42
)
X_tsne = tsne.fit_transform(X)

print("--- Création du graphique ---")
plt.figure(figsize=(12, 10))

class_names_mapped = [signals_train.index_to_label[label] if label != -1 else "-1" for label in y]
palette = sns.color_palette("tab20", n_colors=signals_train.num_class)

sns.scatterplot(
    x=X_tsne[:, 0],
    y=X_tsne[:, 1],
    hue=class_names_mapped,
    palette=palette,
    s=50,
    alpha=0.8,
    edgecolor=None
)

plt.title(f'Projection t-SNE de l\'espace SupCon (Validation)\nModèle : {model_name}_{d_model}_{nhead}_{num_layers}', fontsize=14, pad=15)

plt.xlabel('Dimension t-SNE 1', fontsize=12)
plt.ylabel('Dimension t-SNE 2', fontsize=12)

plt.legend(title='Classes', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
plt.tight_layout()

save_path = f'save_path'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Graphique sauvegardé sous : {save_path}")
