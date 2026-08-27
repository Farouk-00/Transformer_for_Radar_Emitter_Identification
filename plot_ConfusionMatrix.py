"""
Last updated on 26/08/2026

@author : Foucault BERNARD

Ce fichier permet de tracer les matrices de confusion finales d'un modèle après entraînement pour différents types de rejet
(pour la classification open set).
"""

from dataset_chunk_seq import Dataset_pulse
from torch.utils.data import DataLoader
import argparse
import torch
from utils import collate_fn, generate_openset_final_confusion_matrix_mahalanobis
import transformer
import logging

parser = argparse.ArgumentParser(description='plot confusion matrix')
parser.add_argument('--dir', default="", type=str, help='dossier contenant les données')
parser.add_argument('--weights', default="", type=str, help='poids du NN à évaluer')
args = parser.parse_args()

# Logging setup
logging.basicConfig(level=logging.INFO, filename='SupCon_Mahalanobis_OpenSet/plot_conf_mat.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# -------------- PREPARATION DES DONNEES --------------
print('loading data...')
logging.info("loading data...")
min_length = 20 # nb minimal de data dans une classe pour être acceptée pour l'entraînement
n_views = 2 # nombre de vues par biais par data
rejection_rate = 0.01 # taux de rejet pour la classification open set

classif_mode = 'open_set' # 'open_set' gives label "-1" to unused classes, 'close_set' throws away unused classes
signals_train = Dataset_pulse(args.dir, dirname='train', mode='train', min_nb_data=min_length, classif_mode=classif_mode)#, mode_library=loaded_library)
signals_valid = Dataset_pulse(args.dir, dirname='valid', mode='test', min_nb_data=0,
                              label_to_index=signals_train.label_to_index, mini=signals_train.mini, maxi=signals_train.maxi, classif_mode=classif_mode)

# DATALOADERS EVALUATION (Prototypes)
dataloader_train_eval = DataLoader(signals_train, batch_size=1, shuffle=True, collate_fn=collate_fn)
dataloader_valid = DataLoader(signals_valid, batch_size=1, shuffle=True, collate_fn=collate_fn)

# -------------- PREPARATION DU MODELE --------------
d_model = 256
nhead = 8
num_layers = 4
model = transformer.PDWSupConOnlyTransformer(inputs=3, d_model=d_model, nhead=nhead, num_layers=num_layers)
model.cuda()
model_name = model.__class__.__name__
model_tot_name = f"{model_name}_{d_model}_{nhead}_{num_layers}_Rej{rejection_rate}_CloseSet_MinLength={min_length}"

logging.info(f"Model : {model_name} | D_model: {d_model} | Nhead: {nhead} | Layers: {num_layers} | Rejection Rate: {rejection_rate}")

col_min_tensor = torch.tensor(signals_train.mini, dtype=torch.float32).cuda()
denom_tensor = torch.tensor(signals_train.denom, dtype=torch.float32).cuda()

# MATRICE DE CONFUSION (MAHALANOBIS)
best_model_path = args.weights
model.load_state_dict(torch.load(best_model_path))
num_classes = signals_train.num_class
class_names = [signals_train.index_to_label[i] for i in range(num_classes)]

save_path = f'SupCon_Mahalanobis_OpenSet/plot_conf_mat/confusion_matrix_proto_{model_tot_name}.png'

generate_openset_final_confusion_matrix_mahalanobis(
    model=model,
    dataloader_train_eval=dataloader_train_eval,
    dataloader_valid=dataloader_valid,
    num_classes=num_classes,
    col_min_tensor=col_min_tensor,
    denom_tensor=denom_tensor,
    class_names=class_names,
    save_path=save_path,
    rejection_rate=rejection_rate,
    unknown_label=-1
)

"""
Si close set :
generate_final_confusion_matrix_mahalanobis(
    model=model,
    dataloader_train_eval=dataloader_train_eval,
    dataloader_valid=dataloader_valid,
    num_classes=num_classes,
    col_min_tensor=col_min_tensor,
    denom_tensor=denom_tensor,
    class_names=class_names,
    save_path=save_path,
)"""