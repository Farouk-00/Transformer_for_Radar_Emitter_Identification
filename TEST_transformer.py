"""
Last updated on 26/08/2026

@author : Foucault BERNARD

Ce fichier contient la fonction de test de l'architecture préalablement entraînée.
"""

from dataset_chunk_seq import Dataset_pulse, LibraryBiasAugmenter
from torch.utils.data import DataLoader
import argparse
import torch
from utils import collate_fn, generate_closeset_final_TEST_confusion_matrix_mahalanobis
import transformer
import logging
import os
import json

def main():
    parser = argparse.ArgumentParser(description='test du modèle')
    parser.add_argument('--dir', default="", type=str, help='Dossier contenant les données')
    parser.add_argument('--weights', default="", type=str, help='poids du NN à évaluer')
    args = parser.parse_args()

    # Logging setup
    os.makedirs('SupCon_Mahalanobis_OpenSet/results_test', exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename='SupCon_Mahalanobis_OpenSet/TransformerSupCon_Test_closeset.log', filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # -------------- PREPARATION DES DONNEES --------------
    print('loading data...')
    logging.info("loading data...")
    min_length = 4  # nb minimal de data dans une classe pour être acceptée pour l'entraînement
    n_views = 10 # nombre de vues par biais par data

    # fichier Json contenant la biblihotèque des sous-modes par mode pour la data augmentation par biais
    with open(f"{args.dir}/JsonBIB_SCE_PRI_2Good.json", "r") as f:
        loaded_library = json.load(f)
    lib_augmenter = LibraryBiasAugmenter(mode_library=loaded_library, n_views=n_views)

    # Charger train uniquement pour récupérer les normalisations et index exacts
    signals_train_ref = Dataset_pulse(args.dir, dirname='train', mode='train', min_nb_data=min_length)

    # -------------- PREPARATION DU MODELE --------------
    d_model = 256
    nhead = 8
    num_layers = 4
    model = transformer.PDWSupConOnlyTransformer(inputs=3, d_model=d_model, nhead=nhead, num_layers=num_layers)

    # Chargement des poids
    print(f"Chargement des poids depuis : {args.weights}")
    model.load_state_dict(torch.load(args.weights))
    model.cuda()

    # Le dataset de test pour l'évaluation finale
    dirtests = ['dirtest1', 'dirtest2', 'dirtest3']
    for dirtest in dirtests:

        model_name = f'{model.__class__.__name__}_MinLength={min_length}_{dirtest}'

        signals_test = Dataset_pulse(args.dir, dirname=f'test/{dirtest}', mode='test', min_nb_data=0,
                                     mini=signals_train_ref.mini, maxi=signals_train_ref.maxi)

        # DATALOADERS EVALUATION
        # On utilise valid_ft pour extraire les prototypes de référence, et test pour l'évaluation
        dataloader_ft_eval = DataLoader(signals_train_ref, batch_size=1, shuffle=True, collate_fn=collate_fn)
        dataloader_test = DataLoader(signals_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

        col_min_tensor = torch.tensor(signals_train_ref.mini, dtype=torch.float32).cuda()
        denom_tensor = torch.tensor(signals_train_ref.denom, dtype=torch.float32).cuda()

        logging.info(f"Evaluating Model : {model_name} | Weights: {args.weights}")

        train_class_names = [signals_train_ref.index_to_label[i] for i in range(signals_train_ref.num_class)]
        test_class_names = [signals_test.index_to_label[i] for i in range(signals_test.num_class)]

        save_path = 'save_path'

        generate_closeset_final_TEST_confusion_matrix_mahalanobis(
            model=model,
            dataloader_train_eval=dataloader_ft_eval, # Base de référence
            dataloader_test=dataloader_test,         # Données à prédire
            num_train_classes=signals_train_ref.num_class,
            num_test_classes=signals_test.num_class,
            col_min_tensor=col_min_tensor,
            denom_tensor=denom_tensor,
            train_class_names=train_class_names,
            test_class_names=test_class_names,
            save_path=save_path,
            bias_augmenter=lib_augmenter
        )

        print(f"Matrice de confusion sauvegardée sous : {save_path}")

if __name__ == '__main__':
    main()