"""
Last updated on 26/08/2026

@author : Foucault BERNARD

Ce fichier contient la fonction de train de l'architecture.
"""

from dataset_chunk_seq import Dataset_pulse, PDWAugmenter, LibraryBiasAugmenter
from torch.utils.data import DataLoader
import argparse
import torch
import torch.optim as optim
from utils import collate_fn, BalancedBatchSamplerNoReuse, evaluate_prototypes, \
    generate_final_confusion_matrix_mahalanobis
import transformer, losses
import matplotlib.pyplot as plt
import logging
import json

parser = argparse.ArgumentParser(description='train_supcon_only')
parser.add_argument('--lr', default=2e-4, type=float, help='learning rate')
parser.add_argument('--epoch', default=400, type=int, help='max epoch pour SupCon')
parser.add_argument('--dir', default="", type=str, help='Dossier contenant les données')
args = parser.parse_args()

# Logging setup
logging.basicConfig(level=logging.INFO, filename='SupCon_Mahalanobis_OpenSet/TransformerSupCon_BiasAugm.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# -------------- PREPARATION DES DONNEES --------------
print('loading data...')
logging.info("loading data...")
min_length = 20 # nb minimal de data dans une classe pour être acceptée pour l'entraînement
n_views = 2 # nombre de vues par biais par data
rejection_rate = 0.01 # taux de rejet pour la classification open set

# fichier Json contenant la biblihotèque des sous-modes par mode pour la data augmentation par biais
with open(f"{args.dir}/JsonBIB_scenario_SR.json", "r") as f:
    loaded_library = json.load(f)
lib_augmenter = LibraryBiasAugmenter(mode_library=loaded_library, n_views=n_views)

classif_mode = 'close_set'
signals_train = Dataset_pulse(args.dir, dirname='train', mode='train', min_nb_data=min_length, classif_mode=classif_mode)#, mode_library=loaded_library)
signals_valid = Dataset_pulse(args.dir, dirname='valid', mode='test', min_nb_data=0,
                              label_to_index=signals_train.label_to_index, mini=signals_train.mini, maxi=signals_train.maxi, classif_mode=classif_mode)

# DATALOADER ENTRAINEMENT (SupCon - Balanced)
n_classes = 22
n_samples = 6
balanced_sampler = BalancedBatchSamplerNoReuse(dataset=signals_train, n_classes_per_batch=n_classes,
                                               n_samples_per_class=n_samples)
dataloader_train_supcon = DataLoader(signals_train, batch_sampler=balanced_sampler, collate_fn=collate_fn)

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
model_tot_name = f"{model_name}_{d_model}_{nhead}_{num_layers}_CloseSet_MineLength={min_length}"

logging.info(f"Model : {model_name} | D_model: {d_model} | Nhead: {nhead} | Layers: {num_layers} | Rejection Rate: {rejection_rate}")


# -------------- ENTRAINEMENT : SUPERVISED CONTRASTIVE LEARNING --------------
print('--- Beginning of training : SupCon ---')
logging.info('--- Beginning of training : SupCon ---')

optimizer = optim.AdamW(model.parameters(), lr=args.lr)
criterion_supcon = losses.DecoupledSupConLoss(temperature=0.07).cuda()

total_steps = args.epoch * len(dataloader_train_supcon)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1, div_factor=10.0, final_div_factor=1000.0
)

col_min_tensor = torch.tensor(signals_train.mini, dtype=torch.float32).cuda()
denom_tensor = torch.tensor(signals_train.denom, dtype=torch.float32).cuda()
augmenter = PDWAugmenter(col_min_tensor=col_min_tensor, denom_tensor=denom_tensor, use_log=True, noise_std=0.3,
                         drop_prob=0.1)

# Listes pour les graphiques
loss_train_list, loss_valid_list = [], []
acc_train_proto_list, acc_valid_proto_list = [], []
best_valid_loss = float('inf')

for epoch in range(args.epoch):
    model.train()
    training_loss = 0.0

    for i, (inputs, labels) in enumerate(dataloader_train_supcon, 1):
        inputs, labels = inputs.cuda(), labels.squeeze().long().cuda()
        optimizer.zero_grad()

        string_labels = [signals_train.index_to_label[lbl.item()] for lbl in labels]
        biased_inputs = lib_augmenter.generate_views(inputs, string_labels)

        all_final_projections = []

        for biased_input in biased_inputs:
            v1_norm, v2_norm = augmenter.generate_views(biased_input)
            proj_1 = model(v1_norm, mode="train")
            proj_2 = model(v2_norm, mode="train")

            all_final_projections.extend([proj_1, proj_2])

        projections_cat = torch.cat(all_final_projections, dim=0)
        nb_tot_views = len(all_final_projections)
        labels_cat = torch.cat([labels] * nb_tot_views, dim=0)

        loss = criterion_supcon(projections_cat, labels_cat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        training_loss += loss.item()

    avg_train_loss = training_loss / len(dataloader_train_supcon)
    loss_train_list.append(avg_train_loss)

    # --- VALIDATION (Loss SupCon) ---
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader_valid:
            inputs, labels = inputs.cuda(), labels.squeeze().long().cuda()
            projections = model(inputs, mode="train")
            loss = criterion_supcon(projections, labels)
            valid_loss += loss.item()

    avg_valid_loss = valid_loss / len(dataloader_valid)
    loss_valid_list.append(avg_valid_loss)

    # --- EVALUATION DES PROTOTYPES SUR LE BACKBONE ---
    num_classes = signals_train.num_class
    class_names = [signals_train.index_to_label[i] for i in range(num_classes)]
    if (epoch % 10 == 0 or epoch == args.epoch - 1):
        train_p_acc, valid_p_acc = evaluate_prototypes(
            model,
            dataloader_train_eval,
            dataloader_valid,
            signals_train.num_class,
            col_min_tensor,
            denom_tensor,
            mode='train'
        )
        acc_train_proto_list.append((epoch, train_p_acc))
        acc_valid_proto_list.append((epoch, valid_p_acc))
        proto_msg = f"Proto Train Acc: {train_p_acc:.4f}, Proto Valid Acc: {valid_p_acc:.4f}, \n"
        logging.info(proto_msg)
        print(f"[{epoch}/{args.epoch}] {proto_msg}")

    # SAUVEGARDE
    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        torch.save(model.state_dict(),
                   f'SupCon_Mahalanobis_OpenSet/results/best_backbone_{model_tot_name}.pth')

    log_msg = f"Epoch: {epoch}/{args.epoch} | Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_valid_loss:.4f}"
    print(log_msg)
    logging.info(log_msg)

torch.save(model.state_dict(),
           f'SupCon_Mahalanobis_OpenSet/results/best_backbone_{model_tot_name}_epoch{args.epoch}.pth')
print("End of training.")

# -------------- PLOTS --------------
# Plot SupCon Loss
plt.figure()
plt.plot(loss_train_list, label='Train SupCon Loss')
plt.plot(loss_valid_list, label='Valid SupCon Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.yscale('log')
plt.title('Evolution de la perte SupCon')
plt.savefig(f'SupCon_Mahalanobis_OpenSet/results/loss_{model_tot_name}.png', bbox_inches='tight')
plt.close()

# Plot Prototype Accuracy
if acc_train_proto_list:
    epochs_proto, train_accs = zip(*acc_train_proto_list)
    _, valid_accs = zip(*acc_valid_proto_list)

    plt.figure()
    plt.plot(epochs_proto, train_accs, marker='o', label='Train Proto Acc')
    plt.plot(epochs_proto, valid_accs, marker='o', label='Valid Proto Acc')
    plt.legend()
    plt.ylim((0, 1))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy sur Prototypes (Espace latent brut)')
    plt.savefig(f'SupCon_Mahalanobis_OpenSet/results/acc_proto_{model_tot_name}.png',
                bbox_inches='tight')
    plt.close()

# MATRICE DE CONFUSION (MAHALANOBIS)
best_model_path = f'SupCon_Mahalanobis_OpenSet/results/best_backbone_{model_tot_name}_epoch{args.epoch}.pth'
model.load_state_dict(torch.load(best_model_path))
num_classes = signals_train.num_class
class_names = [signals_train.index_to_label[i] for i in range(num_classes)]

save_path = f'save_path'

generate_final_confusion_matrix_mahalanobis(
    model=model,
    dataloader_train_eval=dataloader_train_eval,
    dataloader_valid=dataloader_valid,
    num_classes=num_classes,
    col_min_tensor=col_min_tensor,
    denom_tensor=denom_tensor,
    class_names=class_names,
    save_path=save_path,
)