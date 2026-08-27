Projet issu du stage de recherche (02/04/2026 - 28/08/2026) de Foucault BERNARD (foucaultbernard2003@gmail.com) pour le traitement des ambigüités SPECTRA par IA.
Une architecture de Transformer est développée pour identifier des PDW déjà désentrelacées, les données d'entraînement, de validation et de test ainsi que les poids finaux sont sur réseau secret.
L'entraînement est adapté au rejet (classification open set).

dataset_chunk_seq.py : Ce fichier contient les classes de traitement des données (chunking des séquences de PDW à la bonne taille, normalisation...)
et de création des augmentations pour l'apprentissage contrastif (noising, cropping, dropping) et des augmentations par
biais (pour data augmentation).

losses.py : Ce fichier contient les différentes loss implémentées pour l'apprentissage contrastif : loss SupCon, Decoupled loss SupCon et Decoupled loss SupCon Mahalanobis.

metrics.py : Ce fichier contient les fonction de calcul des métriques d'évaluation de classification (recall, precision et F1-score).

plot_ConfusionMatrix.py : Ce fichier permet de tracer les matrices de confusion finales d'un modèle après entraînement pour différents types de rejet
(pour la classification open set).

TEST_transformer.py : Ce fichier contient la fonction de test de l'architecture préalablement entraînée.

TRAIN_transformer.py : Ce fichier contient la fonction de train de l'architecture.

transformer.py : Ce fichier contient les classes de construction des modèles de NN.

tSNE_proj.py : Ce fichier contient les fonction de projection en 2 dimension des résultats via algorithme t-SNE.

utils.py : Ce fichier contient les fonction et classes utiles pour le reste du projet