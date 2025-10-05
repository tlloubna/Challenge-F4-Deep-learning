#  Mushroom Classification Challenge  

###  Objectif du projet  
Le projet consiste à développer un modèle de **classification d’images de champignons** appartenant à trois classes :  
`amanita`, `oyster` et `crimini`.  
L’objectif est d’identifier automatiquement l’espèce d’un champignon à partir d’une image, à l’aide de **réseaux de neurones convolutionnels (CNN)** préentraînés.

---

##  Contexte et approche  
Nous avons débuté par une **analyse visuelle du dataset** afin de mieux comprendre la variabilité des formes, couleurs et textures.  
Après une phase de **recherche bibliographique**, nous avons sélectionné deux architectures CNN de référence :  
- **Inception-V3**  
- **VGG-16**

Ces modèles ont été adaptés à notre dataset en **fine-tuning**, après un préentraînement sur **ImageNet**.  
Des expérimentations multiples ont été menées sur le **redimensionnement des images**, la **préparation des données**, et les **hyperparamètres** d'entraînement.

---

##  Architecture et pipeline

1. **Préparation du dataset**
   - Séparation en trois ensembles :  
     - 70 % entraînement  
     - 15 % validation  
     - 15 % test  
   - Chargement via `flow_from_directory()`  
   - Environ **1643 images** au total.

2. **Augmentation des données**
   - Utilisation de `ImageDataGenerator` pour augmenter artificiellement la taille du dataset :
     - Rotation aléatoire (jusqu’à 20 %)  
     - Zoom (10 %)  
     - Décalage horizontal/vertical (10 %)  
     - Symétrie horizontale  
   - Normalisation des pixels entre 0 et 1.  

3. **Modèles testés**
   - **Inception-V3** (préféré)
     - Entrées : (299×299×3)
     - Fine-tuning avec différents niveaux de gel (100, 150, 200 couches)
     - Couches ajoutées :  
       `GlobalAveragePooling2D` → Dense (L2 + Dropout) → Softmax (3 classes)
     - Optimiseur : Adam (lr = 1e-4)
     - Perte : `categorical_crossentropy`

   - **VGG-16**
     - Entrées : (224×224×3)
     - Architecture testée mais écartée suite à des performances moindres.  

---

##  Résultats expérimentaux  

| Configuration | Split (train/val/test) | Batch | Fine-tune | Train Acc. | Val Acc. | Test Acc. |
|----------------|-----------------------|--------|------------|-------------|-----------|------------|
| InceptionV3 #1 | 70 / 10 / 20 | 32 | 150 | 99 % | 89 % | 91 % |
| InceptionV3 #2 | 70 / 15 / 15 | 16 | 150 | 99 % | 94 % | **95 %** |
| InceptionV3 #3 | 70 / 15 / 15 | 32 | 200 | 98 % | 90 % | 91 % |
| VGG-16 | 70 / 15 / 15 | 32 | - | 85 % | 80 % | 79 % |

 **Meilleure configuration** : InceptionV3, batch size = 16, fine-tune = 150  
→ **Précision test = 95 %**, **validation = 94 %**, **entraînement = 99 %**

---

## Contraintes rencontrées  
- Limitation du **temps GPU sur Google Colab** (2h max).  
  ➜ Contournement via la création de plusieurs environnements Colab.  
- Entraînement CPU très lent (~10 minutes/epoch).  
- Gestion des tailles d’images :  
  ➜ Le redimensionnement global à la plus petite taille entraînait une **perte d’information visuelle importante**.  
  ➜ Solution : conserver un format uniforme mais adapté à Inception (299×299).  

---

## Références scientifiques  
1. **Going Deeper with Convolutions (Inception)**  
   C. Szegedy, W. Liu, Y. Jia, et al. (2015)  
   [arXiv:1409.4842](https://arxiv.org/abs/1409.4842)  
   → Top-1 accuracy : 78.8 %, Top-5 accuracy : 93.9 %

2. **Fully Convolutional Networks for Semantic Segmentation**  
   J. Long, E. Shelhamer, T. Darrell (CVPR 2015)  
   [arXiv:1411.4038](https://arxiv.org/abs/1411.4038)  
   → Top-5 accuracy : 92.7 % sur ImageNet

---

##  Conclusion  
Le modèle **InceptionV3** s’est révélé le plus performant et robuste pour ce jeu de données.  
Malgré les contraintes d’entraînement liées à Colab, nous avons atteint une **précision de 95 % sur le test**, confirmant l’efficacité de cette architecture pour la classification multi-classes de champignons.  

---

##  Technologies utilisées  
- **Langage :** Python  
- **Bibliothèques principales :** TensorFlow / Keras, NumPy, Pandas, Matplotlib  
- **Environnement :** Google Colab  
- **Architecture CNN :** InceptionV3, VGG-16  

