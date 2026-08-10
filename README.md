# Multimodal Emotion Recognition (FER-2013 & RAVDESS)

## Présentation du Projet
Ce projet implémente un système de **Reconnaissance Multimodale des Émotions faciales et temporelles**, combinant l'analyse d'images statiques et le traitement de séquences vidéo dynamiques. 

L'objectif est d'utiliser un modèle entraîné sur des expressions faciales de base pour extraire des caractéristiques visuelles, puis d'analyser l'évolution de ces émotions au fil du temps dans des vidéos à l'aide d'un réseau récurrent.

## Architecture du Pipeline
Le projet s'articule en deux grandes phases complémentaires :

### Phase 1 : Apprentissage de la vision des émotions (FER-2013)
Cette première partie consiste à enseigner à un réseau de neurones comment identifier une émotion à partir d'une image fixe de visage :
* **Le Dataset :** Utilisation de FER-2013, composé d'images de 48x48 pixels en niveaux de gris classées en 7 émotions fondamentales (Colère, Dégoût, Peur, Joie, Neutre, Tristesse, Surprise).
* **Le Modèle :** Conception et entraînement d'un CNN léger nommé **Mini-Xception**, optimisé grâce à des convolutions séparables en profondeur pour apprendre efficacement les traits faciaux. Le meilleur modèle entraîné est sauvegardé sous forme de poids (`best_fer_model.keras` / `final_fer_mini_xception.keras`).

### Phase 2 : Analyse de l'expression temporelle (RAVDESS + LSTM)
Cette seconde partie exploite le modèle Mini-Xception pré-entraîné pour l'appliquer à des séquences vidéo dynamiques :
* **Le Dataset :** Utilisation de RAVDESS (*Ryerson Audio-Visual Database of Emotional Speech and Song*), qui contient des vidéos d'acteurs simulant différentes émotions. 
* **La Correspondance des Labels :** Un dictionnaire de mapping est implémenté pour relier les 8 classes de RAVDESS aux 7 catégories de FER-2013.
* **Le Traitement Vidéo :** Pour chaque vidéo, un script extrait uniformément 20 frames clés, utilise OpenCV (Haar Cascade) pour détecter et rogner le visage, puis normalise l'image au format 48x48.
* **Le Modèle Séquentiel :** Les caractéristiques extraites par le CNN pour ces 20 frames sont transmises à un réseau de neurones récurrents de type **LSTM** (*Long Short-Term Memory*), qui analyse la dynamique temporelle pour prédire l'émotion globale de la vidéo.

*En somme : FER-2013 entraîne les « yeux » du modèle (le CNN), et RAVDESS entraîne le « cerveau » (le LSTM) à comprendre comment ces émotions se manifestent et évoluent dans le temps.*

## Stack Technique
* **Langage :** Python
* **Deep Learning :** TensorFlow / Keras (CNN, Mini-Xception, LSTM)
* **Traitement d'Images & Vidéos :** OpenCV (`cv2`)
* **Manipulation de Données :** NumPy, Pandas, Scikit-learn

## Structure du Dépôt
```text
emotion-recognition-multimodal/
├── notebooks/
│   ├── Fer2013_Ram.ipynb        # Entraînement du CNN Mini-Xception sur FER-2013 (optimisé RAM)
│   └── RAVDESS.ipynb            # Extraction des frames vidéo et entraînement du modèle LSTM temporel
├── .gitignore                   # Exclusion des datasets lourds et des fichiers de poids
└── README.md