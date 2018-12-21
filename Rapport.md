# Rapport

**Objectif:**
Effectuer la classification du type de documents dans les documents disponible pour le procès des groupes américain de tabac. Les texte on été obtenu par OCR, on utilisera directement les contenus de texte obtenu par l'OCR.

## 1-Analyse des données

On dispose d'un dataset de 3482 images, dont on a collecté le texte. Les 3482 textes sont séparé en 10 classes qui sont les différents types de documents disponibles, les classes (type de document) présentes sont:
* Advertissement
* Email
* Form
* Letter
* Memo
* News
* Note
* Report
* Resume
* Scientific


**Proportion de des classe**

![Image proportion de chaque classe](https://github.com/Rouen-NLP/final-lab-EstelleAlemy/blob/master/images/plot.png)

### Opération sur les données 

Avant l'analyse de celle-ci on a réaliser quelques opérations sur celle-ci:

* retraits des indentation dans le texte (\n)
* transformation en sac de mots (bag of word) 
>Pour réaliser cela on utilise la méthode **CountVectorizer** de **sklearn**. Cette méthode effectue la convertion d'une collection de document texte en une matrice qui renvoie le nombre d'occurence de chaque symbole (du dico associé) pour chaque texte. On a fixe le nombre de paramètres des matrices à 3000
* réalisation d'une tf-idf
>Calculer la fréquence de chaque mots (nombre d'occurence du mot/ sur le nombre total de mots) *tf* et on prend l'inverse, la *tf-idf* qui nous permet d'associer des poids qui permettrons de différencier les mot-clés des mots de liason ou pronom...etc (que l'on retouve avec de grande fréquences dans tous les documents car ils correspondent à la syntaxes de la langue mais n'apporte pas d'informations).

## 2-Classification

Pour la classification nous avons décidé de tester 2 approches :
* Approches naives bayes
Pour cette approches on séparer notre ensemble de donner en 80%-20% soit:

Apprentissage | Test
--------------|-------------------
2785 | 697

Ensuite on utilise **MultinomialNB** de **sklearn**, qui est un classifieur de Bayes pour une classification multiclasses.


* Approches réseaux de neurones

Pour l'approche réseaux de neurones (il est très important d'avoir beaucoup de données d'apprentissage) le dataset étant petit on a choisit de prendre un plus petit ensemble de test (90%-10%), de plus durant l'évaluation environs 0.1% des donnés sont utilisées pour de la validation.

Apprentissage | Test
--------------|-------------------
3313 | 349

On a choisit d'entrainer le modèle sur 30 epochs avec  batch_size de 16 (nombre d'exemple d'apprentissage à chaque itérations)
Architecture du réseaux
* 1 Embedding
* 2 Dropout: 0.3 et 0.2 (sert à ne pas prendre en compte certain poid)
* 1 couche de convolution: 64 filtre, fenêtre de taille 5
* 1 Maxpooling : fenêtre de taille 4
* 1 LSTM (recurent neural network, utilisé pour les sequences, ici on a des séquences de texte)
* 1 couche dense (pour le calcule des proba)
 
## 3-Evaluation des performance
### 3-1 Indicateur de performance

On évaluer différentes métriques pour correctement évaluer notre classifieur, celles-ci sont:
* **la précision :** nombre de document pertinent retourné/nombre de document total, permet d'avoir une vision global du systeme
* **le rappel :** nombre de document pertinent retourné/nombre de document réellement pertinent, permet de voir aussi les erreurs 
* **f-mesure :**  f=2x(pxr)/(p+r), elle combine la précision et le rappel et permet d'avoir leur moyenne

On évalue ces métrique sur le systeme complet mais on réalise aussi une mesure de la précision,rappel et f-score sur chaque classe car on a un système multi-classe, ce qui nous permet de voir sur quelle classe le système le plus de mal à effectuer la reconnaissance.

### 3-2 performance du classifieur de Bayes

**Resultats système global**
 Precision | Recall | f-score
-----------|--------|-----------
 0.73 | 0.73 | 0.73



Le systeme donne les même valeur pour les 3 mesures, donc a un système qui se comporte de manière plutot homogène.

**Résulats par classe**

   Classe    | precision  |  recall | f1-score  | support
-------------|------------|---------|-----------|--------------
Advertisement |0.73 | 0.67 |0.70 | 57
Email | 0.93 | 0.93 | 0.93 | 135
Form | 0.81 | 0.82 | 0.81  | 88
Letter | 0.75 | 0.72 |0.74 | 122
Memo | 0.60 | 0.73 | 0.66 |109
News | 0.69 | 0.74 |0.71  | 34
Note | 0.33 | 0.33 | 0.33 | 36
Report | 0.59 | 0.56 | 0.57 | 48
Resume | 1.00 | 1.00 | 1.00 | 15
Scientific | 0.68 | 0.49 | 0.57 | 53

### 3-3 performance du réseaux de neurones

**Resultats système global**

Precision | Recall | f-score
-----------|--------|-----------
 0.75 | 0.75 | 0.75

**Résulats par classe**

Classe  | precision | recall | f1-score | support
--------|-----------|--------|----------|------------
0 | 0.78  | 0.55  |   0.64   |   33
1 | 0.96 | 0.97  |   0.96   |   66
2 | 0.81 | 0.73  |   0.77   |   48
3 | 0.85 | 0.78  |   0.82   |   60
4 | 0.72 | 0.82  |   0.77   |   50
5 | 0.70 | 0.47  |   0.56   |   15
6 | 0.46 | 0.55  |   0.50   |   20
7 | 0.36 | 0.50  |   0.42   |   18
8 | 1.00 | 1.00  |   1.00   |   9
9 | 0.58 | 0.70  |   0.64   |   30



## Pistes d'améliorations

Pour le classifieur de bayes, on pourrait améliorer la classification en jouant un peu plus sur les hyperparamètres du classifieur de bayes.
Pour l'approche réseaux de neurones augmenter le nombre d'exemple d'apprentissage(soit en collectant plus de données ou en génrant de manière artificielle des données -> data augmentation) serait un bon moyen de pouvoir augmenter les performance et aussi tester plusieur algorithme d'optimisation (ici on a utilisé que Adam) comme par exmple CTC et rmsprop (utilisé pour les séquences de texte) .
Pour les 2 méthode de manière générale on a seulement retirer les espaces pour nettoyer le texte, on pourrait voir si le retrait de toute les ponctuation influent sur les performance.








