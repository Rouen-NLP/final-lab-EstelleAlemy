# Rapport

**Objectif:**
Effectuer la classification du type de documents, dans les documents disponible pour le procès des groupes américain de tabac. Les texte on été obtenu par OCR, on utilisera directement les contenus de texte obtenu par l'OCR.

## 1-Analyse des données

On dispose d'un dataset de 3482 images, dont on a collecté le texte. Les 3482 textes sont séparé en 10 classes qui sont les différents type de documents disponibles, les classes (type de document) présent sont:
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

![Image proportion de chaque classe](https://github.com/Rouen-NLP/final-lab-EstelleAlemy/blob/master/images/plot.png)

### Opération sur les données 

Avant l'analyse de celle-ci on a réaliser quelques opérations sur celle-ci:

* retraits des indentation dans le texte (\n)
* transformation en sac de mots (bag of word)
>Pour réaliser cela on utilise la méthode **CountVectorizer** de **sklearn**. Cette méthode effectue la convertion d'une collection de document texte en une matrice qui renvoie le nombre d'occurence de chaque symbole (du dico associé) pour chaque texte. On a fixe le nombre de paramètres des matrices à 3000
* réalisation d'une tf-idf
>Calculer la fréquence de chaque mots (nombre d'occurence du mot/ sur le nombre total de mots) *tf* et on prend l'inverse, la *tf-idf* qui nous permet d'associer des poids qui permettrons de différencier les mot-clés des mots de liason ou pronom...etc (que l'on retouve avec de grande fréquences dans tous les documents car ils correspondent à la syntaxes de la langue mais n'apporte pas d'informations).

## 2-Classification

Pour la classifiction nous avons décidé de tester 2 approches :
* Approches naives bayes
Pour cette approches on séparer notre ensemble de donner en 80%-20% soit:

Apprentissage | Test/Validation
2785 | 697

Ensuite on utilise **MultinomialNB** de **sklearn**, qui est un classifieur de Bayes pour une classification multiclasses.


* Approches réseaux de neurones


## 3-Evaluation des performance
### Indicateur de performance

On évaluer différentes métriques pour correctement évaluer notre classifieur, celles-ci sont:
* **la précision :** nombre de document pertinent retourné/nombre de document total, permet d'avoir une vision global du systeme
* **le rappel :** nombre de document pertinent retourné/nombre de document réellement pertinent, permet de voir aussi les erreurs 
* **f-mesure :**  f=2x(pxr)/(p+r), elle combine la précision et le rapelle et permet d'avoir leur moyenne

On évalue ces métrique sur le systeme complet mais on réalise aussi une mesure de la précision,rappel et f-score sur chaque classe car on a un système multi-classe, ce qui nous permet de voir sur quel classe le système le plus de mal à effectuer la reconnaissance.
### 3-2 performance du classifieur de Bayes

précision | recall | f-score
0.73 | 0.73 | 0.73

Le systeme donne les même valeur pour les 3 mesures, donc a un système qui se comporte de manière plutot homogène.
               precision    recall  f1-score   support

Advertisement       0.73      0.67      0.70        57
        Email       0.93      0.93      0.93       135
         Form       0.81      0.82      0.81        88
       Letter       0.75      0.72      0.74       122
         Memo       0.60      0.73      0.66       109
         News       0.69      0.74      0.71        34
         Note       0.33      0.33      0.33        36
       Report       0.59      0.56      0.57        48
       Resume       1.00      1.00      1.00        15
   Scientific       0.68      0.49      0.57        53

### 3-3 performance du réseaux de neurones




## Pistes d'améliorations





