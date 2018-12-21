# Rapport

**Objectif:**
Effectuer la classification du type de documents, dans les documents discponible pour le procès des groupes américain de tabac. Les texte on été obtenu par OCR, on utilisera directement les contenus de texte obtenu par l'OCR.

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

* Approches réseaux de neurones


## 3-Evaluation des performance

## Pistes d'améliorations





