# Credit Risk ML POC

Objectif :
+ prédire le risque de défaut de crédit.
+ Mise en production du moddèle de scoring.

## Création d'un modèle de prédiction

La réalisation des étapes suivantes doit tenir compte de l'objectif 2 qui consiste en la mise en production du modèle.
+ Une gestion des Secrets, credentials, URLs, Paths, env vars runtime via pydantic-setting + .env
+ Hyperparamètres ML, structure d'expérience, via *.yaml
+ Separation des role,
    - un repo research/ avec sous repo notebooks/, experiments/, configs/
    - un repo src/credit_risk avec le code de production partagé (preprocessing partagé R&D + prod)
    - un repo api/, FastAPI, serving
    - un repo ui/, Gradio

N.B la compettion Kaggle étant fini on ne s'interesse pas au application_test.csv.

1. Créer les fonctions d'import des donées:
    + Une classe BaseDataLoader, des classes filles PDDataLoader / PLDataLoader / PLLazyDataLoader, qui implementent
2. Effectuer une EDA, qui servira de base pour justifier les choix concernant aux étapes suivantes (cleaning, imputation, preprocessing, ...)
    + On pourra utiliser le package missingno pour l'études des valeurs manquantes
    + On verifira qu'un valeur manquante n'est pas un alias pour une valeur (=0) par exemple
    + Analyse des correlation, entre continue-continue, cat-continue, cat-cat.
    + Analyse des distributions
    + On peferera seaborn pour l'élaboration des graphiques
3. Implementer le data cleaning et aggregation des tables,
4. Implementer l'imputation si pertinent
5. Implementer le preprocessing et feature engineering
6. Modelisation
7. Evaluation des modèles

L'évalutaiton des modèle se fait sur ROC-AUC, on veillera a rentre les resultats reproductibles, le dataset étant desequilibré on pensera au weight des obseravation pour l'entrainement du model, un faux negatif étant couteux on implementera une fonction de cout
