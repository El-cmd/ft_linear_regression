# ft_linear_regression

Ce projet introduit les bases du machine learning en implémentant une
régression linéaire simple sans utiliser de bibliothèque qui entraîne le
modèle à notre place.

L'objectif est de prédire le prix d'une voiture à partir de son kilométrage.
Les paramètres du modèle sont appris avec une descente de gradient.

## Étapes du projet

```text
1. Lire le CSV
   ↓
2. Créer estimate_price()
   ↓
3. Initialiser theta0 = 0 et theta1 = 0
   ↓
4. Normaliser les kilométrages
   ↓
5. Calculer les prédictions et les erreurs
   ↓
6. Calculer les corrections de theta0 et theta1
   ↓
7. Mettre theta0 et theta1 à jour simultanément
   ↓
8. Répéter jusqu'à la convergence : c'est l'entraînement
   ↓
9. Sauvegarder les theta et les bornes de normalisation
   ↓
10. Charger ces valeurs dans predict.py pour faire une prédiction
   ↓
Bonus : afficher les graphiques et mesurer la précision
```

### 1. Lire le CSV

Le fichier `data/data.csv` contient deux colonnes :

- `km` : le kilométrage de la voiture ;
- `price` : son prix réel.

`dataset.py` lit ces colonnes et les place dans deux listes : `mileages` et
`prices`.

### 2. Estimer un prix

La fonction `estimate_price()` applique l'hypothèse de la régression
linéaire :

```text
estimated_price(x) = theta0 + theta1 × x
```

Avec :

- `x` : le kilométrage normalisé ;
- `theta0` : l'ordonnée à l'origine ;
- `theta1` : la pente de la droite.

### 3. Initialiser les paramètres

Avant l'entraînement, le modèle ne connaît encore rien :

```text
theta0 = 0
theta1 = 0
```

Toutes les premières prédictions valent donc zéro.

### 4. Normaliser les kilométrages

Les kilométrages sont ramenés entre `0` et `1` avec la normalisation
min-max :

```text
x_normalized = (x - min_mileage) / (max_mileage - min_mileage)
```

La normalisation évite de travailler directement avec de grandes valeurs de
kilométrage et rend la descente de gradient plus stable.

`min_mileage` et `max_mileage` doivent être conservés : `predict.py` doit
utiliser exactement les mêmes bornes pour normaliser la saisie de
l'utilisateur.

### 5. Calculer l'erreur

Pour chaque voiture `i`, le programme calcule d'abord une prédiction, puis la
différence avec le prix réel :

```text
prediction_i = theta0 + theta1 × mileage_i
error_i = prediction_i - real_price_i
```

Une erreur positive signifie que le prix est surestimé. Une erreur négative
signifie qu'il est sous-estimé.

### 6. Calculer les corrections

Soit `m` le nombre de voitures et `learning_rate` le taux d'apprentissage.
Les deux corrections sont calculées avec les formules du sujet :

```text
tmp_theta0 = learning_rate × (1 / m) × somme(error_i)

tmp_theta1 = learning_rate × (1 / m)
             × somme(error_i × mileage_i)
```

Le projet utilise actuellement :

```text
learning_rate = 0.1
```

Un taux trop grand peut empêcher la convergence. Un taux trop petit rend
l'entraînement plus lent.

### 7. Mettre les paramètres à jour

Les deux nouvelles valeurs sont d'abord calculées avec les anciens theta :

```text
new_theta0 = theta0 - tmp_theta0
new_theta1 = theta1 - tmp_theta1
```

Elles sont ensuite appliquées simultanément :

```text
theta0, theta1 = new_theta0, new_theta1
```

Cette mise à jour simultanée est importante : `new_theta1` ne doit pas être
calculé avec un `theta0` déjà modifié.

### 8. Répéter l'entraînement

Une seule correction ne suffit pas. Le programme recommence les prédictions,
les erreurs et les mises à jour jusqu'à ce que les deux corrections soient
presque nulles.

```text
abs(tmp_theta0) < tolerance
et
abs(tmp_theta1) < tolerance
```

La tolérance actuelle vaut :

```text
tolerance = 0.000001
```

La tolérance ne signifie pas que toutes les erreurs de prix sont nulles. Elle
indique que les theta ne changent presque plus et que le modèle a convergé.
Une limite de sécurité de `100 000` itérations empêche une boucle infinie.

### 9. Sauvegarder le modèle

Après l'entraînement, `model_io.py` enregistre dans `theta.json` :

```json
{
    "theta0": 8008.439756329932,
    "theta1": -4656.591243282682,
    "min_mileage": 22899.0,
    "max_mileage": 240000.0
}
```

Les valeurs exactes peuvent changer si le dataset ou les paramètres
d'entraînement sont modifiés.

### 10. Faire une prédiction

`predict.py` suit ces étapes :

1. charger `theta.json` ;
2. demander un kilométrage à l'utilisateur ;
3. normaliser ce kilométrage avec `min_mileage` et `max_mileage` ;
4. appeler `estimate_price()` avec les theta entraînés ;
5. afficher le prix estimé.

Si le kilométrage sort de l'intervalle du dataset, le programme continue avec
une extrapolation et affiche un avertissement. Cette estimation sera moins
fiable qu'une prédiction située dans l'intervalle d'entraînement.

## Bonus : graphiques et précision

### Visualisation

`visualization.py` affiche :

- les voitures du dataset sous forme de points ;
- la droite de régression en rouge ;
- l'évolution de la droite pendant l'entraînement, avec un rafraîchissement
  toutes les 10 itérations.

La figure finale est également enregistrée dans `data_plot.png`.

### Mesure de la précision

Une mesure possible est l'erreur quadratique moyenne, ou MSE :

```text
MSE = (1 / m) × somme((prediction_i - real_price_i)²)
```

Plus la MSE est proche de zéro, plus les prédictions sont proches des prix
réels.

Le programme calcule actuellement l'erreur absolue moyenne, ou MAE :

```text
MAE = (1 / m) × somme(abs(prediction_i - real_price_i))
```

La MAE s'exprime directement dans la même unité que les prix. Par exemple,
une MAE de `500 euros` signifie que les prédictions s'écartent en moyenne de
`500 euros` des prix réels.

## Organisation des fichiers

```text
data/data.csv    Dataset des voitures
dataset.py       Lecture du fichier CSV
model.py         Prédiction et normalisation
model_io.py      Sauvegarde et chargement du modèle
train.py         Descente de gradient
predict.py       Saisie et prédiction utilisateur
visualization.py Animation Matplotlib
metrics.py       Calcul de l'erreur absolue moyenne
theta.json       Paramètres entraînés
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Sous Linux, une interface Tk peut être nécessaire pour ouvrir la fenêtre
Matplotlib :

```bash
sudo apt install python3-tk
```

Sans interface graphique, le programme enregistre tout de même la figure
finale dans `data_plot.png`.

## Utilisation

Entraîner le modèle et générer `theta.json` :

```bash
python3 train.py
```

Faire une prédiction :

```bash
python3 predict.py
```

Exemple :

```text
Enter a mileage: 100000
Estimated price: 6354.70
```

Le projet n'utilise aucune fonction de régression prête à l'emploi comme
`numpy.polyfit`. La descente de gradient est entièrement implémentée dans le
projet.
