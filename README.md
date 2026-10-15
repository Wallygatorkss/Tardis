# TARDIS — Analyse intelligente des retards ferroviaires

> Un tableau de bord interactif pour explorer les retards de trains, comparer les services et estimer le retard moyen d’une liaison.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Interface-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Données-Pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-189FDD)](https://xgboost.readthedocs.io/)

## Aperçu du projet

TARDIS est un projet d’analyse de données et de machine learning consacré à la ponctualité ferroviaire. Il transforme un historique de circulations en indicateurs lisibles et en visualisations interactives afin d’identifier les tendances de retard selon la gare, la liaison, le service et la période étudiée.

Le projet propose également un simulateur de prédiction : à partir des caractéristiques historiques d’une liaison, plusieurs modèles pré-entraînés peuvent estimer le retard moyen attendu à l’arrivée.

## Fonctionnalités clés

- **Exploration interactive** des retards avec sélection d’une gare de départ, d’une année et d’une période de l’année.
- **Indicateurs synthétiques** : retard moyen, volume de circulations prévues et taux de ponctualité.
- **Visualisations statistiques** : distribution des retards, comparaison entre services et évolution temporelle.
- **Simulateur de machine learning** capable d’estimer un retard moyen à partir des données historiques d’une liaison.
- **Notebooks reproductibles** dédiés au nettoyage, à l’exploration et à la préparation des modèles.

## Technologies utilisées

| Domaine | Technologies |
| --- | --- |
| Langage | Python |
| Interface | Streamlit |
| Manipulation des données | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine learning | Scikit-learn, XGBoost |
| Sérialisation des modèles | Joblib |
| Analyse expérimentale | Jupyter Notebook |

## Pré-requis

- Python **3.9 ou supérieur**
- `pip`
- Un environnement virtuel Python recommandé
- Le fichier de données `cleaned_dataset.csv`
- Le modèle `model.pkl` pour utiliser le modèle de régression disponible dans le dépôt

Les autres choix proposés dans le simulateur (`modele_random_tree_huge.pkl`, `modele_random_tree_low.pkl` et `modele_XGB_Regressor.pkl`) nécessitent que les fichiers de modèles correspondants soient présents à la racine du projet.

## Arborescence du projet

```text
Tardis/
├── cleaned_dataset.csv    # Jeu de données nettoyé, séparé par des « ; »
├── graph.py               # Fonctions de visualisation utilisées par Streamlit
├── model.pkl              # Modèle pré-entraîné disponible
├── requirements.txt        # Dépendances Python
├── tardis_dashboard.py    # Application Streamlit principale
├── tardis_eda.ipynb       # Exploration et visualisation des données
├── tardis_model.ipynb     # Préparation des données et modélisation
└── PasOublier.md          # Notes sur les métriques et points d’attention
```

## Installation

### 1. Cloner le projet

```bash
git clone https://github.com/Wallygatorkss/Tardis.git
cd Tardis
```

### 2. Créer un environnement virtuel

Sous Linux ou macOS :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Sous Windows PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Utilisation

### Lancer le dashboard

Depuis la racine du projet, exécuter :

```bash
streamlit run tardis_dashboard.py
```

L’application s’ouvre généralement à l’adresse [http://localhost:8501](http://localhost:8501).

Dans l’interface, il est possible de :

1. Sélectionner une gare de départ.
2. Choisir une année et une plage de mois.
3. Consulter les indicateurs de retard et de ponctualité.
4. Analyser les graphiques par date et par service.
5. Sélectionner une destination et lancer une prédiction.

Si un graphique indique qu’aucune donnée n’est disponible, élargir la période ou modifier la gare sélectionnée.

### Utiliser les notebooks

Pour travailler sur l’exploration ou la modélisation :

```bash
jupyter notebook
```

Puis ouvrir `tardis_eda.ipynb` ou `tardis_model.ipynb`.

## Données et modèle

Le fichier `cleaned_dataset.csv` contient notamment les informations suivantes :

- la date, l’année et le mois de circulation ;
- le service et les gares de départ et d’arrivée ;
- le nombre de circulations prévues, d’annulations et de trains en retard ;
- les retards moyens au départ et à l’arrivée ;
- les catégories de causes de retard.

Le dashboard lit le CSV avec un séparateur `;`. Toute nouvelle source de données doit donc conserver cette structure ou être adaptée dans `tardis_dashboard.py`.

Les modèles chargés par l’application doivent exposer `feature_names_in_` et accepter les variables préparées par le dashboard. Cette vérification permet d’identifier rapidement une incompatibilité entre le modèle et les colonnes fournies en entrée.

## Ce que j’ai appris / Défis techniques

- **Préparer des données réelles** : nettoyage des doublons, traitement des colonnes inutiles et transformation des variables temporelles pour rendre le dataset exploitable.
- **Construire une analyse lisible** : passage de données tabulaires brutes à des indicateurs et graphiques compréhensibles par un utilisateur non spécialiste.
- **Concevoir une interface interactive** : gestion des filtres Streamlit, des sélections dépendantes entre gare de départ et destination, ainsi que des cas où le filtre ne renvoie aucune donnée.
- **Comparer des modèles de régression** : utilisation de métriques comme la MAE, la RMSE et le coefficient de détermination R² pour interpréter les performances, plutôt que de se limiter à une seule valeur de précision.
- **Garantir la compatibilité entre données et modèle** : construction des variables d’entrée dans le même ordre que celui attendu par le modèle et affichage des colonnes attendues en cas d’erreur.
- **Prendre en compte les valeurs atypiques** : les retards extrêmes peuvent fortement influencer les moyennes et la RMSE ; les distributions et les quantiles permettent de mieux les identifier.

## Limites et pistes d’amélioration

- Ajouter les modèles `.pkl` manquants ou rendre leur disponibilité configurable.
- Déporter les chemins de données et de modèles dans un fichier de configuration ou des variables d’environnement.
- Ajouter des tests automatisés pour le calcul de la ponctualité, la préparation des entrées et les fonctions de visualisation.
- Améliorer la validation des données à l’ouverture du dashboard afin de fournir un message explicite lorsqu’une colonne obligatoire manque.
- Enrichir l’évaluation des modèles avec une séparation stricte entre données d’entraînement et données de test, ainsi qu’une validation croisée.