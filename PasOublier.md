# MAE — Mean Absolute Error

Si MAE = 4
> Ton modèle se trompe en moyenne de 4 minutes.




# RMSE — Root Mean Squared Error

RMSE = 6
> L’erreur typique est d’environ 6 minutes.

**Différence avec MAE**

Le RMSE pénalise beaucoup plus les grosses erreurs.

Exemple :

Se tromper de 2 minutes → petit impact

Se tromper de 40 minutes → énorme impact

Donc :

RMSE > MAE quand il y a des outliers.




# R² — Coefficient de détermination

Interprétation

R² = 1 → modèle parfait

R² = 0 → aussi mauvais que prédire la moyenne

R² < 0 → pire que la moyenne

Exemple :

R² = 0.75

> Ton modèle explique 75% de la variation des retards.

