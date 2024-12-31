import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

# Charger les jeux de données
df_train = pd.read_csv('../data/train.csv')
df_test = pd.read_csv('../data/test.csv')

# Prétraitement des données
df_train = df_train.drop(columns=['Surname', 'CustomerId'])
df_test = df_test.drop(columns=['Surname', 'CustomerId'])


df_train = pd.get_dummies(df_train, columns=['Geography', 'Gender'], drop_first=True)
df_test = pd.get_dummies(df_test, columns=['Geography', 'Gender'], drop_first=True)

# Nouvelle feature : client inactif avec solde élevé
df_train['Inactive_high_balance'] = ((df_train['Balance'] > 100000) & (df_train['IsActiveMember'] == 0)).astype(int)
df_test['Inactive_high_balance'] = ((df_test['Balance'] > 100000) & (df_test['IsActiveMember'] == 0)).astype(int)

# Création de nouvelles variables pour maximiser l'AUC
df_train['Age * Balance'] = df_train['Age'] * df_train['Balance']
df_test['Age * Balance'] = df_test['Age'] * df_test['Balance']

df_train['Tenure * NumOfProducts'] = df_train['Tenure'] * df_train['NumOfProducts']
df_test['Tenure * NumOfProducts'] = df_test['Tenure'] * df_test['NumOfProducts']

df_train['NumOfProducts * IsActiveMember'] = df_train['NumOfProducts'] * df_train['IsActiveMember']
df_test['NumOfProducts * IsActiveMember'] = df_test['NumOfProducts'] * df_test['IsActiveMember']

df_train['Balance / EstimatedSalary'] = df_train['Balance'] / df_train['EstimatedSalary']
df_test['Balance / EstimatedSalary'] = df_test['Balance'] / df_test['EstimatedSalary']

# Transformation logarithmique
df_train['Log_Balance'] = np.log1p(df_train['Balance'])
df_test['Log_Balance'] = np.log1p(df_test['Balance'])

df_train['Log_EstimatedSalary'] = np.log1p(df_train['EstimatedSalary'])
df_test['Log_EstimatedSalary'] = np.log1p(df_test['EstimatedSalary'])

# Colonnes continues à scaler
continuous_cols = [
    'CreditScore', 'Balance', 'Age', 'EstimatedSalary',
    'Age * Balance', 'Tenure * NumOfProducts',
    'NumOfProducts * IsActiveMember', 'Balance / EstimatedSalary', 'Log_Balance', 'Log_EstimatedSalary'
]

# Initialisation du scaler
scaler = StandardScaler()
df_train[continuous_cols] = scaler.fit_transform(df_train[continuous_cols])
df_test[continuous_cols] = scaler.transform(df_test[continuous_cols])

# Features et cible
X = df_train.drop(columns=['Exited', 'id'])
y = df_train['Exited']
X_test = df_test.drop(columns=['id'])

# Définir les modèles de base avec des hyperparamètres ajustés
base_learners = [
    ('catboost', CatBoostClassifier(depth=7, iterations=1000, learning_rate=0.03, random_state=42, verbose=0)),
    ('xgb', XGBClassifier(learning_rate=0.03, max_depth=7, n_estimators=1200, colsample_bytree=0.8, subsample=0.8, random_state=42)),
    ('lgbm', LGBMClassifier(num_leaves=10, learning_rate=0.03, n_estimators=100, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=100, random_state=42))
]

# Modèle final (métamodèle)
meta_learner = LGBMClassifier(num_leaves=10, learning_rate=0.03, n_estimators=100, random_state=42)

# Validation croisée pour le stacking
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
roc_auc_scores = []
stacking_preds_test = np.zeros(X_test.shape[0])

for train_idx, val_idx in cv.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner)
    stacking_model.fit(X_train, y_train)

    y_pred_prob = stacking_model.predict_proba(X_val)[:, 1]
    stacking_preds_test += stacking_model.predict_proba(X_test)[:, 1] / cv.n_splits

    auc_score = roc_auc_score(y_val, y_pred_prob)
    roc_auc_scores.append(auc_score)

print(f"Mean AUC score from cross-validation: {np.mean(roc_auc_scores):.4f}")

# Créer le DataFrame avec les prédictions moyennes
output_df = pd.DataFrame({
    'id': df_test['id'],
    'Exited': stacking_preds_test
})

# Sauvegarder les résultats
output_df.to_csv('predictions_stacking_rf_et.csv', index=False)
print("Fichier 'predictions_stacking_rf_et.csv' généré avec succès.")
