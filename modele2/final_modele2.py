# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# Chargement des jeux de données
train_df = pd.read_csv('../data/train.csv')  # Données d'entraînement
test_df = pd.read_csv('../data/test.csv')  # Données de test

# Suppression des colonnes inutiles
train_df = train_df.drop(columns=['Surname', 'CustomerId'])
test_df = test_df.drop(columns=['Surname', 'CustomerId'])

# Ajout d'une variable catégorielle pour regrouper les âges
train_df['AgeGroup'] = pd.cut(train_df['Age'], bins=[-np.inf, 30, 50, np.inf], labels=['Young', 'Adult', 'Senior'])
test_df['AgeGroup'] = pd.cut(test_df['Age'], bins=[-np.inf, 30, 50, np.inf], labels=['Young', 'Adult', 'Senior'])

# Encodage des variables catégoriques en one-hot encoding
train_df = pd.get_dummies(train_df, columns=['Geography', 'Gender', 'AgeGroup'])
test_df = pd.get_dummies(test_df, columns=['Geography', 'Gender', 'AgeGroup'])

# Suppression des valeurs aberrantes sur 'CreditScore'
train_df = train_df[train_df['CreditScore'] <= 900]

# Ajout d'une feature : client inactif avec un solde élevé
train_df['Inactive_high_balance'] = ((train_df['Balance'] > 100000) & (train_df['IsActiveMember'] == 0)).astype(int)
test_df['Inactive_high_balance'] = ((test_df['Balance'] > 100000) & (test_df['IsActiveMember'] == 0)).astype(int)

# Ajout d'un score d'engagement basé sur plusieurs variables
train_df['EngagementScore'] = train_df['Tenure'] * train_df['IsActiveMember'] * train_df['NumOfProducts']
test_df['EngagementScore'] = test_df['Tenure'] * test_df['IsActiveMember'] * test_df['NumOfProducts']

# Création de nouvelles interactions entre variables
train_df['Age * Balance'] = train_df['Age'] * train_df['Balance']
test_df['Age * Balance'] = test_df['Age'] * test_df['Balance']
train_df['Balance / EstimatedSalary'] = train_df['Balance'] / train_df['EstimatedSalary']
test_df['Balance / EstimatedSalary'] = test_df['Balance'] / test_df['EstimatedSalary']

# Transformation logarithmique pour réduire les effets des valeurs extrêmes
train_df['Balance'] = np.log1p(train_df['Balance'])
test_df['Balance'] = np.log1p(test_df['Balance'])

# Renommer la colonne Balance après transformation
train_df.rename(columns={'Balance': 'Log_Balance'}, inplace=True)
test_df.rename(columns={'Balance': 'Log_Balance'}, inplace=True)

# Mise à l'échelle des colonnes continues
continuous_cols = ['CreditScore', 'Age * Balance', 'EngagementScore', 'Balance / EstimatedSalary']
scaler = StandardScaler()
train_df[continuous_cols] = scaler.fit_transform(train_df[continuous_cols])
test_df[continuous_cols] = scaler.transform(test_df[continuous_cols])

# Définition des features et de la cible
X = train_df.drop(columns=["Exited", "id"])  # Features
y = train_df["Exited"]  # Cible
X_test = test_df.drop(columns=["id"])  # Features du test

# Définir les modèles de base pour le stacking
base_learners = [
    ('catboost', CatBoostClassifier(depth=7, iterations=1000, learning_rate=0.03, random_state=42, verbose=0)),
    ('xgb', XGBClassifier(learning_rate=0.03, max_depth=7, n_estimators=1200, colsample_bytree=0.8, subsample=0.8, random_state=42))
]

# Définir le méta-modèle
meta_learner = LGBMClassifier(num_leaves=10, learning_rate=0.03, n_estimators=100, random_state=42)

# Configuration de la validation croisée
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
roc_auc_scores = []
stacking_preds_test = np.zeros(X_test.shape[0])

# Entraînement du modèle avec validation croisée
for train_idx, val_idx in cv.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Modèle de stacking
    stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner)
    stacking_model.fit(X_train, y_train)

    # Calibration du modèle
    calibrated_model = CalibratedClassifierCV(stacking_model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_val, y_val)

    # Prédictions sur validation et test
    y_pred_prob = calibrated_model.predict_proba(X_val)[:, 1]
    stacking_preds_test += calibrated_model.predict_proba(X_test)[:, 1] / cv.n_splits

    # Calcul du score AUC
    auc_score = roc_auc_score(y_val, y_pred_prob)
    roc_auc_scores.append(auc_score)

# Affichage des performances moyennes
print(f"Mean AUC score from cross-validation: {np.mean(roc_auc_scores):.4f}")

# Création du DataFrame pour les prédictions finales
output_df = pd.DataFrame({
    "id": test_df["id"],
    "Exited": stacking_preds_test
})

# Sauvegarde des prédictions
output_df.to_csv('predictions_stacking_lgbm.csv', index=False)
print("Fichier 'predictions_stacking_lgbm.csv' généré avec succès.")

# Chargement des jeux de données
df_train = pd.read_csv('../data/train.csv')  # Données d'entraînement
df_test = pd.read_csv('../data/test.csv')  # Données de test

# Suppression des colonnes inutiles
df_train = df_train.drop(columns=['Surname', 'CustomerId'])
df_test = df_test.drop(columns=['Surname', 'CustomerId'])

# Ajout d'une variable catégorielle pour regrouper les âges
df_train['AgeGroup'] = pd.cut(df_train['Age'], bins=[-np.inf, 30, 50, np.inf], labels=['Young', 'Adult', 'Senior'])
df_test['AgeGroup'] = pd.cut(df_test['Age'], bins=[-np.inf, 30, 50, np.inf], labels=['Young', 'Adult', 'Senior'])

# Encodage des variables catégoriques
df_train = pd.get_dummies(df_train, columns=['Geography', 'Gender', 'AgeGroup'])
df_test = pd.get_dummies(df_test, columns=['Geography', 'Gender', 'AgeGroup'])

# Suppression des valeurs aberrantes (exemple : CreditScore > 900)
df_train = df_train[df_train['CreditScore'] <= 900]

# Ajout d'une feature : client inactif avec solde élevé
df_train['Inactive_high_balance'] = ((df_train['Balance'] > 100000) & (df_train['IsActiveMember'] == 0)).astype(int)
df_test['Inactive_high_balance'] = ((df_test['Balance'] > 100000) & (df_test['IsActiveMember'] == 0)).astype(int)

# Ajout d'un score d'engagement basé sur plusieurs variables
df_train['EngagementScore'] = df_train['Tenure'] * df_train['IsActiveMember'] * df_train['NumOfProducts']
df_test['EngagementScore'] = df_test['Tenure'] * df_test['IsActiveMember'] * df_test['NumOfProducts']

# Création d'interactions entre les variables pour maximiser l'AUC
df_train['Age * Balance'] = df_train['Age'] * df_train['Balance']
df_test['Age * Balance'] = df_test['Age'] * df_test['Balance']
df_train['Balance / EstimatedSalary'] = df_train['Balance'] / df_train['EstimatedSalary']
df_test['Balance / EstimatedSalary'] = df_test['Balance'] / df_test['EstimatedSalary']

# Transformation logarithmique pour réduire les effets des valeurs extrêmes
df_train['Balance'] = np.log1p(df_train['Balance'])
df_test['Balance'] = np.log1p(df_test['Balance'])

# Renommer la colonne Balance après transformation
df_train.rename(columns={'Balance': 'Log_Balance'}, inplace=True)
df_test.rename(columns={'Balance': 'Log_Balance'}, inplace=True)

# Mise à l'échelle des colonnes continues
continuous_cols = ['CreditScore', 'Age * Balance', 'EngagementScore', 'Balance / EstimatedSalary']
scaler = StandardScaler()
df_train[continuous_cols] = scaler.fit_transform(df_train[continuous_cols])
df_test[continuous_cols] = scaler.transform(df_test[continuous_cols])

# Définition des features et de la cible
X = df_train.drop(columns=["Exited", "id"])  # Features
y = df_train["Exited"]  # Cible
X_test = df_test.drop(columns=["id"])  # Features du test

# Définir les modèles de base pour le stacking
base_learners = [
    ('lgbm', LGBMClassifier(num_leaves=10, learning_rate=0.03, n_estimators=100, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=100, random_state=42))
]

# Définir le méta-modèle pour le stacking
meta_learner = LGBMClassifier(num_leaves=10, learning_rate=0.03, n_estimators=100, random_state=42)

# Configuration de la validation croisée
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
roc_auc_scores = []
stacking_preds_test = np.zeros(X_test.shape[0])

# Entraînement et validation avec stacking
for train_idx, val_idx in cv.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Entraînement du modèle de stacking
    stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_learner)
    stacking_model.fit(X_train, y_train)

    # Calibration des probabilités pour améliorer les prédictions
    calibrated_model = CalibratedClassifierCV(stacking_model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_val, y_val)

    # Prédictions sur validation et test
    y_pred_prob = calibrated_model.predict_proba(X_val)[:, 1]
    stacking_preds_test += calibrated_model.predict_proba(X_test)[:, 1] / cv.n_splits

    # Calcul du score AUC pour chaque fold
    auc_score = roc_auc_score(y_val, y_pred_prob)
    roc_auc_scores.append(auc_score)

# Affichage du score AUC moyen
print(f"Mean AUC score from cross-validation: {np.mean(roc_auc_scores):.4f}")

# Génération des prédictions finales
output_df = pd.DataFrame({
    "id": df_test["id"],
    "Exited": stacking_preds_test
})
output_df.to_csv('predictions_stacking_lgbm_rf_et.csv', index=False)
print("Fichier 'predictions_stacking_lgbm_rf_et.csv' généré avec succès.")

# Fusion des prédictions pour créer une moyenne finale
stacking_lgbm_rf_et_df = pd.read_csv('predictions_stacking_lgbm_rf_et.csv')
stacking_lgbm_df = pd.read_csv('predictions_stacking_lgbm.csv')
merged_df = pd.merge(stacking_lgbm_rf_et_df[['id', 'Exited']], stacking_lgbm_df[['id', 'Exited']], on='id', suffixes=('_rf_et', '_lgbm'))
merged_df['Exited'] = (merged_df['Exited_rf_et'] + merged_df['Exited_lgbm']) / 2

# Sauvegarde des résultats finaux
output_file_path = 'predictions_stacking_average.csv'
merged_df[['id', 'Exited']].to_csv(output_file_path, index=False)
print(f"Fichier '{output_file_path}' généré avec succès.")
