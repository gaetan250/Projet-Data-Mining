import pandas as pd
# Recharger les fichiers CSV récemment téléchargés pour créer la moyenne des prédictions
stacking_lgbm_m1 = pd.read_csv('modele1/predictions_stacking_average.csv')
stacking_et_m1 = pd.read_csv('modele1/predictions_stacking_average_all.csv')

merged_df = pd.merge(stacking_et_m1[['id', 'Exited']],
                     stacking_lgbm_m1[['id', 'Exited']], on='id', suffixes=('_rf_et', '_lgbm'))

merged_df['Exited'] = (merged_df['Exited_rf_et'] + merged_df['Exited_lgbm']) / 2

# Sauvegarder le résultat dans un nouveau fichier CSV
output_file_path = 'modele1/predictions_stacking_averagefbzc.csv'
merged_df[['id', 'Exited']].to_csv(output_file_path, index=False)

print(f"Fichier '{output_file_path}' généré avec succès.")