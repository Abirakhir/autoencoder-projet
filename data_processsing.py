import pandas as pd
import os

def process_data(file_path):
    # Charger le dataset
    df = pd.read_csv(file_path)
    
    # Nettoyage des données
    print("Cleaning data...")
    df.isnull().sum()  # Vérifier les valeurs manquantes
    df.duplicated().sum()  # Vérifier les doublons
    df.drop_duplicates(inplace=True)  # Supprimer les doublons
    df.dropna(inplace=True)  # Supprimer les valeurs manquantes

    # Définir le chemin du fichier nettoyé
    cleaned_file_path = file_path.replace('.csv', '_cleaned.csv')

    # Supprimer le fichier existant s'il existe
    if os.path.exists(cleaned_file_path):
        os.remove(cleaned_file_path)
        print(f"Fichier existant '{cleaned_file_path}' supprimé.")

    # Sauvegarder le DataFrame nettoyé
    df.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned data saved to {cleaned_file_path}")

    return cleaned_file_path

# Exemple d'utilisation
# cleaned_path = process_data('chemin/vers/le/fichier.csv')
