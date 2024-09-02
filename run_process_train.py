import os
from data_processsing import process_data
from train import train_autoencoder

def main():
    # Définir le répertoire actuel et le nom du fichier CSV
    current_directory = os.getcwd()  # Obtient le répertoire actuel
    file_name = 'creditcard.csv'  # Nom du fichier de données
    file_path = os.path.join(current_directory, file_name)  # Combine le répertoire actuel avec le nom du fichier
    
    # Étape de traitement des données
    cleaned_file_path = process_data(file_path)
    
    # Étape d'entraînement du modèle
    model_save_path = train_autoencoder(cleaned_file_path)
    
    print(f"Le modèle a été sauvegardé à l'emplacement {model_save_path}.")

if __name__ == '__main__':
    main()
