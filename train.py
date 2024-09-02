import pandas as pd
import os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from joblib import dump
def train_autoencoder(file_path):
    # Charger le dataset nettoyé
    df = pd.read_csv(file_path)

    # Séparer les transactions normales et frauduleuses
    normal_data = df[df['Class'] == 0]
    fraud_data = df[df['Class'] == 1]

    # Diviser les transactions normales en train, validation, et test
    normal_train_data, normal_test_data = train_test_split(normal_data, test_size=0.2, random_state=42)
    normal_train_data, normal_val_data = train_test_split(normal_train_data, test_size=0.2, random_state=42)

    # Séparer les features et labels pour l'entraînement et la validation
    X_train = normal_train_data.drop(columns=['Class'])
    X_val = normal_val_data.drop(columns=['Class'])
    X_test = normal_test_data.drop(columns=['Class'])

    # Normalisation des données
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Définition de l'encodeur
    n_features = X_train.shape[1]
    encoder = models.Sequential(name='encoder')
    encoder.add(layers.Dense(units=100, activation='relu', input_shape=[n_features]))
    encoder.add(layers.Dropout(0.2))  # Dropout pour régularisation
    encoder.add(layers.Dense(units=50, activation='relu'))
    encoder.add(layers.Dropout(0.2))
    encoder.add(layers.Dense(units=25, activation='relu'))

    # Définition du décodeur
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(units=50, activation='relu', input_shape=[25]))
    decoder.add(layers.Dropout(0.2))  # Dropout pour régularisation
    decoder.add(layers.Dense(units=100, activation='relu'))
    decoder.add(layers.Dense(units=n_features, activation='sigmoid'))

    # Combinaison de l'encodeur et du décodeur dans un autoencodeur
    autoencoder = models.Sequential([encoder, decoder])

    # Compilation du modèle
    autoencoder.compile(
        loss='mean_squared_error',
        optimizer=optimizers.Adam(learning_rate=0.001),
        metrics=['mean_squared_error']
    )

    # Early Stopping
    es = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, restore_best_weights=True)

    # Entraînement du modèle
    history = autoencoder.fit(
        X_train_scaled, X_train_scaled,
        epochs=100,
        batch_size=32,
        validation_data=(X_val_scaled, X_val_scaled),
        callbacks=[es],
        verbose=1
    )
  
    # Sauvegarde du modèle
    current_directory = os.getcwd()  # Obtenir le répertoire actuel
    model_save_name = 'model.h5'  # Nom du fichier modèle
    model_save_path = os.path.join(current_directory, model_save_name)  # Chemin complet de sauvegarde
     
# Vérifier si le fichier existe déjà
    if os.path.exists(model_save_path):
     os.remove(model_save_path)  # Supprimer le fichier existant
     print(f"Le fichier existant {model_save_name} a été supprimé.")

   # Sauvegarder le modèle
    autoencoder.save(model_save_path)
    print(f" Model saved to {model_save_path}")
    

    #la partie du saving du scaler pour using 
    scaler_save_name = 'scaler.joblib'
    scaler_save_path= os.path.join(current_directory,scaler_save_name)
    if os.path.exists(scaler_save_path):
        os.remove(scaler_save_path)
        print(f"Le fichier existant {scaler_save_name} a ete supprimé")
    #sauvegarde du scaler 
    dump(scaler,scaler_save_path)


    return model_save_path
