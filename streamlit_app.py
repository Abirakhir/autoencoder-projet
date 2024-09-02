import streamlit as st
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from joblib import load

# Fonction pour détecter les anomalies avec l'autoencodeur
def detect_anomalies(autoencoder, data, scaler, threshold):
    data_scaled = scaler.transform(data)
    reconstructions = autoencoder.predict(data_scaled)
    errors = np.mean(np.abs(reconstructions - data_scaled), axis=1)
    anomalies = (errors > threshold).astype(int)
    return anomalies, errors

# Charger le modèle d'autoencodeur
current_directory = os.getcwd()
model_name = 'model.h5'
scaler_name = 'scaler.joblib'
chemin_modele = os.path.join(current_directory, model_name)
chemin_scaler = os.path.join(current_directory, scaler_name)

try:
    autoencoder = load_model(chemin_modele)
    st.success("Modèle chargé avec succès.")
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")

try:
    scaler = load(scaler_name)
    st.success("Scaler chargé avec succès.")
except Exception as e:
    st.error(f"Erreur lors du chargement du scaler : {e}")

st.title("Détection de Fraude avec Autoencodeur")

# Choix entre utiliser un DataFrame de test pré-défini ou entrer des transactions
option = st.selectbox("Choisissez une option :", ["Utiliser un fichier de test", "Entrer des transactions"])

if option == "Utiliser un fichier de test":
    uploaded_file = st.file_uploader("Choisissez un fichier JSON ou CSV", type=["json", "csv"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)

        # Nettoyage des données pour supprimer les guillemets simples entourant certaines valeurs
        df = df.applymap(lambda x: x.strip("'") if isinstance(x, str) else x)

        if 'Class' in df.columns:
            normal_data = df[df['Class'] == 0].drop(columns=['Class'])
            fraud_data = df[df['Class'] == 1].drop(columns=['Class'])

            val_data = normal_data.sample(frac=0.2, random_state=42)
            normal_train_data = normal_data.drop(val_data.index)

            val_data_scaled = scaler.transform(val_data)
            val_reconstructions = autoencoder.predict(val_data_scaled)
            val_errors = np.mean(np.abs(val_reconstructions - val_data_scaled), axis=1)
            threshold = np.percentile(val_errors, 99.20)

            anomalies, errors = detect_anomalies(autoencoder, pd.concat([normal_data, fraud_data]), scaler, threshold)
            df['Anomaly'] = np.concatenate([anomalies[:len(normal_data)], anomalies[len(normal_data):]])

            st.subheader("Analyse des Anomalies")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Distribution des Erreurs de Reconstruction")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(errors, bins=50, kde=True)
                plt.axvline(threshold, color='r', linestyle='--', label='Threshold')
                plt.xlabel('Reconstruction Error')
                plt.ylabel('Frequency')
                plt.legend()
                st.pyplot(fig)

            with col2:
                st.subheader("Transactions Anormales Détectées")
                st.write(df[df['Anomaly'] == 1])

            st.subheader("Matrice de Confusion et Rapport de Classification")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Matrice de Confusion")
                y_true = np.concatenate([np.zeros(len(normal_data)), np.ones(len(fraud_data))])
                cm = confusion_matrix(y_true, df['Anomaly'])
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'], ax=ax)
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                st.pyplot(fig)

            with col2:
                st.subheader("Rapport de Classification")
                report = classification_report(y_true, df['Anomaly'], output_dict=True)
                st.write(pd.DataFrame(report).transpose())

            st.subheader("Statistiques de Prédiction")
            st.write(f"Nombre de transactions anormales détectées : {np.sum(df['Anomaly'] == 1)}")
            st.write(f"F1-score : {f1_score(y_true, df['Anomaly']):.4f}")

        else:
         st.write("Prédictions sur les transactions")
         threshold = 0.02
         anomalies, errors = detect_anomalies(autoencoder, df, scaler, threshold)
         df['Anomaly'] = anomalies
         col1, col2 = st.columns(2)

         with col1: 
          st.subheader("Transactions Anormales détectées")
          st.write(df[df['Anomaly'] == 1])
        
         with col2:
           st.subheader("Répartition des transactions")
           plt.figure(figsize=(8, 8))  # Taille ajustée pour éviter le chevauchement
           anomaly_counts = df['Anomaly'].value_counts()
           plt.pie(anomaly_counts, labels=['Normal', 'Fraud'], autopct='%1.1f%%', colors=['blue', 'grey'])
           st.pyplot(plt)

elif option == "Entrer des transactions":
    st.write("Entrez les valeurs des transactions à prédire en une seule ligne, séparées par des virgules :")

    num_transactions = st.number_input("Entrez le nombre de transactions à prédire", min_value=1, value=1)
    
    if num_transactions > 0:
        transactions_input = st.text_area("Entrez les valeurs des transactions (une ligne par transaction) :", value='')

        if transactions_input:
            data_entries = [list(map(float, line.split(','))) for line in transactions_input.strip().split('\n')]
            columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
            
            if all(len(entry) == len(columns) for entry in data_entries):
                input_data = pd.DataFrame(data_entries, columns=columns)
                data_scaled = scaler.transform(input_data)
                reconstructions = autoencoder.predict(data_scaled)
                errors = np.mean(np.abs(reconstructions - data_scaled), axis=1)
                threshold = np.percentile(errors, 99.20)
                anomalies, _ = detect_anomalies(autoencoder, input_data, scaler, threshold)
                input_data['Anomaly'] = anomalies

                st.subheader("Résultats de Prédiction")
                results = input_data.copy()
                results['Status'] = results['Anomaly'].apply(lambda x: 'Frauduleuse' if x == 1 else 'Normale')
                st.write(results[['Status']])

            else:
                st.error("Les données entrées ne correspondent pas au nombre de colonnes attendues. Veuillez vérifier les entrées.")
