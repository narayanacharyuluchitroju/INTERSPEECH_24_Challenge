import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



def extract_audio_features(file_path):
    # Load audio file
    y, sr = librosa.load(file_path)

    # Extract MFCCs (Mel-frequency cepstral coefficients)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

    # Extract Chroma Features
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)

    # Extract Spectral Contrast
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)

    # Extract Spectral Roll-off
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # Extract Zero Crossing Rate
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))

    # Extract Root Mean Square Energy (RMSE)
    rmse = np.mean(librosa.feature.rms(y=y))

    # Additional features can be added based on your requirements

    return mfccs, chroma, spectral_contrast, spectral_rolloff, zero_crossing_rate, rmse

def process_audio_directory(directory_path):
    audio_features = []

    # Iterate through all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".wav"):  # Assuming your audio files are in WAV format
            file_path = os.path.join(directory_path, filename)
            mfccs, chroma, spectral_contrast, spectral_rolloff, zero_crossing_rate, rmse = extract_audio_features(file_path)
            audio_features.append((filename,mfccs, chroma, spectral_contrast, spectral_rolloff, zero_crossing_rate, rmse))

    return audio_features



def train_model(data, features, target,fname):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'),
             features),
        ])

    # Classifier
    classifier = RandomForestClassifier(random_state=2)

    # Create the model pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    # Fit the model
    model.fit(X_train, y_train)

    joblib.dump(model, "saved_models/{}".format(fname))

    return model, X_test, y_test


def test_model(model, X_test, y_test):
    # Predictions
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred)
    # Print actual vs. predicted values
    print("Actual vs. Predicted:")
    for actual, predicted in zip(y_test, y_pred):
        print(actual, predicted)

    # Evaluate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    return y_test,y_pred,report

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_feature_importance(model, features):
    importances = model.named_steps['classifier'].feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(features)), importances[indices], align='center')
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.show()

