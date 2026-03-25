import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# 1. Download Dataset if it doesn't exist
DATASET_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
DATASET_PATH = "diabetes_dataset.csv"

def download_dataset():
    if not os.path.exists(DATASET_PATH):
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)
        print("Dataset downloaded and saved as:", DATASET_PATH)
    else:
        print("Dataset already exists in the folder.")

def train_and_save_model():
    download_dataset()
    
    # Define columns
    columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    df = pd.read_csv(DATASET_PATH, names=columns)
    
    # 2. Add 'Diabetes Types' to simulate Westat EHR type classification
    def classify_type(row):
        if row['Outcome'] == 0:
            return 0 # No Diabetes
        else:
            if row['Age'] <= 30 and row['BMI'] <= 30:
                return 1 # Type 1 Diabetes
            else:
                return 2 # Type 2 Diabetes
                
    df['Diabetes_Type'] = df.apply(classify_type, axis=1)
    
    X = df.drop(['Outcome', 'Diabetes_Type'], axis=1)
    y = df['Diabetes_Type']
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Model
    clf = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(clf, 'diabetes_model.pkl')
    print("Model saved as diabetes_model.pkl")
    
    return clf

if __name__ == "__main__":
    train_and_save_model()

