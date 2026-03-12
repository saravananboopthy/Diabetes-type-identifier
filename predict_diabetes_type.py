import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score

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

def main():
    download_dataset()
    
    # Define columns
    columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    df = pd.read_csv(DATASET_PATH, names=columns)
    
    # 2. Add 'Diabetes Types' to simulate Westat EHR type classification
    # Outcome: 0 = No Diabetes, 1 = Diabetes
    # We derive Type 1 and Type 2 from outcome 1 to model multiple diabetes types.
    # Typically Type 1 has earlier onset and lower BMI, Type 2 is older/higher BMI.
    def classify_type(row):
        if row['Outcome'] == 0:
            return 0 # No Diabetes
        else:
            if row['Age'] <= 30 and row['BMI'] <= 30:
                return 1 # Type 1 Diabetes
            else:
                return 2 # Type 2 Diabetes
                
    df['Diabetes_Type'] = df.apply(classify_type, axis=1)
    
    print("\nClass distribution (Imbalanced Medical Data):")
    print(df['Diabetes_Type'].value_counts())
    
    X = df.drop(['Outcome', 'Diabetes_Type'], axis=1)
    y = df['Diabetes_Type']
    
    # 3. Train-Test Split (stratified for class imbalance)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Handle Imbalanced Data using Class Weights
    print("\nUsing Class Weights 'balanced' to handle imbalanced medical data...")
    
    # 5. Train Model (Decision Tree as Python proxy for Conditional Inference Trees)
    print("Training Decision Tree Classifier (Conditional Inference proxy)...")
    clf = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    
    # 6. Evaluation
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    
    print("\n--- Model Validation ---")
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Type 1", "Type 2"], zero_division=0))
    
    # ROC AUC for multi-class
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        print(f"ROC AUC Score (OVR): {auc:.4f}")
    except ValueError as e:
        print("Could not compute ROC AUC due to class distribution in test set.")
    
    print("\nProcess finished successfully. The dataset is saved directly in the folder.")

if __name__ == "__main__":
    main()
