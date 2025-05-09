import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import os

df = pd.read_csv('loan_sanction_train.csv')

df['Dependents'] = df['Dependents'].replace('3+', '3')
df['Dependents'] = df['Dependents'].fillna('0')
df['Dependents'] = df['Dependents'].astype(int)

df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
            'Loan_Amount_Term', 'Credit_History', 'Property_Area']
target = 'Loan_Status'
X = df[features]
y = df[target]

imputer = SimpleImputer(strategy='most_frequent')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=features)

label_encoders = {}
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

X = X.astype({
    'Gender': int, 'Married': int, 'Dependents': int, 'Education': int,
    'Self_Employed': int, 'ApplicantIncome': float, 'CoapplicantIncome': float,
    'LoanAmount': float, 'Loan_Amount_Term': float, 'Credit_History': float,
    'Property_Area': int
})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

os.makedirs('model', exist_ok=True)
with open('model/loan_model.pkl', 'wb') as f:
    pickle.dump({
        'model': model,
        'label_encoders': label_encoders,
        'imputer': imputer
    }, f)

print("Model trained and saved successfully!")