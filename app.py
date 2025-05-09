from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

def load_model():
    with open('model/loan_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data

data = load_model()
model = data['model']
label_encoders = data['label_encoders']
imputer = data['imputer']

@app.route('/', methods=['GET', 'POST'])
def index():
    form_data = None
    if request.method == 'POST':
        form_data = {
            'gender': request.form['gender'],
            'married': request.form['married'],
            'dependents': request.form['dependents'],
            'education': request.form['education'],
            'self_employed': request.form['self_employed'],
            'applicant_income': request.form['applicant_income'],
            'coapplicant_income': request.form['coapplicant_income'],
            'loan_amount': request.form['loan_amount'],
            'loan_term': request.form['loan_term'],
            'credit_history': request.form['credit_history'],
            'property_area': request.form['property_area']
        }

        input_data = np.array([
            form_data['gender'], form_data['married'], form_data['dependents'], 
            form_data['education'], form_data['self_employed'],
            float(form_data['applicant_income']), float(form_data['coapplicant_income']), 
            float(form_data['loan_amount']), float(form_data['loan_term']), 
            float(form_data['credit_history']), form_data['property_area']
        ]).reshape(1, -1)

        input_data[0, 0] = label_encoders['Gender'].transform([input_data[0, 0]])[0]  # Gender
        input_data[0, 1] = label_encoders['Married'].transform([input_data[0, 1]])[0]  # Married
        input_data[0, 3] = label_encoders['Education'].transform([input_data[0, 3]])[0]  # Education
        input_data[0, 4] = label_encoders['Self_Employed'].transform([input_data[0, 4]])[0]  # Self_Employed
        input_data[0, 10] = label_encoders['Property_Area'].transform([input_data[0, 10]])[0]  # Property_Area

        input_data = input_data.astype(float)

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        result = "Approved" if prediction == 1 else "Not Approved"
        confidence = round(probability * 100, 2)

        return render_template('index.html', 
                             result=result, 
                             confidence=confidence,
                             show_result=True,
                             form_data=form_data)

    return render_template('index.html', show_result=False)

if __name__ == '__main__':
    app.run(debug=True)