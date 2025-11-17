# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
import os
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODEL_FILE = "models.joblib"
DATA_FILE = "life_insurance_prediction.csv"

app = Flask(__name__)
CORS(app)

# global holders
model = None
premium_model = None
label_encoders = {}
model_accuracy = None

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

def safe_label_fit_transform(series):
    le = LabelEncoder()
    # fill NaN and cast to str
    s = series.fillna("Unknown").astype(str)
    le.fit(s)
    return le, le.transform(s)

def train_and_save_models():
    global model, premium_model, label_encoders, model_accuracy

    print("Training models (this happens only if models.joblib doesn't exist)...")
    data = pd.read_csv(DATA_FILE)

    features = ['Age', 'Gender', 'Income', 'Health_Status', 'Smoking_Habit', 'Policy_Type']
    target = 'Prediction_Target'

    # standardize capitalization & handle missing
    for col in ['Gender', 'Health_Status', 'Smoking_Habit', 'Policy_Type']:
        if col in data.columns:
            data[col] = data[col].fillna("Unknown").astype(str).str.capitalize()
        else:
            data[col] = "Unknown"

    # Fit label encoders on training data
    for col in ['Gender', 'Health_Status', 'Smoking_Habit', 'Policy_Type']:
        le = LabelEncoder()
        data[col] = data[col].astype(str)
        le.fit(data[col])
        data[col] = le.transform(data[col])
        label_encoders[col] = le

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_local = RandomForestClassifier(random_state=42)
    model_local.fit(X_train, y_train)
    y_pred = model_local.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Premium model: ensure Premium_Amount exists
    if 'Premium_Amount' in data.columns:
        premium_local = RandomForestClassifier(random_state=42)
        premium_local.fit(X_train, data.loc[X_train.index, 'Premium_Amount'])
    else:
        # fallback: create a dummy premium model that predicts a constant
        from sklearn.dummy import DummyRegressor
        premium_local = DummyRegressor(strategy="mean")
        premium_local.fit(X_train, data.loc[X_train.index].get('Premium_Amount', pd.Series(0, index=X_train.index)))

    # assign globals
    model = model_local
    premium_model = premium_local
    model_accuracy = float(acc)

    # save to disk
    dump({
        'model': model,
        'premium_model': premium_model,
        'label_encoders': label_encoders,
        'accuracy': model_accuracy
    }, MODEL_FILE)
    print("Models trained and saved to", MODEL_FILE)

def load_models_if_present():
    global model, premium_model, label_encoders, model_accuracy
    if os.path.exists(MODEL_FILE):
        print("Loading models from", MODEL_FILE)
        obj = load(MODEL_FILE)
        model = obj['model']
        premium_model = obj['premium_model']
        label_encoders = obj['label_encoders']
        model_accuracy = float(obj.get('accuracy', 0.0))
        return True
    return False

# initialize at startup
if not load_models_if_present():
    if os.path.exists(DATA_FILE):
        train_and_save_models()
    else:
        print(f"Warning: {DATA_FILE} not found. Prediction will not work until data & training are available.")

def encode_input_value(col, value):
    """Encode a single categorical value using a fitted LabelEncoder.
       If unseen, add it as 'Unknown' mapping if 'Unknown' exists, else map to -1."""
    le = label_encoders.get(col)
    if le is None:
        return value
    val_str = str(value).capitalize()
    try:
        return le.transform([val_str])[0]
    except Exception:
        # if 'Unknown' present, use it
        if 'Unknown' in le.classes_:
            return int(le.transform(['Unknown'])[0])
        else:
            # fallback to most frequent class (index 0)
            return 0

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or premium_model is None:
        return jsonify({
            'eligible': False,
            'message': 'Model is not available. Ensure training completed and models.joblib exists.',
            'model_accuracy': None
        }), 500

    try:
        data = request.get_json()
        age = int(data.get('age', 0))
        gender = str(data.get('gender', 'Unknown')).capitalize()
        income = float(data.get('income', 0))
        health_status = str(data.get('health', 'Unknown')).capitalize()
        smoking = str(data.get('smoke', 'Unknown')).capitalize()
    except Exception as e:
        return jsonify({'error': 'Invalid input format', 'detail': str(e)}), 400

    # Underage & smoking check
    if age < 18 and smoking == "Yes":
        return jsonify({
            'eligible': False,
            'message': 'Not Eligible for Insurance - Underage smoking detected.',
            'suggestions': 'It is recommended to adopt a healthier lifestyle and reapply after turning 18.',
            'model_accuracy': model_accuracy
        })

    # Build input dataframe
    input_data = pd.DataFrame([[age, gender, income, health_status, smoking, 'None']],
                              columns=['Age', 'Gender', 'Income', 'Health_Status', 'Smoking_Habit', 'Policy_Type'])

    # encode categories safely
    for col in ['Gender', 'Health_Status', 'Smoking_Habit']:
        input_data[col] = input_data[col].apply(lambda v: encode_input_value(col, v))

    # policy eligibility logic (same as original)
    if income > 100000 and health_status == 'Excellent':
        eligible_policies = ['Whole', 'Universal', 'Term']
    elif income > 50000 and health_status in ['Good', 'Average', 'Excellent']:
        eligible_policies = ['Universal', 'Term']
    elif income > 5000:
        eligible_policies = ['Term']
    else:
        return jsonify({
            'eligible': False,
            'message': 'Not Eligible for Insurance - Income is below the minimum threshold of 5000.',
            'suggestions': 'Consider increasing your income or opting for alternative financial security options.',
            'model_accuracy': model_accuracy
        })

    premium_estimates = {}
    for policy in eligible_policies:
        # encode policy type safely
        policy_encoded = encode_input_value('Policy_Type', policy)
        input_data['Policy_Type'] = policy_encoded
        # premium prediction expects same feature columns
        try:
            pred = premium_model.predict(input_data)[0]
            premium_estimates[policy] = float(pred)
        except Exception:
            premium_estimates[policy] = None

    # Suggestions
    suggestions = "Make sure to pay premiums on time to avoid policy lapse. "
    if "Whole" in eligible_policies:
        suggestions += "Whole life policies require consistent payments. If missed, your policy might lapse, but some offer cash value. "
    if "Universal" in eligible_policies:
        suggestions += "Universal policies offer flexible premiums. Missing payments may impact benefits, so monitor your cash value. "
    if "Term" in eligible_policies:
        suggestions += "Term insurance has no cash value. If you miss payments, coverage stops. Consider setting up auto-payments."

    response = {
        'eligible': True,
        'message': 'You are eligible for life insurance!',
        'policies': eligible_policies,
        'premiums': premium_estimates,
        'suggestions': suggestions,
        'model_accuracy': model_accuracy
    }
    return jsonify(response)


if __name__ == '__main__':
    # debug True for development only
    app.run(debug=True, port=5000)
