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

app = Flask(__name__, static_url_path='')
CORS(app)

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


def train_and_save_models():
    global model, premium_model, label_encoders, model_accuracy

    print("Training model...")
    data = pd.read_csv(DATA_FILE)

    features = ['Age', 'Gender', 'Income', 'Health_Status', 'Smoking_Habit', 'Policy_Type']
    target = 'Prediction_Target'

    cols = ['Gender', 'Health_Status', 'Smoking_Habit', 'Policy_Type']
    for col in cols:
        data[col] = data[col].fillna("Unknown").astype(str).str.capitalize()

        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    X = data[features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_pred)

    if 'Premium_Amount' in data:
        premium_model = RandomForestClassifier(random_state=42)
        premium_model.fit(X_train, data.loc[X_train.index, 'Premium_Amount'])
    else:
        from sklearn.dummy import DummyRegressor
        premium_model = DummyRegressor(strategy="mean")
        premium_model.fit(X_train, pd.Series(5000, index=X_train.index))

    dump({
        'model': model,
        'premium_model': premium_model,
        'label_encoders': label_encoders,
        'accuracy': model_accuracy
    }, MODEL_FILE)

    print("Model trained & saved.")


def load_models():
    global model, premium_model, label_encoders, model_accuracy

    if os.path.exists(MODEL_FILE):
        print("Loading saved model...")
        obj = load(MODEL_FILE)
        model = obj['model']
        premium_model = obj['premium_model']
        label_encoders = obj['label_encoders']
        model_accuracy = obj['accuracy']
        return True
    return False


if not load_models():
    if os.path.exists(DATA_FILE):
        train_and_save_models()
    else:
        print("Dataset missing!")


def encode(col, val):
    le = label_encoders.get(col)
    if not le:
        return val
    val = str(val).capitalize()
    if val in le.classes_:
        return le.transform([val])[0]
    return le.transform(['Unknown'])[0]


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    age = int(data.get('age', 0))
    gender = data.get('gender', 'Unknown')
    income = float(data.get('income', 0))
    health = data.get('health', 'Unknown')
    smoke = data.get('smoke', 'Unknown')

    if age < 18 and smoke == "Yes":
        return jsonify({
            "eligible": False,
            "message": "Not eligible due to underage smoking.",
            "suggestions": "Reapply after age 18.",
            "model_accuracy": model_accuracy
        })

    enc_gender = encode("Gender", gender)
    enc_health = encode("Health_Status", health)
    enc_smoke = encode("Smoking_Habit", smoke)

    policies = []
    if income > 100000 and health == "Excellent":
        policies = ["Whole", "Universal", "Term"]
    elif income > 50000:
        policies = ["Universal", "Term"]
    elif income > 5000:
        policies = ["Term"]
    else:
        return jsonify({
            "eligible": False,
            "message": "Income too low.",
            "suggestions": "Increase your financial status.",
            "model_accuracy": model_accuracy
        })

    premium_estimates = {}
    for p in policies:
        enc_policy = encode("Policy_Type", p)
        df = pd.DataFrame([[age, enc_gender, income, enc_health, enc_smoke, enc_policy]],
                          columns=['Age', 'Gender', 'Income', 'Health_Status', 'Smoking_Habit', 'Policy_Type'])
        premium_estimates[p] = float(premium_model.predict(df)[0])

    return jsonify({
        "eligible": True,
        "message": "You are eligible!",
        "policies": policies,
        "premiums": premium_estimates,
        "suggestions": "Choose a policy wisely.",
        "model_accuracy": model_accuracy
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
