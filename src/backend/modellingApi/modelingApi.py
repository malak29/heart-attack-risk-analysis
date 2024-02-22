from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

# Your existing functions here (e.g., load_data, clean_data, etc.)

# Flask route to load data
@app.route('/load_data', methods=['POST'])
def api_load_data():
    file_path = request.json['file_path']
    df = load_data(file_path)
    return jsonify({"message": "Data loaded successfully", "shape": df.shape})

# Flask route for data cleaning
@app.route('/clean_data', methods=['POST'])
def api_clean_data():
    file_path = request.json['file_path']
    df = load_data(file_path)
    df_clean = clean_data(df)
    return jsonify({"message": "Data cleaned successfully", "shape": df_clean.shape})

# Add more Flask routes for other functionalities like EDA, model training, etc.

# Flask route to train RandomForest model
@app.route('/train_random_forest', methods=['POST'])
def api_train_random_forest():
    data = request.json
    df = pd.DataFrame(data)
    X = df.drop('Heart Attack Risk', axis=1)
    y = df['Heart Attack Risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return jsonify({"message": "Random Forest model trained", "accuracy": accuracy})

# Flask route to train GradientBoosting model
@app.route('/train_gradient_boosting', methods=['POST'])
def api_train_gradient_boosting():
    data = request.json
    df = pd.DataFrame(data)
    X = df.drop('Heart Attack Risk', axis=1)
    y = df['Heart Attack Risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return jsonify({"message": "Gradient Boosting model trained", "accuracy": accuracy})

if __name__ == '__main__':
    app.run(debug=True)