# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pycountry
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import GridSearchCV, train_test_split

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Data Cleaning Functions
def clean_continent_column(data, column='Continent'):
    data[column] = data[column].str.title()
    return data

def clean_data(df):
    # Add data cleaning steps here
    df = clean_continent_column(df)
    # Add other cleaning functions as needed
    return df

# EDA Functions
def plot_boxplots(df):
    # Add code to plot boxplots
    pass

def plot_heatmap(df):
    # Add code to plot heatmap
    pass

# Model Training and Evaluation Functions
def train_random_forest(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    evaluate_model(y_test, y_pred)

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    gb = GradientBoostingClassifier(random_state=42)
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)
    evaluate_model(y_test, y_pred)

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

# Main function to orchestrate data loading, cleaning, EDA, and model training
def main():
    file_path = 'heart_attack_prediction_dataset.csv'
    df = load_data(file_path)
    df_clean = clean_data(df)
    
    # Perform EDA
    plot_boxplots(df_clean)
    plot_heatmap(df_clean)
    
    # Prepare data for modeling
    X = df_clean.drop('Heart Attack Risk', axis=1)
    y = df_clean['Heart Attack Risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    train_random_forest(X_train, y_train, X_test, y_test)
    train_gradient_boosting(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    main()