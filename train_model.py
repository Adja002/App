import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset
def load_dataset(filepath):
    try:
        df = pd.read_csv(filepath)
        logging.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        raise

df = load_dataset('data/Senegal_Misinformation_Dataset.csv')

# Display information about the dataset
logging.info("\nDataset Information:")
logging.info(df.info())

# Preprocess the dataset
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])
        logging.info(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

logging.info("Preprocessing completed.")

# Create new features
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 30, 50, 100], labels=['0-18', '19-30', '31-50', '51+'])
df['Age_Group'] = le.fit_transform(df['Age_Group'])

# Prepare the target variable
y = df['Believability_in_Misinformation']

# Define X as all columns except 'Believability_in_Misinformation'
X = df.drop('Believability_in_Misinformation', axis=1)

logging.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use XGBoost with GridSearchCV for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 0.9, 1.0]
}

xgb_model = xgb.XGBClassifier(random_state=42)
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

logging.info("Model training completed.")

# Evaluate model performance
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Model Accuracy: {accuracy}")

logging.info("\nClassification Report:")
logging.info(classification_report(y_test, y_pred))

# Save the model and feature columns
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')

logging.info("Model and feature columns saved successfully.")

# Print feature importances
importances = best_model.feature_importances_
feature_importance = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)
logging.info("\nFeature Importances:")
for feature, importance in feature_importance:
    logging.info(f"{feature}: {importance}")
