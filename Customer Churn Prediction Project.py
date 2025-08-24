# Customer Churn Prediction - Complete Python Project
# Dataset: Telco Customer Churn (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ======================
# 1. Data Preparation
# ======================

# Create synthetic data if you don't have the Kaggle dataset
def generate_sample_data(num_customers=1000):
    np.random.seed(42)
    data = {
        'CustomerID': [f'C{1000+i}' for i in range(num_customers)],
        'Gender': np.random.choice(['Male', 'Female'], num_customers),
        'SeniorCitizen': np.random.choice([0, 1], num_customers, p=[0.8, 0.2]),
        'Partner': np.random.choice(['Yes', 'No'], num_customers),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], num_customers, p=[0.5, 0.3, 0.2]),
        'Tenure': np.random.randint(1, 72, num_customers),
        'MonthlyCharges': np.round(np.random.uniform(20, 120, num_customers), 2),
        'TotalCharges': lambda x: np.round(x['MonthlyCharges'] * x['Tenure'], 2),
        'Churn': np.random.choice(['Yes', 'No'], num_customers, p=[0.3, 0.7])
    }
    df = pd.DataFrame(data)
    df['TotalCharges'] = df['MonthlyCharges'] * df['Tenure']
    return df

# Load data (replace with your actual data)
try:
    df = pd.read_csv('telco_churn.csv')
    print("Loaded real dataset")
except:
    df = generate_sample_data()
    print("Generated synthetic data")

# ======================
# 2. Data Preprocessing
# ======================

# Convert categorical variables
label_encoders = {}
categorical_cols = ['Gender', 'Partner', 'Contract', 'Churn']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Handle missing values
df.fillna(0, inplace=True)

# Feature selection
features = ['Gender', 'SeniorCitizen', 'Partner', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'Contract']
X = df[features]
y = df['Churn']

# ======================
# 3. Exploratory Analysis
# ======================

print("\n=== Basic Statistics ===")
print(df.describe())

print("\n=== Churn Rate ===")
print(f"Overall churn rate: {df['Churn'].mean():.2%}")

plt.figure(figsize=(12, 5))

# Plot 1: Churn by contract type
plt.subplot(1, 2, 1)
sns.barplot(x='Contract', y='Churn', data=df)
plt.title('Churn Rate by Contract Type')
plt.xlabel('Contract (0=Monthly, 1=1yr, 2=2yr)')
plt.ylabel('Churn Probability')

# Plot 2: Monthly charges distribution
plt.subplot(1, 2, 2)
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges vs Churn')
plt.xlabel('Churn (0=No, 1=Yes)')

plt.tight_layout()
plt.show()

# ======================
# 4. Model Building
# ======================

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\n=== Model Performance ===")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ======================
# 5. Feature Importance
# ======================

importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n=== Feature Importance ===")
print(importance)

plt.figure(figsize=(10, 5))
sns.barplot(x='Importance', y='Feature', data=importance)
plt.title('Feature Importance')
plt.show()

# ======================
# 6. Prediction Example
# ======================

# Sample customer data
new_customer = {
    'Gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Tenure': 24,
    'MonthlyCharges': 80.50,
    'TotalCharges': 1932.00,
    'Contract': 'One year'
}

# Preprocess new data
new_df = pd.DataFrame([new_customer])
for col in categorical_cols:
    if col in new_df.columns:
        new_df[col] = label_encoders[col].transform(new_df[col])

# Make prediction
prediction = model.predict(new_df[features])
prob = model.predict_proba(new_df[features])[0][1]

print("\n=== Prediction for New Customer ===")
print(f"Features: {new_customer}")
print(f"Will churn?: {'Yes' if prediction[0] == 1 else 'No'}")
print(f"Probability of churning: {prob:.2%}")