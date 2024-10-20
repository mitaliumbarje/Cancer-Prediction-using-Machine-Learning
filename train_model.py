import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load your data
data = pd.read_csv('data/your_data.csv')

# Separate features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Print current working directory
print("Current working directory:", os.getcwd())

# Create 'model' directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')
    print("Model directory created.")

# Save the model
try:
    joblib.dump(model, 'model/cancer_detection_model.pkl')
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")
