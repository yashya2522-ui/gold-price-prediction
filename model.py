import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib

print("Indian Gold Price Prediction Started")

# Load dataset
df = pd.read_csv("data.csv")

print(df.head())

# Remove Date column
if 'Date' in df.columns:
    df = df.drop(['Date'], axis=1)

# Features and Target
X = df.drop(['GoldPrice'], axis=1)
Y = df['GoldPrice']

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# Model
model = RandomForestRegressor(n_estimators=100)

# Train model
model.fit(X_train, Y_train)

print("Training Completed")

# Prediction
prediction = model.predict(X_test)

# Accuracy
score = r2_score(Y_test, prediction)

print("Accuracy :", score)

# Save model
joblib.dump(model, "indian_gold_model.pkl")

print("Indian Gold Model Saved Successfully")