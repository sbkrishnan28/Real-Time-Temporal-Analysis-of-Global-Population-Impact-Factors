import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix

# Load dataset
df = pd.read_csv(r"C:\vscode\People and their life affected prediction\global_population_affected_conditions.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Set target column
target_column = 'Affected by Diseases (Millions)'

# Validate target column
if target_column not in df.columns:
    raise ValueError(f"❌ Column '{target_column}' not found in dataset.")

# Features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation metrics
print("\n✅ Model Evaluation:")
print("MAE :", round(mean_absolute_error(y_test, y_pred), 3))
print("MSE :", round(mean_squared_error(y_test, y_pred), 3))
print("R²  :", round(r2_score(y_test, y_pred), 3))

# 📊 Full Correlation Matrix
print("\n📊 Full Correlation Matrix:")
correlation_matrix = df.corr(numeric_only=True)
print(correlation_matrix.round(3))

# 🔍 Strong Correlations
print("\n🔍 Strong Correlations (> 0.7 or < -0.7):")
strong_corrs = correlation_matrix[((correlation_matrix > 0.7) | (correlation_matrix < -0.7)) & (correlation_matrix != 1.0)]
print(strong_corrs.dropna(how='all').dropna(axis=1, how='all').round(3))

# 📈 Year-wise Trend Analysis
plt.figure(figsize=(12, 6))
yearly_data = df.groupby('Year').mean(numeric_only=True)

features_to_plot = [
    'Affected by Diseases (Millions)',
    'Affected by Viruses (Millions)',
    'Natural Disasters Affected (Millions)',
    'Conflict Zones Affected (Millions)',
    'Economic Downturns Affected (Millions)'
]

for feature in features_to_plot:
    if feature in yearly_data.columns:
        plt.plot(yearly_data.index, yearly_data[feature], marker='o', label=feature)

plt.title("📈 Year-wise Life Impact Trend")
plt.xlabel("Year")
plt.ylabel("People Affected (Millions)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 📊 Actual vs Predicted Plot
plt.figure(figsize=(10, 5))
plt.plot(range(len(y_test)), y_test.values, label="Actual", marker='o')
plt.plot(range(len(y_pred)), y_pred, label="Predicted", marker='x')
plt.title("📊 Actual vs Predicted Affected Population")
plt.xlabel("Sample Index")
plt.ylabel(target_column)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 🧩 Confusion Matrix (Binary Classification)
threshold = y.median()
y_test_cls = (y_test > threshold).astype(int)
y_pred_cls = (y_pred > threshold).astype(int)
cm = confusion_matrix(y_test_cls, y_pred_cls)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("🧩 Confusion Matrix (Binary)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
