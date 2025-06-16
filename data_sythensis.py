import pandas as pd
import numpy as np

np.random.seed(42)

# Generate synthetic data
n_samples = 1000
age = np.random.normal(65, 10, n_samples).astype(int)
gender = np.random.choice(['Male', 'Female'], n_samples)
heart_rate = np.random.normal(80, 12, n_samples)
blood_pressure = np.random.normal(120, 15, n_samples)
oxygen = np.random.normal(95, 3, n_samples)
affected_population = (
    0.1 * age + 
    0.3 * heart_rate + 
    0.2 * blood_pressure + 
    0.15 * oxygen +
    np.random.normal(0, 5, n_samples)
)

df = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'HeartRate': heart_rate,
    'BloodPressure': blood_pressure,
    'OxygenSaturation': oxygen,
    'AffectedPopulation': affected_population
})

df.to_csv("global_population_affected_conditions.csv", index=False)
print("Data saved to 'global_population_affected_conditions.csv'")
