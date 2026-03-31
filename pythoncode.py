import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. DATA REPRESENTATION (Syllabus CO4/CO5)
# This mimics the "Data Representations" mentioned in your syllabus 
data = {
    'RAM_GB': [4, 8, 8, 16, 16, 32, 4, 8, 12, 16, 64, 8, 16, 32, 8],
    'CPU_Speed_GHz': [2.1, 2.4, 2.8, 3.2, 3.5, 3.8, 1.8, 2.4, 2.6, 3.0, 4.2, 2.5, 3.1, 3.9, 2.2],
    'SSD_GB': [128, 256, 512, 512, 1024, 1024, 128, 256, 256, 512, 2048, 256, 512, 1024, 256],
    'Price_INR': [30000, 45000, 55000, 80000, 95000, 150000, 25000, 42000, 50000, 78000, 250000, 44000, 82000, 145000, 41000]
}

df = pd.DataFrame(data)

# 2. FEATURE SELECTION (Syllabus: Feature Learning)
X = df[['RAM_GB', 'CPU_Speed_GHz', 'SSD_GB']]
y = df['Price_INR']

# 3. TRAINING & VALIDATION SETS (Syllabus: Validation Sets) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. MODEL BUILDING: Linear Regression (Syllabus: Supervised Learning) 
model = LinearRegression()
model.fit(X_train, y_train)

# 5. MODEL EVALUATION
y_pred = model.predict(X_test)
print("--- Model Performance Metrics ---")
print(f"Mean Absolute Error: ₹{mean_absolute_error(y_test, y_pred):.2f}")
print(f"R-squared Score: {r2_score(y_test, y_pred):.4f}")
print("-" * 40)

# 6. INTERACTIVE USER INPUT (This part waits for you in the terminal)
print("\n[AI Agent]: Ready to predict. Please enter specifications:")

try:
    # These lines will stop the execution and wait for your input
    input_ram = float(input("Step 1: Enter RAM size in GB (e.g., 16): "))
    input_cpu = float(input("Step 2: Enter CPU Speed in GHz (e.g., 3.2): "))
    input_ssd = float(input("Step 3: Enter SSD Capacity in GB (e.g., 512): "))

    # Creating a DataFrame to avoid "Feature Names" warnings seen in your terminal
    user_specs = pd.DataFrame([[input_ram, input_cpu, input_ssd]], 
                              columns=['RAM_GB', 'CPU_Speed_GHz', 'SSD_GB'])
    
    # Model Prediction
    predicted_price = model.predict(user_specs)

    print(f"\n--- Prediction Result ---")
    print(f"Based on the input, the predicted price is: ₹{predicted_price[0]:.2f}")

except ValueError:
    print("\nError: Please enter only numbers (e.g., 16 instead of '16GB').")
