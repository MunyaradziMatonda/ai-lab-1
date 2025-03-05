import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

# Step 1: Create a simple dataset
# Let's predict house prices based on house size (in square feet)
data = {
    'Size': [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
    'Price': [300000, 400000, 550000, 700000, 850000, 1000000, 1150000, 1300000]
}
df = pd.DataFrame(data)

# Step 2: Prepare the data
X = df[['Size']]  # Feature (input)
y = df['Price']   # Target (output)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make predictions
y_pred = model.predict(X_test)

# Step 6: Display the results
print("Test Sizes:", X_test.values.flatten())
print("Actual Prices:", y_test.values)
print("Predicted Prices:", y_pred)

# Step 7: Save the model for later use
joblib.dump(model, 'linear_regression_model.pkl')
print("Model saved as 'linear_regression_model.pkl'")