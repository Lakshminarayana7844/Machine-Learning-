import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Load the dataset
# Replace 'car_data.csv' with the path to your dataset
df = pd.read_csv('/Users/lakshminarayanamandi/Downloads/CarPricesPrediction.csv')



X = df[['Year', 'Mileage']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


# Function to predict car price based on year and mileage
def predict_car_price(year, mileage):
    prediction = model.predict([[year, mileage]])
    return prediction[0]

# Example usage of predict_car_price
year = 2015
mileage = 50000
predicted_price = predict_car_price(year, mileage)
print(f"Predicted price for a car from {year} with {mileage} miles is ${predicted_price:.2f}")
