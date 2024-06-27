import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Sample dataset
data = {
    'Size': [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400],
    'Bedrooms': [3, 3, 3, 4, 4, 4, 5, 5, 5, 5],
    'Price': [300000, 320000, 340000, 360000, 380000, 400000, 420000, 440000, 460000, 480000]
}

# Load the dataset into a pandas DataFrame
df = pd.DataFrame(data)

# Select features and target
X = df[['Size', 'Bedrooms']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


def predict_house_price(size, bedrooms):
    prediction = model.predict([[size, bedrooms]])
    return prediction[0]

# Example usage of predict_house_price
size = 2500
bedrooms = 4
predicted_price = predict_house_price(size, bedrooms)
print(f"Predicted price for a house with {size} sq ft and {bedrooms} bedrooms is ${predicted_price:.2f}")
