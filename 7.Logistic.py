from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load dataset
X, y = load_iris(return_X_y=True)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Ma'ke predictions and print output
print("Predictions=",model.predict(X))
