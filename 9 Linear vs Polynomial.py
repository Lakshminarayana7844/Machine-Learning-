import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Sample data (for illustration)
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])

# Linear Regression
model_linear = LinearRegression().fit(x, y)

# Polynomial Regression (degree=2)
model_poly = make_pipeline(PolynomialFeatures(degree=2), LinearRegression()).fit(x, y)

# Plotting
plt.scatter(x, y, color='blue', label='Data')
plt.plot(x, model_linear.predict(x), color='red', label='Linear')
plt.plot(x, model_poly.predict(x), color='green', label='Polynomial')
plt.legend()
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear vs Polynomial Regression')
plt.show()
