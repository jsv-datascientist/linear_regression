import numpy as np
from sklearn.linear_model import Ridge
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#  Load the diabetes dataset and extract features X and target y
df = load_diabetes()
X = df.data
y = df.target
#  Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#  Train a Ridge Regression model with alpha set to 0.5
ridge_model = Ridge(alpha = 0.05)
ridge_model.fit(X_train,y_train)

#  Print the coefficients of the model using `ridge_model.coef_`
print(f" Coff are {ridge_model.coef_}")

#  Print the intercept of the model using `ridge_model.intercept_`
print(f"Intercept is {ridge_model.intercept_}")

#  Predict on the test set using the trained model and calculate the MSE
y_predict = ridge_model.predict(X_test)

#  Print the Mean Squared Error (MSE)
result = mean_squared_error(y_test, y_predict)
print(result)