import numpy as np
from sklearn.linear_model import Lasso
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load and split the diabetes dataset
data = load_diabetes()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train a Lasso Regression model and make predictions
lasso = Lasso(alpha=0.5)
lasso.fit(X_train, y_train)

y_predict = lasso.predict(X_test)

#Calculate and print Mean Squared Error (MSE) for the Lasso Regression model
result = mean_squared_error(y_test, y_predict)
print(result)