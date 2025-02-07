from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the dataset
X, y = load_diabetes(return_X_y=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=0.1),
    "Lasso Regression": Lasso(alpha=0.1),
    "Elastic Regression" : ElasticNet(alpha=0.1, l1_ratio=0.5)
    #  Initialize Elastic Net Regression model with alpha=0.1 and l1_ratio=0.5
}

# Fit models, predict and calculate MSE
for name, model in models.items():
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    result = mean_squared_error(y_test, y_predict)
    #  fit each model on the training set, make test predictions, calculate and print MSE
    print(f"Model  {name} has MSE as {result}")