from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.datasets import load_wine

X, y = load_wine(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Initialize the Linear Regression model
linear_model = LinearRegression()

# Train the Linear Regression model
linear_model.fit(X_train, y_train)

# Make predictions with Linear Regression
# here linear model is used with threshold  you can use it with a threshold to classify,
lin_predictions = (linear_model.predict(X_test) > 0.5).astype(int)

#  Initialize the Logistic Regression model
log_model = LogisticRegression(max_iter=1000)

#  Train the Logistic Regression model
log_model.fit(X_train, y_train)

#  Make predictions with Logistic Regression
y_predict = log_model.predict(X_test)

# Calculate accuracy for both models
lin_accuracy = accuracy_score(y_test, lin_predictions)
#  calculate the logistic regression accuracy
log_accuracy = accuracy_score(y_test, y_predict)

print(f'Linear Regression Accuracy: {lin_accuracy:.2f}')
#  print the logistic regression accuracy
print(f'Logistic Regression Accuracy: {log_accuracy:.2f}')