# Import the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Load the Titanic dataset
df = pd.read_csv("titanic.csv")
# Drop the unnecessary columns
df = df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"], axis=1)
# Convert the categorical variables to numerical variables
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Pclass"] = df["Pclass"].map({1: 0, 2: 1, 3: 2})
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop("Survived", axis=1), df["Survived"], test_size=0.2)
# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)