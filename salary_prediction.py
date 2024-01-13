import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load your dataset
# Replace 'your_dataset.csv' with the actual file path
# Assuming the file is in the same directory as your script
data = pd.read_csv('dataset.csv')

# Extract features (YearsExperience) and target variable (Salary)
X = data[['YearsExperience']]
y = data['Salary']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize the results
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('Salary Prediction')
plt.show()

# Take input from the user for years of experience
user_experience = float(input("Enter years of experience: "))
new_experience = [[user_experience]]

# Use the trained model to predict salary for the user's input
predicted_salary = model.predict(new_experience)
print(f'Predicted Salary: {predicted_salary[0]}')

# Plot the new data point and its predicted salary on the graph
plt.scatter(user_experience, predicted_salary, color='red', marker='x', s=100, label='Predicted Salary')
plt.legend()

# Show the plot
plt.show()

