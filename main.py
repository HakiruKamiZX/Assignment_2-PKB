import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def classify_iris(data, new_data, clf):
    # Predict the class for the new data point using the trained classifier (clf)
    predicted_class = clf.predict(new_data[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']])
    return predicted_class[0]

def check_accuracy(data, clf):
    # Calculate accuracy using the trained classifier (clf)
    X = data[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']]
    y = data['class']
    accuracy = clf.score(X, y)
    return accuracy

# Load the Iris dataset from a CSV file
data = pd.read_csv('iris_dataset.csv')

# Handle missing values by imputing with column means
imputer = SimpleImputer(strategy='mean')
data[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']] = imputer.fit_transform(data[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']])

# Encode the 'class' column into numeric labels
label_encoder = LabelEncoder()
data['class'] = label_encoder.fit_transform(data['class'])

# Train a Gaussian Naive Bayes classifier on the data
X = data[['sepallength', 'sepalwidth', 'petallength', 'petalwidth']].values
y = data['class']

clf = GaussianNB()
clf.fit(X, y)

# User input for new data
try:
    new_sepallength = float(input("Enter the sepal length of the new data: "))
    new_sepalwidth = float(input("Enter the sepal width of the new data: "))
    new_petallength = float(input("Enter the petal length of the new data: "))
    new_petalwidth = float(input("Enter the petal width of the new data: "))
except ValueError:
    print("Invalid input. Please enter numeric values for attributes.")
    exit()

new_data = pd.DataFrame({'sepallength': [new_sepallength],
                         'sepalwidth': [new_sepalwidth],
                         'petallength': [new_petallength],
                         'petalwidth': [new_petalwidth]})

# Classify the new data using the trained classifier
result = classify_iris(data, new_data, clf)

# Inverse transform the predicted numeric class label to the original string label
predicted_class_label = label_encoder.inverse_transform([result])[0]
print(f"Predicted class for the new data: {predicted_class_label}")

# Append the new data to the existing data
data = data.append(new_data, ignore_index=True)

# Encode the 'class' column into numeric labels for accuracy calculation
data['class'] = label_encoder.transform(data['class'])

# Save the updated data to 'iris_dataset.csv'
data.to_csv('iris_dataset.csv', index=False)

# Calculate accuracy using the trained classifier
accuracy = check_accuracy(data, clf)
print(f"Accuracy: {accuracy * 100:.2f}%")
