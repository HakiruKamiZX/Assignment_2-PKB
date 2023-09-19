import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

def calculate_thresholds(data):
    # Calculate thresholds based on the first half of the data
    half_point = len(data) // 2
    first_half_data = data.iloc[:half_point]

    big_leaf_data = first_half_data[first_half_data['Species'] == 'big-leaf']
    small_leaf_data = first_half_data[first_half_data['Species'] == 'small-leaf']

    if big_leaf_data.empty or small_leaf_data.empty:
        # If there are no data points of one or both classes in the first half,
        # calculate thresholds for the entire dataset
        big_leaf_data = data[data['Species'] == 'big-leaf']
        small_leaf_data = data[data['Species'] == 'small-leaf']

    if big_leaf_data.empty or small_leaf_data.empty:

       
        big_width_mean = big_leaf_data['Width'].mean()
        big_length_mean = big_leaf_data['Length'].mean()
        big_width_std = big_leaf_data['Width'].std()
        big_length_std = big_leaf_data['Length'].std()

    small_width_mean = small_leaf_data['Width'].mean()
    small_length_mean = small_leaf_data['Length'].mean()
    small_width_std = small_leaf_data['Width'].std()
    small_length_std = small_leaf_data['Length'].std()

    width_lower_threshold = min(big_width_mean - 2 * big_width_std, small_width_mean - 2 * small_width_std)
    width_upper_threshold = max(big_width_mean + 2 * big_width_std, small_width_mean + 2 * small_width_std)

    length_lower_threshold = min(big_length_mean - 2 * big_length_std, small_length_mean - 2 * small_length_std)
    length_upper_threshold = max(big_length_mean + 2 * big_length_std, small_length_mean + 2 * small_length_std)

    return width_lower_threshold, width_upper_threshold, length_lower_threshold, length_upper_threshold


def classify_leaf(data, new_width, new_length, thresholds):
    if thresholds is None:
        return "Unable to calculate thresholds due to missing or invalid data"

    width_lower_threshold, width_upper_threshold, length_lower_threshold, length_upper_threshold = thresholds

    # If both width and length are above upper thresholds, classify as big leaf
    if new_width > width_upper_threshold and new_length > length_upper_threshold:
        return "big-leaf"

    # Train a Gaussian Naive Bayes classifier on the data
    X = data[['Width', 'Length']].values
    y = data['Species']

    clf = GaussianNB()
    clf.fit(X, y)

    # Predict the probabilities for each class for the new data point
    class_probabilities = clf.predict_proba([[new_width, new_length]])[0]

    # Use the class probabilities and thresholds to make a probabilistic prediction
    if class_probabilities[0] > class_probabilities[1]:
        if new_width < width_lower_threshold or new_length < length_lower_threshold:
            return "small-leaf"
        else:
            return "big-leaf"
    else:
        if new_width < width_lower_threshold or new_length < length_lower_threshold:
            return "big-leaf"
        else:
            return "small-leaf"

def check_accuracy(data, thresholds):
    if thresholds is None:
        return None, None

    correct_predictions_original = 0
    correct_predictions_new = 0

    # Calculate accuracy for the original dataset (excluding the last row)
    for index, row in data.iterrows():
        if index == len(data) - 1:
            # Skip the last (new) row
            continue

        actual = row['Species']
        predicted = classify_leaf(data.iloc[:-1], row['Width'], row['Length'], thresholds)

        if actual == predicted:
            correct_predictions_original += 1

    accuracy_original = correct_predictions_original / (len(data) - 1)  # Exclude the last row from total

    # Calculate accuracy for the new row (the last row)
    actual_new = data.iloc[-1]['Species']
    predicted_new = classify_leaf(data.iloc[:-1], new_width, new_length, thresholds)

    if actual_new == predicted_new:
        correct_predictions_new = 1

    accuracy_new = correct_predictions_new

    return accuracy_original, accuracy_new

# Load the source leaf data from a CSV file
data = pd.read_csv('daun.csv')

width_lower_threshold, width_upper_threshold, length_lower_threshold, length_upper_threshold = calculate_thresholds(data)

print(f"Width Lower Threshold: {width_lower_threshold}")
print(f"Width Upper Threshold: {width_upper_threshold}")
print(f"Length Lower Threshold: {length_lower_threshold}")
print(f"Length Upper Threshold: {length_upper_threshold}")

# Calculate thresholds based on the first half of the data
thresholds = calculate_thresholds(data)

try:
    new_width = float(input("Enter the width of the new leaf: "))
    new_length = float(input("Enter the length of the new leaf: "))
except ValueError:
    print("Invalid input. Please enter numeric values for width and length.")
    exit()

result = classify_leaf(data, new_width, new_length, thresholds)
print(result)

# Create a new DataFrame for the result of the prediction and the new data
new_data = pd.DataFrame({'Width': [new_width], 'Length': [new_length], 'Species': [result]})

#Append the new data to the existing data
data = data.append(new_data, ignore_index=True)

# Save the updated data to 'daun.csv'
data.to_csv('daun.csv', index=False)

accuracy_original, accuracy_new = check_accuracy(data, thresholds)
print(f"Accuracy: {accuracy_original * 100:.2f}%")
