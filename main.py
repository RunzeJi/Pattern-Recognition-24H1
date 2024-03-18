from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import datetime
import pandas as pd
import sys


# Load the CSV file to check its structure
train_path = "../train/1.csv"
data = pd.read_csv(train_path)



# Display the first few rows of the dataframe
print(data.head())

# Convert 'time' column to datetime
data['time'] = pd.to_datetime(data['time'])

# Extracting features from 'time'
data['hour'] = data['time'].dt.hour
data['day_of_week'] = data['time'].dt.dayofweek

# Encoding the 'type' column
label_encoder = LabelEncoder()
data['type_encoded'] = label_encoder.fit_transform(data['type'])

print(data['type_encoded'])

# Splitting the dataset into features (X) and target variable (y)
X = data[['lat', 'lon', '速度', '方向', 'hour', 'day_of_week']]
y = data['type_encoded']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Output the shapes of the training and testing sets to confirm splitting
X_train.shape, X_test.shape, y_train.shape, y_test.shape


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
