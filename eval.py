import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV file to examine its structure and contents
file_path = '../../PR/eval3.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
data.head()

# Sort the dataframe by epoch to ensure the line plot makes sense
data_sorted = data.sort_values(by='epoch')
data_sorted['loss'] = data_sorted['loss'] * 100

# Plotting
plt.figure(figsize=(10, 5))

# Accuracy plot
plt.plot(data_sorted['epoch'], data_sorted['accuracy'], label='Accuracy', marker='o', linestyle='-', color='blue')

# Loss plot
plt.plot(data_sorted['epoch'], data_sorted['loss'], label='Loss', marker='x', linestyle='--', color='red')

# Adding titles and labels
plt.title('Model Training Performance')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Show the plot
plt.tight_layout()
plt.show()
