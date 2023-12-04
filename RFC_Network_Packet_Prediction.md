# Mount Google Drive to access the dataset
from google.colab import drive
drive.mount('/content/drive')

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the dataset from Google Drive
dataset = pd.read_csv('/content/drive/MyDrive/cicdssaa.csv')
dataset.head()

# Extract features (X) and target (Y) from the dataset
X = dataset.iloc[:, 0:-1].values
Y = dataset.iloc[:, 4:].values

# Plot the training data
plt.plot(X)
plt.title("Training Data")
plt.xlabel("Number of systems in DB")
plt.ylabel("Peaks of records")
plt.grid()
plt.show()

# Plot the testing data
plt.plot(Y)
plt.title("Testing Data")
plt.xlabel("Number of systems in DB")
plt.ylabel("Peaks of records")
plt.grid()
plt.show()

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Extract relevant features for further analysis
flow_Rate = Y[:, 3]
fwPac = [X[:, 4], X[:, 4]]
bwPac = X[:, 78]
diff = bwPac - fwPac
bwPac
pred_an = max(diff[1:10])
pred_an

# Train a Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rtree = RandomForestClassifier(n_estimators=100, random_state=3)
rtree.fit(fwPac, fwPac)

# Make predictions using the trained model
Y_rt_predict = rtree.predict(fwPac)

# Plot the predicted values
pred_ops = Y_rt_predict[1:100]
plt.plot(pred_ops)
plt.scatter(pred_ops, pred_ops, label='Random Data', color='cyan', marker='o', s=50)
plt.title('Malicious Fw packet')
plt.xlabel('Number of iterations')
plt.ylabel('Impacted Fw packet')
