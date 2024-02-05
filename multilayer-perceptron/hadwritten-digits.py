# Read in the mnist digit dataset

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
import random
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

def inspect_mnist_data(X, y):
    # Load the MNIST dataset
    # Print the shape of the data
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)

    # Print unique labels in y
    unique_labels = set(y)
    print("Unique labels in y:", unique_labels)

    # Print summary statistics of X
    print("Summary statistics of X:")
    print("Mean:", X.mean())
    print("Standard Deviation:", X.std())

# Call the function to inspect the data
#inspect_mnist_data(X, y)

# divide the data into a training set 
# and test set, randomly selecting 5000 examples for training.
train_samples = 5000

# X is in pandas format for some reason. Convert to numpy.
X = np.array(X)
y = np.array(y)

random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0])
X = X[permutation]
y = y[permutation]
X = X.reshape((X.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=1000)

# print out the 417 th item in the dataset and its label.
i = 417
img = np.array(X_train[i]).reshape(28,28)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()
y_train[i]

# Let's see how a decision tree with 170 decision 
# rules performs by training it and printing its accuracy.
clf = DecisionTreeClassifier(max_leaf_nodes = 170)	
clf = clf.fit(X_train, y_train)			
correct = 0						
for i in range(len(X_test)):	
  if clf.predict([X_test[i]]) == y_test[i]: correct = correct + 1
  acc = [100.0* correct / len(X_test)]
print("decision tree")
print("--------------")
print("{:.3f}".format(acc[0]/100))

# Now let's try a simple neural network, a multi-layer 
# perceptron with no hidden layers.
clf = MLPClassifier(hidden_layer_sizes=[], max_iter = 10000, activation = 'identity')
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("multi-layer perceptron")
print("--------------")
print(score)

# Now, we will add one hidden layer and expand the 
# number of hidden units from 10 to 200 in intervals of 10. 
# We'll print the accuracy of each model given the number 
# of hidden units.
print("multi-layer perceptron hidden layers")
print("--------------")
for i in range(1,21):
  nhidden = i*10
  clf = MLPClassifier(hidden_layer_sizes=[nhidden], max_iter = 10000)
  clf.fit(X_train, y_train)
  score = clf.score(X_test, y_test)
  print(nhidden, score)