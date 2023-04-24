# Packages and Imports

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc


# Pyswarms - for algorithm and visualization
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from pyswarms.utils.plotters.formatters import Mesher


# Accuracy score and confusion matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix 

# Nueural network packages
import tensorflow
from tensorflow.keras.optimizers.legacy import Adam 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

# Load dataset
wine_data = np.loadtxt('/Users/freddiejones/Desktop/EvolutionaryML/winequality-red.csv', delimiter=',', skiprows=1) # print(wine_data.shape)
np.set_printoptions(formatter={'float': lambda x: '{0:0.2f}'.format(x)})

# Binary classification transformation
wine_data[wine_data[:, -1] < 5.5, -1] = 0 
wine_data[wine_data[:, -1] >= 5.5, -1] = 1
print(wine_data)

# Balance dataset
np.random.shuffle(wine_data)

# Split into test and train
thirty_percent = int(0.3 * len(wine_data[:, 0]))

X_test = wine_data[:thirty_percent, :-1] 
Y_test = wine_data[:thirty_percent, -1]

X_train = wine_data[thirty_percent:, 0:-1] 
Y_train = wine_data[thirty_percent:, -1]

# Normalize wine data
X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

# Forward propagation - taken directly from PySwarms documentation
def forward_prop(params):
    """Forward propagation as objective function

    This computes for the forward propagation of the neural network, as
    well as the loss. It receives a set of parameters that must be
    rolled-back into the corresponding weights and biases.

    Inputs
    ------
    params: np.ndarray
        The dimensions should include an unrolled version of the
        weights and biases.

    Returns
    -------
    float
        The computed negative log-likelihood loss given the parameters
    """
    # Neural network architecture
    # ------------- Modified Code------------- #
    n_inputs = 11
    n_hidden = 22
    n_classes = 2

    # Roll-back the weights and biases
    # Values calculated using dimension calculations
    W1 = params[0:242].reshape((n_inputs,n_hidden))
    b1 = params[242:264].reshape((n_hidden,))
    W2 = params[264:308].reshape((n_hidden,n_classes))
    b2 = params[308:310].reshape((n_classes,))
    
    # Perform forward propagation
    z1 = X_train.dot(W1) + b1    # Pre-activation in Layer 1
    a1 = 1 / (1 + np.exp(-z1))   # Sigmoid activation 
    z2 = a1.dot(W2) + b2         # Pre-activation in Layer 2
    logits = 1 / (1 + np.exp(-z2))  # Sigmoid activation

    # Compute for the sigmoid of the logits
    sigmoid_scores = 1 / (1 + np.exp(-logits))
    probs = sigmoid_scores / np.sum(sigmoid_scores, axis=1, keepdims=True)

    # Compute for the negative log likelihood
    N = 1120 # Number of samples for my problem
    corect_logprobs = -np.log(probs[range(N), Y_train.astype(int)]) # Change to type int since it needs an int or bool
    
    # ------------- End of Modified Code in function------------- #
    
    loss = np.sum(corect_logprobs) / N

    return loss

# Executes forward propagation function in the entire swarm - taken directly from PySwarms documentation
# Nothing edited, just runs my forward_prop function on the swarm size

def f(x):
    """Higher-level method to do forward_prop in the
    whole swarm.

    Inputs
    ------
    x: numpy.ndarray of shape (n_particles, dimensions)
        The swarm that will perform the search

    Returns
    -------
    numpy.ndarray of shape (n_particles, )
        The computed loss for each particle
    """
    n_particles = x.shape[0]
    j = [forward_prop(x[i]) for i in range(n_particles)]
    return np.array(j)

# Initialize swarm with hyper-parameters
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

# Call instance of PSO and calculate dimensions for network
dimensions = (11 * 22) + (22 * 2) + 22 + 2

# Swarm size 30
optimizer = ps.single.GlobalBestPSO(n_particles=30, dimensions=dimensions, options=options)

# Perform optimization
cost, pos = optimizer.optimize(f, iters=10, verbose=3)

def predict(X_train, pos):
    """
    Use the trained weights to perform class predictions.

    Inputs
    ------
    X: numpy.ndarray
        
    pos: numpy.ndarray
        Position matrix found by the swarm. Will be rolled
        into weights and biases.
    """
    # Neural network architecture
    n_inputs = 11
    n_hidden = 22
    n_classes = 2

    # Roll-back the weights and biases
    # 
    W1 = pos[0:242].reshape((n_inputs,n_hidden))
    b1 = pos[242:264].reshape((n_hidden,))
    W2 = pos[264:308].reshape((n_hidden,n_classes))
    b2 = pos[308:310].reshape((n_classes,))

    # Perform forward propagation
    z1 = X_train.dot(W1) + b1    # Pre-activation in Layer 1
    a1 = 1 / (1 + np.exp(-z1))   # Sigmoid activation Layer 1
    z2 = a1.dot(W2) + b2         # Pre-activation in Layer 2
    logits = 1 / (1 + np.exp(-z2))    # Logits sigmoid Layer 2

    y_pred = np.argmax(logits, axis=1)
    return y_pred

 # Mean prediction accuracy for test and train sets
print("Test set prediction accuracy: ", (predict(X_test, pos) == Y_test).mean())
print("Train set prediction accuracy: ", (predict(X_train, pos) == Y_train).mean())

test_predictions = (predict(X_test, pos) == Y_test)
train_predictions = (predict(X_train, pos) == Y_train)

confusion_matrix_test = confusion_matrix(test_predictions, Y_test)
confusion_matrix_train = confusion_matrix(train_predictions, Y_train)

# Visualize monfusion matrices using seaborn like other assignments
import seaborn as sns

# Test data
sns.heatmap(confusion_matrix_test, annot=True)
plt.title('Confusion Matrix Test Data')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()

# Train data
sns.heatmap(confusion_matrix_train, annot=True)
plt.title('Confusion Matrix Train Data')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()

# Cost history over generations plot
plot_cost_history(cost_history=optimizer.cost_history)
plt.show()



