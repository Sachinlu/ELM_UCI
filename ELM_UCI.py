# Importing the required file for incremental ELM
import numpy as np


# To find out root mean square error of the dataset
def r_mean_squared_error(y_actual, y_pred):
    return np.sqrt((np.mean((y_actual - y_pred) ** 2)))


# The main incremental ELM class
class Incremental_ELM:
    def __init__(self, input_nodes, hidden_layer, output_nodes, ):
        self.input_nodes = input_nodes
        self.hidden_layer = 1
        self.output_nodes = output_nodes
        self.bias = np.zeros(shape=(self.hidden_layer,))
        self.weights = np.random.uniform(-1, 1, (self.input_nodes, self.hidden_layer))
        self.beta = np.random.uniform(-1, 1, (hidden_layer, output_nodes))

    # The sigmoid activation function
    def sigmoid_activation(self, X_input):
        return 1. / (1. + np.exp(-X_input))

    # The prediction function
    def prediction(self, value):
        y_value = self.sigmoid_activation(value.dot(self.weights) + self.bias)
        return list(y_value.dot(self.beta))

    # The training function
    def fit(self, X_input, max_hd, Y_output):
        self.weights = np.random.uniform(-1, 1, (self.input_nodes, 1))
        self.beta = np.random.uniform(-1, 1, (1, self.output_nodes))

        hidden_layer_M = self.sigmoid_activation(X_input.dot(self.weights))
        hidden_layer_M_inv = np.linalg.pinv(hidden_layer_M)
        self.beta = hidden_layer_M_inv.dot(Y_output)

        for i in range(1, max_hd):
            h_w = np.random.uniform(-1, 1, (self.input_nodes, 1))
            h_b = np.random.uniform(-1, 1, (1, self.output_nodes))
            self.weights = np.hstack([self.weights, h_w])
            self.beta = np.vstack([self.beta, h_b])

            hidden_layer_M = self.sigmoid_activation(X_input.dot(self.weights))
            hidden_layer_M_inv = np.linalg.pinv(hidden_layer_M)
            self.beta = hidden_layer_M_inv.dot(Y_output)
            self.beta = self.beta.reshape(-1, 1)

        print('Bias shape:', self.bias.shape)
        print('Weights shape:', self.weights.shape)
        print('Beta shape:', self.beta.shape)


# Loading the dataset and importing libraries
from sklearn.model_selection import train_test_split
import pandas as pd
import time

# Number of output classes
max_hidden_node = 1000
df = pd.read_excel('Folds5x2_pp.xlsx')
X = df[['AT', 'V', 'AP', 'RH']].to_numpy()
y = df[['PE']].to_numpy()

# splitting data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# creating a ELM class object model
model = Incremental_ELM(input_nodes=4, hidden_layer=max_hidden_node, output_nodes=1)

# Training and calulating the rmse.
initial_train_record = time.time()
model.fit(X_train, max_hidden_node, y_train)
final_train_record = time.time()
train_pred = model.prediction(X_train)
error_train = r_mean_squared_error(y_train, train_pred)

print(f"<-----------Training Error and Time----------->\n")
print(f"Root mean squared Error of UCI regression Train Dataset : {error_train}")
print(f"model is approximating the training data by {100 - error_train} %")
print(f"Train Time of UCI regression dataset : {final_train_record - initial_train_record} seconds")

# Predicting and Calculating the rmse.
initial_test_record = time.time()
test_pred = model.prediction(X_test)
final_test_record = time.time()
error_test = r_mean_squared_error(y_test, test_pred)

print(f"<-----------Testing Error and Time----------->\n")
print(f"Root mean squared Error of UCI regression Dataset : {error_test}")
print(f"model is approximating the testing data by {100 - error_train} %")
print(f"Test Time of UCI regression Dataset : {final_test_record - initial_test_record} seconds")
