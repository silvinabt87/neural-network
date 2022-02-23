import numpy as np

def sigmoid(x):
  return 1/(1 + np.exp(-x))

# sigmoid derivatives to adjust synaptic weights
def sigmoid_derivative(x):
    return x * (1 - x)
    
training_inputs= np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

training_outputs= np.array([[0,0,1]]).T

#to make every iteration get the same random numbrs
np.random.seed(1)

synaptic_weights=2 * np.random.random((3,1))-1

print('synaptic weights starting the training')
print(synaptic_weights)

for iteration in range(100000):

  input_layer= training_inputs

#use sigmoid to multiply wights by the inputs

  outputs= sigmoid(np.dot(input_layer, synaptic_weights))

# calculate differece between error and real outputs and train
error= training_outputs - outputs

# multiply how much we missed by the
# slope of the sigmoid at the values in outputs
adjustments = error * sigmoid_derivative(outputs)

# update weights
synaptic_weights += np.dot(input_layer.T, adjustments)

print('Synaptic weights after training: ')
print(synaptic_weights)

print('outputs after training')
print(outputs)