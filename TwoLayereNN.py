import numpy as np
import random
Inputs = [[1,2,3,2.5],
          [2,5,-1,2],
          [-1.5,2.7,3.3,-0.8]]

weights = [
        [0.2,0.8,-0.5,1],
        [0.5,-0.91,0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]
    ]
weights2 = [
        [0.1,-0.14,0.5],
        [-0.5,0.12,-0.33],
        [-0.44,0.73,-0.13]
    ]

biases = [2.0, 3.0, 0.5]
biases2 = [-1,2,-0.5]

Layer1_outputs = np.dot(Inputs, np.array(weights).T) + biases
Layer2_outputs = np.dot(Layer1_outputs, np.array(weights2).T) + biases2

print("Layer 2 consisting of 3 neurons:\n",Layer2_outputs)

from nnfs.datasets import spiral_data
import numpy as np
import nnfs
import matplotlib.pyplot as plt

nnfs.init()

X,y = spiral_data(samples=100, classes=3)
plt.scatter(X[: ,0], X[:,1], c=y, cmap='brg')
plt.show()

