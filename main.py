"""
Authors: Kathrina Waugh, Isaac Deerwester, Pankil Chauhan
Date: May 5, 2021

SNN with IF Neurons for Iris Dataset.
"""

import torch
import matplotlib.pyplot as plt
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, IFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from bindsnet.learning import PostPre
from BPSTDP import BPSTDP
from bindsnet.pipeline import BasePipeline

from sklearn import datasets
import numpy as np

# import the dataset
iris = datasets.load_iris()
X = iris.data[:, :4]  # Take the first 4 features (Length and width of both petal and sepal)
y = iris.target

# Parameters

epsilon = 4 #ms
beta = 250
hidden_thresh = 0.9   # pre
delta_t = 1 # 1ms
mu = 0.0005
H = torch.randn(10,1500)
output_thresh = 0.25 * H
U = 0

# Create the Network
network = Network()

# Create and add input, hidden and output layers.
input_layer = Input(n=4, traces=True)
hidden_layer = IFNodes(n=30, traces=True, thresh=hidden_thresh)
output_layer = IFNodes(n=3, traces=True, thresh=output_thresh)

network.add_layer(
    layer=input_layer, name="Input"
)
network.add_layer(
    layer=hidden_layer, name="Hidden"
)
network.add_layer(
    layer=output_layer, name="Output"
)

# Create connection between input and hidden layers.
"""
Input to output layer uses 
"""
# TODO: CREATE A LEARNING RULE IN BINDSNET
input_hidden = Connection(
    source=input_layer,
    target=hidden_layer,
    w=0 * torch.randn(input_layer.n, hidden_layer.n),  # Initial weights - what did the paper use for initial weights e.g. w = 0.1 * size of the connection, i.e. torch.ones(4,30)
) # uses wmax instead of w

# nu parameter is the learning rate
# norm - constant weight normalization

network.add_connection(
    connection=input_hidden, source="Input", target="Hidden"
)

# Create connection between hidden and output layers.
hidden_output = Connection(
    source=hidden_layer,
    target=output_layer,
    w=0, # TODO: USE UPDATED WEIGHTS FROM UPDATE RULE - DECLARE A GLOBAL SELF.WEIGHTS
    update_rule=BPSTDP) #uses wmax instead of w

network.add_connection(
    connection=hidden_output, source="Hidden", target="Output"
)

# w = torch.zeros(3,3) # Recurrent weights proportional to sq(dist) between neurons
# n_sqrt = int(np.sqrt(3))
# for i in range(3):
#     x1, y1 = i // n_sqrt, i % n_sqrt
#     for j in range(3):
#         x2, y2 = j //n_sqrt, j % n_sqrt
#         dx, dy = x2 - x1, y2 - y1
#         w[i, j] = dx *dx + dy * dy
#     w[i, i] = 0.0
# w = torch.pow(w/w.max(), 0.25)
# w = start_inhib + w* max_inhib
#
# recurrent_output_conn = network.Connection()

pipeline = BasePipeline(network, output="Output", time=1, delta=delta_t)

while True:
    pipeline.step()
    if pipeline.done == True:
        pipeline._reset()
