import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt

from torchvision import transforms
from tqdm import tqdm
from time import time as t

from BPSTDP import SNNetwork


from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.analysis.plotting import (
    plot_input,
    plot_assignments,
    plot_performance,
    plot_weights,
    plot_spikes,
    plot_voltages,
)
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, IFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages

from sklearn import datasets

from sklearn import preprocessing

epsilon = 4 #ms
beta = 250
hidden_thresh = 0.9   # pre
delta_t = 1.0 # 1ms
mu = 0.0005
H = 30
output_thresh = 0.025
U = 0

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_train", type=int, default=120)
parser.add_argument("--n_test", type=int, default=30)
parser.add_argument("--n_clamp", type=int, default=1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=150)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=32)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--device_id", type=int, default=0)
parser.set_defaults(plot=True, gpu=True, train=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_train = args.n_train
n_test = args.n_test
n_clamp = args.n_clamp
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu
device_id = args.device_id

np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

if not train:
    update_interval = n_test

# Build BP-STDP network.
network = SNNetwork(dt=1)

# Create and add input, hidden and output layers.
input_layer = Input(n=4, traces=True)
hidden_layer = IFNodes(n=30, traces=True, traces_additive=True, thresh=hidden_thresh, sum_input=True)
output_layer = IFNodes(n=3, traces=True, traces_additive=True, thresh=0.1*H, sum_input=True)

network.add_layer(
    layer=input_layer, name="Input"
)
network.add_layer(
    layer=hidden_layer, name="Hidden"
)
network.add_layer(
    layer=output_layer, name="Output"
)

input_hidden = Connection(
    source=input_layer,
    target=hidden_layer,
    w=torch.randn(input_layer.n, hidden_layer.n)
)

network.add_connection(
    connection=input_hidden, source="Input", target="Hidden"
)

# Create connection between hidden and output layers.
hidden_output = Connection(
    source=hidden_layer,
    target=output_layer,
    w=torch.randn(hidden_layer.n, output_layer.n))

network.add_connection(
    connection=hidden_output, source="Hidden", target="Output"
)

# Directs network to GPU
if gpu:
    network.to("cuda")

# Load Iris data.
iris = datasets.load_iris()
X = iris.data # Take the first 4 features (Length and width of both petal and sepal)
y = iris.target

norm_X = preprocessing.normalize(X)
# Create a dataloader to iterate and batch data
#dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
#spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)
# Train the network.
print("Begin training.\n")
start = t()

inpt_axes = None
inpt_ims = None
spike_ims = None
spike_axes = None
weights1_im = None
voltage_ims = None
voltage_axes = None

n_epochs = 10
n_classes = 10

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

for epoch in range(n_epochs):
    input_data = torch.bernoulli(torch.from_numpy(norm_X)).byte()
    inputs = {"Input": input_data}

    network.run(inputs, time=time)


    network.reset_state_variables()  # Reset state variables.

print("Progress: %d / %d (%.4f seconds)\n" % (n_epochs, n_epochs, t() - start))
print("Training complete.\n")