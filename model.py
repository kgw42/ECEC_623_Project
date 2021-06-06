"""
Authors: Kathrina Waugh, Isaac Deerwester, Pankil Chauhan
Date: May 5, 2021

Bio-Inspired SNN with IF Neurons.
"""

import tensorflow as tf
from sklearn import datasets
from torch import optim
import matplotlib.pyplot as plt
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, IFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from bindsnet.learning import PostPre, WeightDependentPostPre
from sklearn import preprocessing
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from sklearn.model_selection import train_test_split

import numpy as np

import bindsnet
import tempfile
from typing import Dict, Optional, Type, Iterable

import torch

from bindsnet.network.monitors import AbstractMonitor
from bindsnet.network.nodes import Nodes
from bindsnet.network.topology import AbstractConnection


class SNNetwork(torch.nn.Module):
    # language=rst

    def __init__(self, dt: float = 1.0, batch_size: int = 1, learning: bool = True):
        # language=rst
        """
        Initializes network object.
        :param dt: Simulation timestep.
        :param batch_size: Mini-batch size.
        :param learning: Whether to allow connection updates. True by default.
        """
        super().__init__()

        self.dt = dt
        self.batch_size = batch_size

        self.layers = {}
        self.connections = {}
        self.monitors = {}

        self.train(learning)

        self.epsilon = 4
        self.frequency = 250

    def add_layer(self, layer: Nodes, name: str) -> None:
        # language=rst
        """
        Adds a layer of nodes to the network.
        :param layer: A subclass of the ``Nodes`` object.
        :param name: Logical name of layer.
        """
        self.layers[name] = layer
        self.add_module(name, layer)

        layer.train(self.learning)
        layer.compute_decays(self.dt)
        layer.set_batch_size(self.batch_size)

    def add_connection(
            self, connection: AbstractConnection, source: str, target: str
    ) -> None:
        # language=rst
        """
        Adds a connection between layers of nodes to the network.
        :param connection: An instance of class ``Connection``.
        :param source: Logical name of the connection's source layer.
        :param target: Logical name of the connection's target layer.
        """
        self.connections[(source, target)] = connection
        self.add_module(source + "_to_" + target, connection)

        connection.dt = self.dt
        connection.train(self.learning)

    def add_monitor(self, monitor: AbstractMonitor, name: str) -> None:
        # language=rst
        """
        Adds a monitor on a network object to the network.
        :param monitor: An instance of class ``Monitor``.
        :param name: Logical name of monitor object.
        """
        self.monitors[name] = monitor
        monitor.network = self
        monitor.dt = self.dt

    def save(self, file_name: str) -> None:
        # language=rst
        """
        Serializes the network object to disk.
        :param file_name: Path to store serialized network object on disk.
        """
        torch.save(self, open(file_name, "wb"))

    def clone(self) -> "Network":
        # language=rst
        """
        Returns a cloned network object.

        :return: A copy of this network.
        """
        virtual_file = tempfile.SpooledTemporaryFile()
        torch.save(self, virtual_file)
        virtual_file.seek(0)
        return torch.load(virtual_file)

    def _get_inputs(self, layers: Iterable = None) -> Dict[str, torch.Tensor]:
        # language=rst
        """
        Fetches outputs from network layers to use as input to downstream layers.
        :param layers: Layers to update inputs for. Defaults to all network layers.
        :return: Inputs to all layers for the current iteration.
        """
        inputs = {}

        if layers is None:
            layers = self.layers

        # Loop over network connections.
        for c in self.connections:
            if c[1] in layers:
                # Fetch source and target populations.
                source = self.connections[c].source
                target = self.connections[c].target

                if not c[1] in inputs:
                    inputs[c[1]] = torch.zeros(
                        self.batch_size, *target.shape, device=target.s.device
                    )

                # Add to input: source's spikes multiplied by connection weights.
                inputs[c[1]] += self.connections[c].compute(source.s)

        return inputs

    def run(
            self, inputs: Dict[str, torch.Tensor], time: int, one_step=False, **kwargs
    ) -> None:
        # language=rst
        """
        Simulate network for given inputs and time.
        :param inputs: Dictionary of ``Tensor``s of shape ``[time, *input_shape]`` or
                      ``[time, batch_size, *input_shape]``.
        :param time: Simulation time.
        :param one_step: Whether to run the network in "feed-forward" mode, where inputs
            propagate all the way through the network in a single simulation time step.
            Layers are updated in the order they are added to the network.
        Keyword arguments:
        :param Dict[str, torch.Tensor] clamp: Mapping of layer names to boolean masks if
            neurons should be clamped to spiking. The ``Tensor``s have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] unclamp: Mapping of layer names to boolean masks
            if neurons should be clamped to not spiking. The ``Tensor``s should have
            shape ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Dict[str, torch.Tensor] injects_v: Mapping of layer names to boolean
            masks if neurons should be added voltage. The ``Tensor``s should have shape
            ``[n_neurons]`` or ``[time, n_neurons]``.
        :param Union[float, torch.Tensor] reward: Scalar value used in reward-modulated
            learning.
        :param Dict[Tuple[str], torch.Tensor] masks: Mapping of connection names to
            boolean masks determining which weights to clamp to zero.
        """
        # Parse keyword arguments.
        clamps = kwargs.get("clamp", {})
        unclamps = kwargs.get("unclamp", {})
        masks = kwargs.get("masks", {})
        injects_v = kwargs.get("injects_v", {})

        # Dynamic setting of batch size.
        if inputs != {}:
            for key in inputs:
                # goal shape is [time, batch, n_0, ...]
                if len(inputs[key].size()) == 1:
                    # current shape is [n_0, ...]
                    # unsqueeze twice to make [1, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(0).unsqueeze(0)
                elif len(inputs[key].size()) == 2:
                    # current shape is [time, n_0, ...]
                    # unsqueeze dim 1 so that we have
                    # [time, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(1)

            for key in inputs:
                # batch dimension is 1, grab this and use for batch size
                if inputs[key].size(1) != self.batch_size:
                    self.batch_size = inputs[key].size(1)

                    for l in self.layers:
                        self.layers[l].set_batch_size(self.batch_size)

                    for m in self.monitors:
                        self.monitors[m].reset_state_variables()

                break

        # Effective number of timesteps.
        timesteps = int(time / self.dt)

        # Simulate network activity for `time` timesteps.
        output_spikes = [[0 for _ in range(self.layers["Output"].n)] for _ in range(timesteps)]
        hidden_spikes = [[0 for _ in range(self.layers["Hidden"].n)] for _ in range(timesteps)]
        input_spikes = [[0 for _ in range(self.layers["Input"].n)] for _ in range(timesteps)]

        for t in range(timesteps):

            # Get input to all layers (synchronous mode).
            current_inputs = {}
            if not one_step:
                current_inputs.update(self._get_inputs())

            for l in self.layers:
                # Update each layer of nodes.
                if l in inputs:
                    print()
                    # Updates the current inputs
                    if l in current_inputs:
                        current_inputs[l] += inputs[l][t]
                    else:
                        current_inputs[l] = inputs[l][t]


                if one_step:
                    # Get input to this layer (one-step mode).
                    current_inputs.update(self._get_inputs(layers=[l]))

                self.layers[l].forward(x=current_inputs[l])

                # Inject voltage to neurons.
                inject_v = injects_v.get(l, None)
                if inject_v is not None:
                    if inject_v.ndimension() == 1:
                        self.layers[l].v += inject_v
                    else:
                        self.layers[l].v += inject_v[t]

            # Output layer Spike recording per timestep
            output_spikes[t] = self.layers["Output"].s

            sum_spikes = [0 for _ in range(self.layers["Output"].n)]
            error = [0 for _ in range(self.layers["Output"].n)]
            if t >= 4:
                diff = 4
            else:
                diff = t

            # Calculate post synaptic spike frequency in STDP Window t-4ms
            for ep in range(t-diff, t+1):
                sum_spikes = [x + y for x,y in zip(sum_spikes, output_spikes[ep])]

            # Check for target neurons  -- and apply equation 16 -- Error rate updates
            if any(sum_spikes[0].numpy()):
                target_neurons = ((sum_spikes[0] >= 1).nonzero(as_tuple=True)[0])
                silent_neurons = ((sum_spikes[0] == 0).nonzero(as_tuple=True)[0])

                for targ_neur in target_neurons:
                    try:
                        search = tf.math.equal(output_spikes[t][targ_neur], torch.zeros(1, dtype=torch.bool)[0])
                    except:
                        search = tf.math.equal(output_spikes[t][0][targ_neur], torch.zeros(1, dtype=torch.bool)[0])

                    if search:
                        error[targ_neur] = 1

                for sil_neur in silent_neurons:
                    if output_spikes[t][0][sil_neur] == True:
                        error[sil_neur] = -1

            # Calculate Presynaptic spike activity at output layer in STDP window t-4ms

            hidden_spikes[t] = self.layers["Hidden"].s

            sum_spikes_hid = [0 for _ in range(self.layers["Hidden"].n)]
            error_hid = [0 for _ in range(self.layers["Hidden"].n)]
            if t >= 4:
                diff = 4
            else:
                diff = t

            for ep in range(t-diff, t+1):
                sum_spikes_hid = [x + y for x,y in zip(sum_spikes_hid, hidden_spikes[ep])]

            # Check for target hidden neurons  -- and apply equation 16
            if any(sum_spikes_hid[0].numpy()):
                hid_tar_neurons = ((sum_spikes_hid[0] >= 1).nonzero(as_tuple=True)[0])

                # Create boolean of neurons to recive derivative upate rule
                for h_targ_neur in hid_tar_neurons:
                    error_hid[h_targ_neur] = torch.matmul(self.connections[('Hidden', 'Output')].w[h_targ_neur], torch.FloatTensor(error).reshape(-1,1))

            input_spikes[t] = self.layers["Input"].s

            # Calculate spiking activity at input layer in STDP window t-4ms
            sum_spikes_in = [0 for _ in range(self.layers["Input"].n)]
            if t >= 4:
                diff = 4
            else:
                diff = t

            for ep in range(t - diff, t + 1):
                sum_spikes_in = [x + y for x, y in zip(sum_spikes_in, input_spikes[ep])]

            # Weight update for Hidden to Output layer synapses and input to hidden layer synapses
            self.connections[('Hidden', 'Output')].w += torch.matmul(sum_spikes_hid[0].type(torch.FloatTensor).reshape(-1,1), torch.FloatTensor(error).reshape(1,-1)) * mu
            self.connections[('Input', 'Hidden')].w += torch.matmul(
                sum_spikes_in[0].type(torch.FloatTensor).reshape(-1, 1), torch.FloatTensor(error_hid).reshape(1, -1)) * mu

            # Run synapse updates.
            for c in self.connections:
                self.connections[c].update(
                    mask=masks.get(c, None), learning=self.learning, **kwargs
                )

            # Record state variables of interest.
            for m in self.monitors:
                self.monitors[m].record()


        # Re-normalize connections.
        for c in self.connections:
            self.connections[c].normalize()

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Reset state variables of objects in network.
        """
        for layer in self.layers:
            self.layers[layer].reset_state_variables()

        for connection in self.connections:
            self.connections[connection].reset_state_variables()

        for monitor in self.monitors:
            self.monitors[monitor].reset_state_variables()

    def train(self, mode: bool = True) -> "torch.nn.Module":
        # language=rst
        """
        Sets the node in training mode.
        :param mode: Turn training on or off.
        :return: ``self`` as specified in ``torch.nn.Module``.
        """
        self.learning = mode
        return super().train(mode)

# Import the Dataset
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=42)

# Normalize the Dataset to range [0,1]
norm_train_X = preprocessing.normalize(X_train)
norm_test_X = preprocessing.normalize(X_test)

# Network Parameters
epsilon = 4
beta = 250
hidden_thresh = 0.9
delta_t = 1.0
mu = 0.0005
H = 30
output_thresh = 0.025
U = 0

# Create the Network with custom BPSTDP rule
network = SNNetwork(dt=delta_t)

# Create and add input, hidden and output layers.
input_layer = Input(n=4, traces=True)
hidden_layer = IFNodes(n=30, traces=True, thresh=-65 + hidden_thresh)
output_layer = IFNodes(n=3, traces=True, thresh=-65 + 0.025*H)

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
    nu=4e-3,
    wmin=-1,
    wmax=1,
    update_rule=WeightDependentPostPre
)
# w=torch.randn(hidden_layer.n, output_layer.n),
network.add_connection(
    connection=hidden_output, source="Hidden", target="Output"
)

# Initialize optimizer
optimizer = optim.SGD(network.parameters(), lr=5e-5, momentum=0.9)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in network.state_dict():
    print(param_tensor, "\t", network.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

time = 150
dt = 1
lif_neurons = 3    # number of output nuerons
n_classes = 3
per_class  = int(lif_neurons / n_classes)
supervised=True
lif_layer_name = "Output"

# record the spike times of each neuron during the simulation.
spike_record = torch.zeros(1, int(time / dt), lif_neurons)

# record the mapping of each neuron to its corresponding label
assignments = -torch.ones_like(torch.Tensor(lif_neurons))

# how frequently each neuron fires for each input class
rates = torch.zeros_like(torch.Tensor(lif_neurons, n_classes))

# the likelihood of each neuron firing for each input class
proportions = torch.zeros_like(torch.Tensor(lif_neurons, n_classes))


# label(s) of the input(s) being processed
labels = torch.empty(1,dtype=torch.int)

# create a spike monitor for each layer in the network
# this allows us to read the spikes in order to assign labels to neurons and determine the predicted class
layer_monitors = {}

# Create and add input and output layer monitors.
in_monitor = Monitor(
    obj=input_layer,
    state_vars=("s",),  # Record spikes.
    time=time,  # Length of simulation
)

hid_monitor = Monitor(
    obj=hidden_layer,
    state_vars=("s", "v"),  # Record spikes and voltages.
    time=time,  # Length of simulation
)

out_monitor = Monitor(
    obj=output_layer,
    state_vars=("s", "v"),  # Record spikes and voltages.
    time=time,  # Length of simulation
)

network.add_monitor(monitor=in_monitor, name="Input")
network.add_monitor(monitor=hid_monitor, name="Hidden")
network.add_monitor(monitor=out_monitor, name="Output")

# list of encoded images for random selection during training

encoded_train_inputs = []
encoder = bindsnet.encoding.encoders.BernoulliEncoder(time=time, dt=dt)
input_layer_name = "Input"

# loop through encode each image type and store into a list of encoded images
for i, sample in enumerate(norm_train_X):

    # encode the image
    encoded_img = encoder(torch.flatten(torch.from_numpy(sample)))

    # encoded image input for the network
    encoded_img_input = {input_layer_name: encoded_img}

    # encoded image label
    encoded_img_label = y_train[i]

    # add to the encoded input list along with the input layer name
    encoded_train_inputs.append({"Label" : encoded_img_label, "Inputs" : encoded_img_input})

encoded_test_inputs = []

# loop through encode each image type and store into a list of encoded images
for i, sample in enumerate(norm_test_X):

    # encode the image
    encoded_img = encoder(torch.flatten(torch.from_numpy(sample)))

    # encoded image input for the network
    encoded_img_input = {input_layer_name: encoded_img}

    # encoded image label
    encoded_img_label = y_test[i]

    # add to the encoded input list along with the input layer name
    encoded_test_inputs.append({"Label" : encoded_img_label, "Inputs" : encoded_img_input})

# iterate for epochs
epochs = 10

weight_history = None
num_correct = 0.0

supervised = True
log_messages = True
plot_weights = True
input_neurons = 4


for step in range(epochs):
    for sample in encoded_test_inputs:

        # get the label for the current image
        labels[0] = sample['Label']

        # randomly decide which output neuron should spike if more than one neuron corresponds to the class
        choice = np.random.choice(per_class, size=1, replace=False)

        # clamp on the output layer forces the node corresponding to the label's class to spike
        clamp = {"Output": per_class * labels[0] + torch.Tensor(choice).long()} if supervised else {}

        # Step 1: Run the network with the provided inputs
        network.run(inputs=sample['Inputs'], time=time, clamp=clamp)

        # Step 2: Get the spikes produced at the output layer
        spike_record[0] = network.monitors[lif_layer_name].get("s").view(time, lif_neurons)

        # Step 3:

        # Assign labels to the neurons based on highest average spiking activity.
        # Returns a Tuple of class assignments, per-class spike proportions, and per-class firing rates
        assignments, proportions, rates = assign_labels(spike_record, labels, n_classes, rates)

        # Step 4: Classify data based on the neuron (label) with the highest average spiking activity

        # Classify data with the label with highest average spiking activity over all neurons.
        all_activity_pred = all_activity(spike_record, assignments, n_classes)

        # Step 5: Classify data based on the neuron (label) with the highest average spiking activity weighted by class-wise proportion
        proportion_pred = proportion_weighting(spike_record, assignments, proportions, n_classes)

        # Update Accuracy
        num_correct += 1 if (labels.numpy()[0] == all_activity_pred.numpy()[0]) else 0

        # Display Information
        if log_messages:
            print("Actual Label:", labels.numpy(), "|", "Predicted Label:", all_activity_pred.numpy(), "|",
                  "Proportionally Predicted Label:", proportion_pred.numpy())

            print("Neuron Label Assignments:")
            for idx in range(assignments.numel()):
                print(
                    "\t Output Neuron[", idx, "]:", assignments[idx],
                    "Proportions:", proportions[idx],
                    "Rates:", rates[idx]
                )
            print("\n")

    # For Weight Plotting
    if plot_weights:
        weights = network.connections[("Hidden", "Output")].w[:, 0].numpy().reshape((1, 30))
        print("Neuron 0 Weights:\n", network.connections[("Hidden", "Output")].w[:, 0])
        print("Neuron 1 Weights:\n", network.connections[("Hidden", "Output")].w[:, 1])
        print("====================")

    if log_messages:
        print("Epoch #", step, "\tAccuracy:", num_correct / ((step + 1) * len(encoded_train_inputs)))
        print("===========================\n\n")

# Print Final Class Assignments and Proportions
print("Neuron Label Assignments:")
for idx in range(assignments.numel()):
    print(
        "\t Output Neuron[", idx, "]:", assignments[idx],
        "Proportions:", proportions[idx],
        "Rates:", rates[idx]
    )

# Save the trained model and weights in h5 format
torch.save(network, "model.h5")

# Test the model
num_correct = 0
log_messages = True

# Load the model
network = torch.load("model.h5")
network.eval()

labels = torch.empty(1,dtype=torch.int)

# loop through each test example and record performance
for sample in encoded_test_inputs:

    # get the label for the current image
    labels[0] = sample['Label']

    # Step 1: Run the network with the provided inputs
    network.run(inputs=sample["Inputs"], time=time)

    # Step 2: Get the spikes produced at the output layer
    spike_record[0] = network.monitors[lif_layer_name].get("s").view(time, lif_neurons)

    # Step 3:

    # Assign labels to the neurons based on highest average spiking activity.
    # Returns a Tuple of class assignments, per-class spike proportions, and per-class firing rates
    assignments, proportions, rates = assign_labels(spike_record, labels, n_classes, rates)

    # Step 4: Classify data based on the neuron (label) with the highest average spiking activity

    # Classify data with the label with highest average spiking activity over all neurons.
    all_activity_pred = all_activity(spike_record, assignments, n_classes)

    # Step 5: Classify data based on the neuron (label) with the highest average spiking activity weighted by class-wise proportion
    proportion_pred = proportion_weighting(spike_record, assignments, proportions, n_classes)

    # Update Accuracy
    num_correct += 1 if (labels.numpy()[0] == all_activity_pred.numpy()[0]) else 0

    # Display Information
    if log_messages:
        print("Actual Label:", labels.numpy(), "|", "Predicted Label:", all_activity_pred.numpy(), "|",
              "Proportionally Predicted Label:", proportion_pred.numpy())

        print("Neuron Label Assignments:")
        for idx in range(assignments.numel()):
            print(
                "\t Output Neuron[", idx, "]:", assignments[idx],
                "Proportions:", proportions[idx],
                "Rates:", rates[idx]
            )
        print("\n")

print("Accuracy:", num_correct / len(encoded_test_inputs))
