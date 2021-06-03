from abc import ABC
from typing import Union, Optional, Sequence
"""
Authors: Kathrina Waugh, Isaac Deerwester, Pankil Chauhan
Date: May 5, 2021

SNN with IF Neurons for Iris Dataset.
"""

import tensorflow as tf
import torch
import matplotlib.pyplot as plt
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, IFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from bindsnet.learning import PostPre, WeightDependentPostPre
from sklearn import preprocessing
from bindsnet.pipeline import BasePipeline

from sklearn import datasets
import numpy as np


import tempfile
from typing import Dict, Optional, Type, Iterable

import torch

from bindsnet.network.monitors import AbstractMonitor
from bindsnet.network.nodes import Nodes
from bindsnet.network.topology import AbstractConnection
# from bindsnet.network.topology.learning.reward import AbstractReward

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
        **Example:**
        .. code-block:: python
            import torch
            import matplotlib.pyplot as plt
            from pathlib          import Path
            from bindsnet.network import *
            from bindsnet.network import topology
            # Build simple network.
            network = Network(dt=1.0)
            X = nodes.Input(100)  # Input layer.
            Y = nodes.LIFNodes(100)  # Layer of LIF neurons.
            C = topology.Connection(source=X, target=Y, w=torch.rand(X.n, Y.n))  # Connection from X to Y.
            # Add everything to the network object.
            network.add_layer(layer=X, name='X')
            network.add_layer(layer=Y, name='Y')
            network.add_connection(connection=C, source='X', target='Y')
            # Save the network to disk.
            network.save(str(Path.home()) + '/network.pt')
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
        **Example:**
        .. code-block:: python
            import torch
            import matplotlib.pyplot as plt
            from bindsnet.network import Network
            from bindsnet.network.nodes import Input
            from bindsnet.network.monitors import Monitor
            # Build simple network.
            network = Network()
            network.add_layer(Input(500), name='I')
            network.add_monitor(Monitor(network.layers['I'], state_vars=['s']), 'I')
            # Generate spikes by running Bernoulli trials on Uniform(0, 0.5) samples.
            spikes = torch.bernoulli(0.5 * torch.rand(500, 500))
            # Run network simulation.
            network.run(inputs={'I' : spikes}, time=500)
            # Look at input spiking activity.
            spikes = network.monitors['I'].get('s')
            plt.matshow(spikes, cmap='binary')
            plt.xticks(()); plt.yticks(());
            plt.xlabel('Time'); plt.ylabel('Neuron index')
            plt.title('Input spiking')
            plt.show()
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
        spike_train = {'Input':[0 for x in range(4)],'Hidden':[0 for y in range(30)], 'Output':[0 for z in range(3)] }
        target_neuron = {'Input':[0 for x in range(4)],'Hidden':[0 for x in range(30)], 'Output':[0 for x in range(3)]}
        error_rate = {'Input': [0 for x in range(4)], 'Hidden': [0 for y in range(30)],
                       'Output': [0 for z in range(3)]}

        for t in range(timesteps):
            print("Step Time: {}".format(t))
            # Get input to all layers (synchronous mode).
            current_inputs = {}
            if not one_step:
                current_inputs.update(self._get_inputs())

            for l in self.layers:
                # Update each layer of nodes.
                print("Layer {}".format(l))
                if l in inputs:
                    print()
                    # Updates the current inputs
                    if l in current_inputs:
                        current_inputs[l] += inputs[l][t]
                    else:
                        current_inputs[l] = inputs[l][t]


                if one_step:
                    print("Here")
                    # Get input to this layer (one-step mode).
                    current_inputs.update(self._get_inputs(layers=[l]))
                print("CURRENT INPUT")
                print(current_inputs)
                self.layers[l].forward(x=current_inputs[l])

                # Inject voltage to neurons.
                inject_v = injects_v.get(l, None)
                if inject_v is not None:
                    if inject_v.ndimension() == 1:
                        self.layers[l].v += inject_v
                    else:
                        self.layers[l].v += inject_v[t]
                spike_train[l] = [x+y for x,y in zip(spike_train[l], self.layers[l].s)]

            #output layer weight adaptation
            output_spikes[t] = self.layers["Output"].s

            sum_spikes = [0 for _ in range(self.layers["Output"].n)]
            error = [0 for _ in range(self.layers["Output"].n)]
            if t >= 4:
                diff = 4
            else:
                diff = t

            #print(output_spikes)
            #print("________________")
            for ep in range(t-diff, t+1):
                #print(output_spikes[ep])
                sum_spikes = [x + y for x,y in zip(sum_spikes, output_spikes[ep])]

            print("OUTPUT SUM AT T", sum_spikes[0])

            # Check for target neurons  -- and apply equation 16
            if any(sum_spikes[0].numpy()):
                print("Target Neuron found")

                target_neurons = ((sum_spikes[0] >= 1).nonzero(as_tuple=True)[0])
                silent_neurons = ((sum_spikes[0] == 0).nonzero(as_tuple=True)[0])

                for targ_neur in target_neurons:
                    if output_spikes[t][targ_neur] == False or output_spikes[t][targ_neur] == 0:
                        error[targ_neur] = 1

                for sil_neur in silent_neurons:
                    if output_spikes[t][sil_neur] == True or output_spikes[t][sil_neur] == 1:
                        error[sil_neur] = -1

            # Weight adaptation for hidden layer

            hidden_spikes[t] = self.layers["Hidden"].s

            sum_spikes_hid = [0 for _ in range(self.layers["Hidden"].n)]
            error_hid = [0 for _ in range(self.layers["Hidden"].n)]
            if t >= 4:
                diff = 4
            else:
                diff = t

            for ep in range(t-diff, t+1):
                #print(output_spikes[ep])
                sum_spikes_hid = [x + y for x,y in zip(sum_spikes_hid, hidden_spikes[ep])]

            print("HIDDEN SUM AT T", sum_spikes_hid[0])

            # Check for target hidden neurons  -- and apply equation 16
            if any(sum_spikes_hid[0].numpy()):
                hid_tar_neurons = ((sum_spikes_hid[0] >= 1).nonzero(as_tuple=True)[0])

                # Already hadles the derivative
                for h_targ_neur in hid_tar_neurons:
                    print(h_targ_neur)
                    print(self.connections[('Hidden', 'Output')].w[h_targ_neur])
                    print(torch.FloatTensor(error))
                    error_hid[h_targ_neur] = torch.matmul(self.connections[('Hidden', 'Output')].w[h_targ_neur], torch.FloatTensor(error).reshape(-1,1))

            input_spikes[t] = self.layers["Input"].s

            sum_spikes_in = [0 for _ in range(self.layers["Input"].n)]
            error_in = [0 for _ in range(self.layers["Input"].n)]
            if t >= 4:
                diff = 4
            else:
                diff = t

            for ep in range(t - diff, t + 1):
                sum_spikes_in = [x + y for x, y in zip(sum_spikes_in, input_spikes[ep])]

            print("INPUT SUM AT T", sum_spikes_in[0])

            print('____________')
            print(error_hid)

            print("Spike Train")
            print(spike_train)

            # weight update - output and input synapses
            mu = 0.0005

            #self.connections[('Hidden', 'Output')].w += torch.matmul(torch.FloatTensor(sum_spikes_hid), torch.FloatTensor(error)) * mu
            self.connections[('Hidden', 'Output')].w += torch.matmul(sum_spikes_hid[0].type(torch.FloatTensor).reshape(-1,1), torch.FloatTensor(error).reshape(1,-1)) * mu
            self.connections[('Input', 'Hidden')].w += torch.matmul(
                sum_spikes_in[0].type(torch.FloatTensor).reshape(-1, 1), torch.FloatTensor(error_hid).reshape(1, -1)) * mu


            print("Spike Train")
            print(spike_train)


            # Run synapse updates.
            for c in self.connections:
                print(c)
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
