"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""

import math
import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence

import openprotein
from util import get_backbone_positions_from_angles, compute_atan2

# seed random generator for reproducibility
torch.manual_seed(1)

# sample model borrowed from
# https://github.com/lblaabjerg/Master/blob/master/Models%20and%20processed%20data/ProteinNet_LSTM_500.py
class ExampleModel(openprotein.BaseModel):
    def __init__(self, embedding_size, minibatch_size, use_gpu):
        super(ExampleModel, self).__init__(use_gpu, embedding_size)

        self.hidden_size = 25
        self.num_lstm_layers = 2
        self.mixture_size = 500
        self.bi_lstm = nn.LSTM(self.get_embedding_size(), self.hidden_size,
                               num_layers=self.num_lstm_layers,
                               bidirectional=True, bias=True)
        self.hidden_to_labels = nn.Linear(self.hidden_size * 2,
                                          self.mixture_size, bias=True)  # * 2 for bidirectional
        self.init_hidden(minibatch_size)
        self.softmax_to_angle = SoftToAngle(self.mixture_size)
        self.batch_norm = nn.BatchNorm1d(self.mixture_size)

    def init_hidden(self, minibatch_size):
        # number of layers (* 2 since bidirectional), minibatch_size, hidden size
        initial_hidden_state = torch.zeros(self.num_lstm_layers * 2,
                                           minibatch_size, self.hidden_size)
        initial_cell_state = torch.zeros(self.num_lstm_layers * 2,
                                         minibatch_size, self.hidden_size)
        if self.use_gpu:
            initial_hidden_state = initial_hidden_state.cuda()
            initial_cell_state = initial_cell_state.cuda()
        self.hidden_layer = (autograd.Variable(initial_hidden_state),
                             autograd.Variable(initial_cell_state))

    def _get_network_emissions(self, original_aa_string):
        padded_input_sequences = self.embed(original_aa_string)
        minibatch_size = len(original_aa_string)
        batch_sizes = list([v.size(0) for v in original_aa_string])
        packed_sequences = pack_padded_sequence(padded_input_sequences, batch_sizes)

        self.init_hidden(minibatch_size)
        (data, bi_lstm_batches, _, _), self.hidden_layer = self.bi_lstm(
            packed_sequences, self.hidden_layer)
        emissions_padded, batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.PackedSequence(self.hidden_to_labels(data), bi_lstm_batches))
        emissions = emissions_padded.transpose(0, 1)\
            .transpose(1, 2)  # minibatch_size, self.mixture_size, -1
        emissions = self.batch_norm(emissions)
        emissions = emissions.transpose(1, 2)  # (minibatch_size, -1, self.mixture_size)
        probabilities = torch.softmax(emissions, 2)
        output_angles = self.softmax_to_angle(probabilities)\
            .transpose(0, 1)  # max size, minibatch size, 3 (angles)
        backbone_atoms_padded, _ = \
            get_backbone_positions_from_angles(output_angles,
                                               batch_sizes,
                                               self.use_gpu)
        return output_angles, backbone_atoms_padded, batch_sizes


class SoftToAngle(nn.Module):
    def __init__(self, mixture_size):
        super(SoftToAngle, self).__init__()
        # Omega Initializer
        omega_components1 = np.random.uniform(0, 1, int(mixture_size*0.1)) # set omega 90/10 pos/neg
        omega_components2 = np.random.uniform(2, math.pi, int(mixture_size*0.9))
        omega_components = np.concatenate((omega_components1, omega_components2))
        np.random.shuffle(omega_components)

        phi_components = np.genfromtxt("data/mixture_model_pfam_"
                                       + str(mixture_size) + ".txt")[:, 1]
        psi_components = np.genfromtxt("data/mixture_model_pfam_"
                                       + str(mixture_size) + ".txt")[:, 2]

        self.phi_components = nn.Parameter(torch.from_numpy(phi_components)
                                           .contiguous().view(-1, 1).float())
        self.psi_components = nn.Parameter(torch.from_numpy(psi_components)
                                           .contiguous().view(-1, 1).float())
        self.omega_components = nn.Parameter(torch.from_numpy(omega_components)
                                             .view(-1, 1).float())

    def forward(self, x):
        phi_input_sin = torch.matmul(x, torch.sin(self.phi_components))
        phi_input_cos = torch.matmul(x, torch.cos(self.phi_components))
        psi_input_sin = torch.matmul(x, torch.sin(self.psi_components))
        psi_input_cos = torch.matmul(x, torch.cos(self.psi_components))
        omega_input_sin = torch.matmul(x, torch.sin(self.omega_components))
        omega_input_cos = torch.matmul(x, torch.cos(self.omega_components))

        phi = compute_atan2(phi_input_sin, phi_input_cos)
        psi = compute_atan2(psi_input_sin, psi_input_cos)
        omega = compute_atan2(omega_input_sin, omega_input_cos)

        return torch.cat((phi, psi, omega), 2)
