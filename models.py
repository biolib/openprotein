# This file is part of the OpenProtein project.
#
# @author Jeppe Hallgren
#
# For license information, please see the LICENSE file in the root directory.

import torch.autograd as autograd
import torch.nn as nn
from util import *
import torch.nn.utils.rnn as rnn_utils
import time
import numpy as np
import openprotein

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
                               num_layers=self.num_lstm_layers, bidirectional=True, bias=True)
        self.hidden_to_labels = nn.Linear(self.hidden_size * 2, self.mixture_size, bias=True) # * 2 for bidirectional
        self.init_hidden(minibatch_size)
        self.softmax_to_angle = soft_to_angle(self.mixture_size)
        self.soft = nn.LogSoftmax(2)
        self.bn = nn.BatchNorm1d(self.mixture_size)
        if self.use_gpu:
            self.bi_lstm = self.bi_lstm.cuda()
            self.hidden_to_labels = self.hidden_to_labels.cuda()
            self.softmax_to_angle = self.softmax_to_angle.cuda()

    def init_hidden(self, minibatch_size):
        # number of layers (* 2 since bidirectional), minibatch_size, hidden size
        initial_hidden_state = torch.zeros(self.num_lstm_layers * 2, minibatch_size, self.hidden_size)
        initial_cell_state = torch.zeros(self.num_lstm_layers * 2, minibatch_size, self.hidden_size)
        if self.use_gpu:
            initial_hidden_state = initial_hidden_state.cuda()
            initial_cell_state = initial_cell_state.cuda()
        self.hidden_layer = (autograd.Variable(initial_hidden_state),
                             autograd.Variable(initial_cell_state))

    def _get_network_emissions(self, original_aa_string):
        packed_input_sequences = self.embed(original_aa_string)
        minibatch_size = int(packed_input_sequences[1][0])
        self.init_hidden(minibatch_size)
        (data, batch_sizes), self.hidden_layer = self.bi_lstm(packed_input_sequences, self.hidden_layer)
        emissions_padded = torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.PackedSequence(self.hidden_to_labels(data), batch_sizes))
        x = emissions_padded[0].transpose(0,1).transpose(1,2) # minibatch_size, self.mixture_size, -1
        x = self.bn(x)
        x = x.transpose(1,2) #(minibatch_size, -1, self.mixture_size)
        p = torch.exp(self.soft(x))
        output = self.softmax_to_angle(p).transpose(0,1) # max size, minibatch size, 3 (angels)
        structures = get_structures_from_prediction(original_aa_string, output, emissions_padded[1])
        return output, structures, emissions_padded[1]

    def compute_loss(self, original_aa_string, actual_coords_list):
        if self.use_gpu:
            original_aa_string = original_aa_string.cuda()
            actual_coords_list = actual_coords_list.cuda()
        emissions, structures, batch_sizes = self._get_network_emissions(original_aa_string)
        backbone_atoms_list = structures_to_backbone_atoms_list(structures)
        emissions_actual, batch_sizes_actual = \
            calculate_dihedral_angles_over_minibatch(original_aa_string, actual_coords_list)
        drmsd_avg = calc_avg_drmsd_over_minibatch(backbone_atoms_list, actual_coords_list)
        angular_loss = calc_angular_difference(emissions, emissions_actual)
        return angular_loss + drmsd_avg

class soft_to_angle(nn.Module):
    def __init__(self, mixture_size):
        super(soft_to_angle, self).__init__()
        # Omega intializer
        omega_components1 = np.random.uniform(0, 1, int(mixture_size * 0.1))  # Initialize omega 90/10 pos/neg
        omega_components2 = np.random.uniform(2, math.pi, int(mixture_size * 0.9))
        omega_components = np.concatenate((omega_components1, omega_components2))
        np.random.shuffle(omega_components)

        phi_components = np.genfromtxt("data/mixture_model_pfam_"+str(mixture_size)+".txt")[:, 1]
        psi_components = np.genfromtxt("data/mixture_model_pfam_"+str(mixture_size)+".txt")[:, 2]

        self.phi_components = nn.Parameter(torch.from_numpy(phi_components).contiguous().view(-1, 1).float())
        self.psi_components = nn.Parameter(torch.from_numpy(psi_components).contiguous().view(-1, 1).float())
        self.omega_components = nn.Parameter(torch.from_numpy(omega_components).view(-1, 1).float())

    def forward(self, x):
        phi_input_sin = torch.matmul(x, torch.sin(self.phi_components))
        phi_input_cos = torch.matmul(x, torch.cos(self.phi_components))
        psi_input_sin = torch.matmul(x, torch.sin(self.psi_components))
        psi_input_cos = torch.matmul(x, torch.cos(self.psi_components))
        omega_input_sin = torch.matmul(x, torch.sin(self.omega_components))
        omega_input_cos = torch.matmul(x, torch.cos(self.omega_components))

        eps = 10 ** (-4)
        phi = torch.atan2(phi_input_sin, phi_input_cos + eps)
        psi = torch.atan2(psi_input_sin, psi_input_cos + eps)
        omega = torch.atan2(omega_input_sin, omega_input_cos + eps)

        return torch.cat((phi, psi, omega), 2)

