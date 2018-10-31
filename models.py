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

# seed random generator for reproducibility
torch.manual_seed(1)

class ExampleModel(nn.Module):
    def __init__(self, num_labels, embedding, minibatch_size, use_gpu):
        super(ExampleModel, self).__init__()

        # initialize model variables
        self.use_gpu = use_gpu
        self.embedding = embedding
        self.hidden_size = 10
        self.embedding_function = nn.Embedding(24, self.get_embedding_size())
        self.bi_lstm = nn.LSTM(self.get_embedding_size(), self.hidden_size, num_layers=1, bidirectional=True)
        self.hidden_to_labels = nn.Linear(self.hidden_size * 2, num_labels) # * 2 for bidirectional
        self.init_hidden(minibatch_size)
        if self.use_gpu:
            self.embedding_function = self.embedding_function.cuda()
            self.bi_lstm = self.bi_lstm.cuda()
            self.hidden_to_labels = self.hidden_to_labels.cuda()

    def get_embedding_size(self):
        return 21

    def flatten_parameters(self):
        self.bi_lstm.flatten_parameters()

    def embed(self, prot_aa_list):
        # one-hot encoding
        prot_aa_list = prot_aa_list.long().unsqueeze(1)
        embed_tensor = torch.zeros(prot_aa_list.size(0), 21, prot_aa_list.size(2)) # 21 classes
        if self.use_gpu:
            embed_tensor = embed_tensor.cuda()
        return embed_tensor.scatter_(1, prot_aa_list.data, 1).transpose(1,2)

    def init_hidden(self, minibatch_size):
        # number of layers (* 2 since bidirectional), minibatch_size, hidden size
        initial_hidden_state = torch.zeros(1 * 2, minibatch_size, self.hidden_size)
        initial_cell_state = torch.zeros(1 * 2, minibatch_size, self.hidden_size)
        if self.use_gpu:
            initial_hidden_state = initial_hidden_state.cuda()
            initial_cell_state = initial_cell_state.cuda()
        self.hidden_layer = (autograd.Variable(initial_hidden_state),
                             autograd.Variable(initial_cell_state))

    def _get_network_emissions(self, input_sequences):
        if self.embedding == "PYTORCH":
            pad_seq, seq_length = rnn_utils.pad_sequence(input_sequences), [v.size(0) for v in input_sequences]
            pad_seq_embed = self.embedding_function(pad_seq)
            packed = rnn_utils.pack_padded_sequence(pad_seq_embed, seq_length)
        else:
            packed = rnn_utils.pack_sequence(input_sequences)
        minibatch_size = len(input_sequences)
        self.init_hidden(minibatch_size)
        bi_lstm_out, self.hidden_layer = self.bi_lstm(packed, self.hidden_layer)
        data, batch_sizes = bi_lstm_out
        emissions = self.hidden_to_labels(data)
        emissions_padded = torch.nn.utils.rnn.pad_packed_sequence(torch.nn.utils.rnn.PackedSequence(emissions,batch_sizes))
        return emissions_padded

    def neg_log_likelihood(self, original_aa_string, actual_labels):
        if self.use_gpu:
            original_aa_string = original_aa_string.cuda()
            actual_labels = actual_labels.cuda()
        start_compute_embed = time.time()
        input_sequences = self.embed(original_aa_string)
        end = time.time()
        write_out("Embed time:", end - start_compute_embed)
        emissions, batch_sizes = self._get_network_emissions(input_sequences)
        emissions = emissions.transpose(0,1)
        emissions = emissions.transpose(1,2).double()
        loss = -1 * torch.norm(emissions - actual_labels)
        return loss

    def forward(self, original_aa_string):
        if self.use_gpu:
            original_aa_string = original_aa_string.cuda()
        input_sequences = self.embed(original_aa_string)
        emissions, batch_sizes = self._get_network_emissions(input_sequences)
        return emissions.transpose(0,1).transpose(1,2).double()

