# This file is part of the OpenProtein project.
#
# @author Jeppe Hallgren
#
# For license information, please see the LICENSE file in the root directory.
from util import *
import time
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, use_gpu, embedding_size):
        super(BaseModel, self).__init__()

        # initialize model variables
        self.use_gpu = use_gpu
        self.embedding_size = embedding_size

    def get_embedding_size(self):
        return self.embedding_size

    def embed(self, original_aa_string):
        data, batch_sizes = torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.pack_sequence(original_aa_string))

        # one-hot encoding
        start_compute_embed = time.time()
        prot_aa_list = data.unsqueeze(1)
        embed_tensor = torch.zeros(prot_aa_list.size(0), 21, prot_aa_list.size(2)) # 21 classes
        if self.use_gpu:
            prot_aa_list = prot_aa_list.cuda()
            embed_tensor = embed_tensor.cuda()
        input_sequences = embed_tensor.scatter_(1, prot_aa_list.data, 1).transpose(1,2)
        end = time.time()
        write_out("Embed time:", end - start_compute_embed)
        packed_input_sequences = rnn_utils.pack_padded_sequence(input_sequences, batch_sizes)
        return packed_input_sequences

    def compute_loss(self, original_aa_string, actual_coords_list):

        emissions, backbone_atoms_padded, batch_sizes = self._get_network_emissions(original_aa_string)
        actual_coords_list_padded, batch_sizes_coords = torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.pack_sequence(actual_coords_list))
        if self.use_gpu:
            actual_coords_list_padded = actual_coords_list_padded.cuda()
        start = time.time()
        emissions_actual, batch_sizes_actual = \
            calculate_dihedral_angles_over_minibatch(actual_coords_list_padded, batch_sizes_coords, self.use_gpu)
        drmsd_avg = calc_avg_drmsd_over_minibatch(backbone_atoms_padded, actual_coords_list_padded, batch_sizes)
        write_out("Angle and drmsd calculation time:", time.time() - start)
        if self.use_gpu:
            emissions_actual = emissions_actual.cuda()
            drmsd_avg = drmsd_avg.cuda()
        angular_loss = calc_angular_difference(emissions, emissions_actual)

        return angular_loss # + drmsd_avg

    def forward(self, original_aa_string):
        return self._get_network_emissions(original_aa_string)