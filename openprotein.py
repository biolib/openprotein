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

    def forward(self, original_aa_string):
        return self._get_network_emissions(original_aa_string)