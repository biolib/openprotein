"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""
import time

import torch
import torch.nn as nn
import openprotein

from util import initial_pos_from_aa_string, \
    pass_messages, write_out, calc_avg_drmsd_over_minibatch

class RrnModel(openprotein.BaseModel):
    def __init__(self, embedding_size, use_gpu):
        super(RrnModel, self).__init__(use_gpu, embedding_size)
        self.recurrent_steps = 2
        self.hidden_size = 50
        self.msg_output_size = 50
        self.output_size = 9  # 3 dimensions * 3 coordinates for each aa
        self.f_to_hid = nn.Linear((embedding_size * 2 + 9), self.hidden_size, bias=True)
        self.hid_to_pos = nn.Linear(self.hidden_size, self.msg_output_size, bias=True)
        # (last state + orginal state)
        self.linear_transform = nn.Linear(embedding_size + 9 + self.msg_output_size, 9, bias=True)
        self.use_gpu = use_gpu

    def apply_message_function(self, aa_features):
        # aa_features: msg_count * 2 * feature_count
        aa_features_transformed = torch.cat(
            (
                aa_features[:, 0, 0:21],
                aa_features[:, 1, 0:21],
                aa_features[:, 0, 21:30] - aa_features[:, 1, 21:30]
            ), dim=1)
        return self.hid_to_pos(self.f_to_hid(aa_features_transformed))  # msg_count * outputsize

    def _get_network_emissions(self, original_aa_string):
        backbone_atoms_padded, batch_sizes_backbone = \
            initial_pos_from_aa_string(original_aa_string, self.use_gpu)
        embedding_padded = self.embed(original_aa_string)

        if self.use_gpu:
            backbone_atoms_padded = backbone_atoms_padded.cuda()

        for _ in range(self.recurrent_steps):
            combined_features = torch.cat(
                (embedding_padded, backbone_atoms_padded),
                dim=2
            ).transpose(0, 1)

            features_transformed = []

            for aa_features in combined_features.split(1, dim=0):
                msg = pass_messages(aa_features.squeeze(0),
                                    self.apply_message_function,
                                    self.use_gpu)  # aa_count * output size
                features_transformed.append(self.linear_transform(
                    torch.cat((aa_features.squeeze(0), msg), dim=1)))

            backbone_atoms_padded_clone = torch.stack(features_transformed).transpose(0, 1)

        backbone_atoms_padded = backbone_atoms_padded_clone

        return [], backbone_atoms_padded, batch_sizes_backbone

    def compute_loss(self, minibatch):
        (original_aa_string, actual_coords_list, _) = minibatch

        _, backbone_atoms_padded, batch_sizes = \
            self._get_network_emissions(original_aa_string)
        actual_coords_list_padded = torch.nn.utils.rnn.pad_sequence(actual_coords_list)
        if self.use_gpu:
            actual_coords_list_padded = actual_coords_list_padded.cuda()
        start = time.time()
        if isinstance(batch_sizes[0], int):
            batch_sizes = torch.tensor(batch_sizes)

        drmsd_avg = calc_avg_drmsd_over_minibatch(backbone_atoms_padded,
                                                  actual_coords_list_padded,
                                                  batch_sizes)
        write_out("drmsd calculation time:", time.time() - start)
        if self.use_gpu:
            drmsd_avg = drmsd_avg.cuda()

        return drmsd_avg
