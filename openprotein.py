"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""
import math
import time
import torch
import torch.nn as nn
from util import calculate_dihedral_angles_over_minibatch, calc_angular_difference, \
    write_out, calculate_dihedral_angles, \
    get_structure_from_angles, write_to_pdb, calc_rmsd, \
    calc_drmsd, get_backbone_positions_from_angles, calc_avg_drmsd_over_minibatch, \
    structures_to_backbone_atoms_padded\


class BaseModel(nn.Module):
    def __init__(self, use_gpu, embedding_size):
        super(BaseModel, self).__init__()

        # initialize model variables
        self.use_gpu = use_gpu
        self.embedding_size = embedding_size
        self.historical_rmsd_avg_values = list()
        self.historical_drmsd_avg_values = list()

    def get_embedding_size(self):
        return self.embedding_size

    def embed(self, original_aa_string):
        max_len = max([s.size(0) for s in original_aa_string])
        seqs = []
        for tensor in original_aa_string:
            padding_to_add = torch.zeros(max_len-tensor.size(0)).int()
            seqs.append(torch.cat((tensor, padding_to_add)))

        data = torch.stack(seqs).transpose(0, 1)

        # one-hot encoding
        start_compute_embed = time.time()
        arange_tensor = torch.arange(21).int().repeat(
            len(original_aa_string), 1
        ).unsqueeze(0).repeat(max_len, 1, 1)
        data_tensor = data.unsqueeze(2).repeat(1, 1, 21)
        embed_tensor = (arange_tensor == data_tensor).float()

        if self.use_gpu:
            embed_tensor = embed_tensor.cuda()

        end = time.time()
        write_out("Embed time:", end - start_compute_embed)

        return embed_tensor

    def compute_loss(self, minibatch):
        (original_aa_string, actual_coords_list, _) = minibatch

        emissions, _backbone_atoms_padded, _batch_sizes = \
            self._get_network_emissions(original_aa_string)
        actual_coords_list_padded = torch.nn.utils.rnn.pad_sequence(actual_coords_list)
        if self.use_gpu:
            actual_coords_list_padded = actual_coords_list_padded.cuda()
        start = time.time()
        if isinstance(_batch_sizes[0], int):
            _batch_sizes = torch.tensor(_batch_sizes)
        emissions_actual, _ = \
            calculate_dihedral_angles_over_minibatch(actual_coords_list_padded,
                                                    _batch_sizes,
                                                    self.use_gpu)
        drmsd_avg = calc_avg_drmsd_over_minibatch(_backbone_atoms_padded, actual_coords_list_padded, _batch_sizes)

        write_out("Angle calculation time:", time.time() - start)
        if self.use_gpu:
            emissions_actual = emissions_actual.cuda()
            drmsd_avg = drmsd_avg.cuda()
        angular_loss = calc_angular_difference(emissions, emissions_actual)

        return  drmsd_avg + angular_loss 

    def forward(self, original_aa_string):
        return self._get_network_emissions(original_aa_string)

    def evaluate_model(self, data_loader):
        loss = 0
        data_total = []
        dRMSD_list = []
        RMSD_list = []
        for _, data in enumerate(data_loader, 0):
            primary_sequence, tertiary_positions, _mask = data
            start = time.time()
            predicted_angles, backbone_atoms, batch_sizes = self(primary_sequence)
            write_out("Apply model to validation minibatch:", time.time() - start)

            if predicted_angles == []:
                # model didn't provide angles, so we'll compute them here
                output_angles, _ = calculate_dihedral_angles_over_minibatch(backbone_atoms,
                                                                            batch_sizes,
                                                                            self.use_gpu)
            else:
                output_angles = predicted_angles

            cpu_predicted_angles = output_angles.transpose(0, 1).cpu().detach()
            if backbone_atoms == []:
                # model didn't provide backbone atoms, we need to compute that
                output_positions, _ = \
                    get_backbone_positions_from_angles(predicted_angles,
                                                       batch_sizes,
                                                       self.use_gpu)
            else:
                output_positions = backbone_atoms

            cpu_predicted_backbone_atoms = output_positions.transpose(0, 1).cpu().detach()

            minibatch_data = list(zip(primary_sequence,
                                      tertiary_positions,
                                      cpu_predicted_angles,
                                      cpu_predicted_backbone_atoms))
            data_total.extend(minibatch_data)
            start = time.time()
            for primary_sequence, tertiary_positions, _predicted_pos, predicted_backbone_atoms\
                    in minibatch_data:
                actual_coords = tertiary_positions.transpose(0, 1).contiguous().view(-1, 3)

                predicted_coords = predicted_backbone_atoms[:len(primary_sequence)]\
                    .transpose(0, 1).contiguous().view(-1, 3).detach()
                rmsd = calc_rmsd(predicted_coords, actual_coords)
                drmsd = calc_drmsd(predicted_coords, actual_coords)
                RMSD_list.append(rmsd)
                dRMSD_list.append(drmsd)
                error = rmsd
                loss += error

                end = time.time()
            write_out("Calculate validation loss for minibatch took:", end - start)
        loss /= data_loader.dataset.__len__()
        self.historical_rmsd_avg_values.append(float(torch.Tensor(RMSD_list).mean()))
        self.historical_drmsd_avg_values.append(float(torch.Tensor(dRMSD_list).mean()))

        prim = data_total[0][0]
        pos = data_total[0][1]
        pos_pred = data_total[0][3]
        if self.use_gpu:
            pos = pos.cuda()
            pos_pred = pos_pred.cuda()
        angles = calculate_dihedral_angles(pos, self.use_gpu)
        angles_pred = calculate_dihedral_angles(pos_pred, self.use_gpu)
        write_to_pdb(get_structure_from_angles(prim, angles), "test")
        write_to_pdb(get_structure_from_angles(prim, angles_pred), "test_pred")

        data = {}
        data["pdb_data_pred"] = open("output/protein_test_pred.pdb", "r").read()
        data["pdb_data_true"] = open("output/protein_test.pdb", "r").read()
        data["phi_actual"] = list([math.degrees(float(v)) for v in angles[1:, 1]])
        data["psi_actual"] = list([math.degrees(float(v)) for v in angles[:-1, 2]])
        data["phi_predicted"] = list([math.degrees(float(v)) for v in angles_pred[1:, 1]])
        data["psi_predicted"] = list([math.degrees(float(v)) for v in angles_pred[:-1, 2]])
        data["rmsd_avg"] = self.historical_rmsd_avg_values
        data["drmsd_avg"] = self.historical_drmsd_avg_values

        prediction_data = None

        return (loss, data, prediction_data)
