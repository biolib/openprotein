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
        self.historical_rmsd_avg_values = list()
        self.historical_drmsd_avg_values = list()

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

    def compute_loss(self, minibatch):
        (original_aa_string, actual_coords_list, mask) = minibatch

        emissions, backbone_atoms_padded, batch_sizes = self._get_network_emissions(original_aa_string)
        actual_coords_list_padded, batch_sizes_coords = torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.pack_sequence(actual_coords_list))
        if self.use_gpu:
            actual_coords_list_padded = actual_coords_list_padded.cuda()
        start = time.time()
        emissions_actual, batch_sizes_actual = \
            calculate_dihedral_angles_over_minibatch(actual_coords_list_padded, batch_sizes_coords, self.use_gpu)
        #drmsd_avg = calc_avg_drmsd_over_minibatch(backbone_atoms_padded, actual_coords_list_padded, batch_sizes)
        write_out("Angle calculation time:", time.time() - start)
        if self.use_gpu:
            emissions_actual = emissions_actual.cuda()
            #drmsd_avg = drmsd_avg.cuda()
        angular_loss = calc_angular_difference(emissions, emissions_actual)

        return angular_loss # + drmsd_avg

    def forward(self, original_aa_string):
        return self._get_network_emissions(original_aa_string)

    def evaluate_model(self, data_loader):
        loss = 0
        data_total = []
        dRMSD_list = []
        RMSD_list = []
        for i, data in enumerate(data_loader, 0):
            primary_sequence, tertiary_positions, mask = data
            start = time.time()
            predicted_angles, backbone_atoms, batch_sizes = self(primary_sequence)
            write_out("Apply model to validation minibatch:", time.time() - start)
            cpu_predicted_angles = predicted_angles.transpose(0, 1).cpu().detach()
            cpu_predicted_backbone_atoms = backbone_atoms.transpose(0, 1).cpu().detach()
            minibatch_data = list(zip(primary_sequence,
                                      tertiary_positions,
                                      cpu_predicted_angles,
                                      cpu_predicted_backbone_atoms))
            data_total.extend(minibatch_data)
            start = time.time()
            for primary_sequence, tertiary_positions, predicted_pos, predicted_backbone_atoms in minibatch_data:
                actual_coords = tertiary_positions.transpose(0, 1).contiguous().view(-1, 3)
                predicted_coords = predicted_backbone_atoms[:len(primary_sequence)].transpose(0, 1).contiguous().view(
                    -1, 3).detach()
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
