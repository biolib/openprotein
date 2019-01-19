# This file is part of the OpenProtein project.
#
# @author Jeppe Hallgren
#
# For license information, please see the LICENSE file in the root directory.

import torch
import torch.utils.data
import h5py
from datetime import datetime
import PeptideBuilder
import Bio.PDB
from Bio.PDB.vectors import Vector
import math
import numpy as np

AA_ID_DICT = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9,
              'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17,
              'V': 18, 'W': 19,'Y': 20}

def contruct_dataloader_from_disk(filename, minibatch_size):
    return torch.utils.data.DataLoader(H5PytorchDataset(filename), batch_size=minibatch_size,
                                       shuffle=True, collate_fn=H5PytorchDataset.merge_samples_to_minibatch)


class H5PytorchDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        super(H5PytorchDataset, self).__init__()

        self.h5pyfile = h5py.File(filename, 'r')
        self.num_proteins, self.max_sequence_len = self.h5pyfile['primary'].shape

    def __getitem__(self, index):
        mask = torch.Tensor(self.h5pyfile['mask'][index,:]).type(dtype=torch.uint8)
        prim = torch.masked_select(torch.Tensor(self.h5pyfile['primary'][index,:]).type(dtype=torch.long), mask)
        pos = torch.masked_select(torch.Tensor(self.h5pyfile['tertiary'][index]), mask).view(9, -1).transpose(0, 1)
        (phi_list, psi_list, omega_list) = calculate_dihedral_angels(pos)
        aa_list = protein_id_to_str(prim)
        structure = get_structure_from_angles(aa_list, phi_list[1:], psi_list[:-1], omega_list[:-1])
        tertiary = structure_to_backbone_atoms(structure)
        return  prim, \
                tertiary, \
               mask

    def __len__(self):
        return self.num_proteins

    def merge_samples_to_minibatch(samples):
        samples_list = []
        for s in samples:
            samples_list.append(s)
        # sort according to length of aa sequence
        samples_list.sort(key=lambda x: len(x[0]), reverse=True)
        return zip(*samples_list)

def set_experiment_id(data_set_identifier, learning_rate, minibatch_size):
    output_string = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    output_string += "-" + data_set_identifier
    output_string += "-LR" + str(learning_rate).replace(".","_")
    output_string += "-MB" + str(minibatch_size)
    globals().__setitem__("experiment_id",output_string)

def write_out(*args, end='\n'):
    output_string = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + str.join(" ", [str(a) for a in args]) + end
    if globals().get("experiment_id") is not None:
        with open("output/"+globals().get("experiment_id")+".txt", "a+") as output_file:
            output_file.write(output_string)
            output_file.flush()
    print(output_string, end="")

def evaluate_model(data_loader, model):
    loss = 0
    data_total = []
    dRMSD_list = []
    RMSD_list = []
    for i, data in enumerate(data_loader, 0):
        primary_sequence, tertiary_positions, mask = data

        predicted_positions, structures, batch_sizes = model(primary_sequence)
        backbone_atoms_list = structures_to_backbone_atoms_list(structures)
        predicted_pos_list =  list([a[:batch_sizes[idx],:] for idx,a in enumerate(predicted_positions.transpose(0,1))])
        minibatch_data = list(zip(primary_sequence,
                                  tertiary_positions,
                                  predicted_pos_list,
                                  structures,
                                  backbone_atoms_list))
        data_total.extend(minibatch_data)
        for primary_sequence, tertiary_positions,predicted_pos, structure, predicted_backbone_atoms in minibatch_data:
            actual_coords = tertiary_positions.transpose(0,1).contiguous().view(-1,3)
            rmsd = calc_rmsd(predicted_backbone_atoms.transpose(0,1).contiguous().view(-1,3), actual_coords)
            drmsd = calc_drmsd(predicted_backbone_atoms.transpose(0,1).contiguous().view(-1,3), actual_coords)
            RMSD_list.append(rmsd)
            dRMSD_list.append(drmsd)
            error = 1
            loss += error
    loss /= data_loader.dataset.__len__()
    return (loss, data_total, float(torch.Tensor(RMSD_list).mean()), float(torch.Tensor(dRMSD_list).mean()))

def write_model_to_disk(model):
    path = "output/models/"+globals().get("experiment_id")+".model"
    torch.save(model,path)
    return path

def draw_plot(fig, plt, validation_dataset_size, sample_num, train_loss_values,
              validation_loss_values):
    def draw_with_vars():
        ax = fig.gca()
        ax2 = ax.twinx()
        plt.grid(True)
        plt.title("Training progress (" + str(validation_dataset_size) + " samples in validation set)")
        train_loss_plot, = ax.plot(sample_num, train_loss_values)
        ax.set_ylabel('Train Negative log likelihood')
        ax.yaxis.labelpad = 0
        validation_loss_plot, = ax2.plot(sample_num, validation_loss_values, color='black')
        ax2.set_ylabel('Validation loss')
        ax2.set_ylim(bottom=0)
        plt.legend([train_loss_plot, validation_loss_plot],
                   ['Train loss on last batch', 'Validation loss'])
        ax.set_xlabel('Minibatches processed (=network updates)', color='black')
    return draw_with_vars

def draw_ramachandran_plot(fig, plt, phi, psi):
    def draw_with_vars():
        ax = fig.gca()
        plt.grid(True)
        plt.title("Ramachandran plot")
        train_loss_plot, = ax.plot(phi, psi)
        ax.set_ylabel('Psi')
        ax.yaxis.labelpad = 0
        plt.legend([train_loss_plot],
                   ['Phi psi'])
        ax.set_xlabel('Phi', color='black')
    return draw_with_vars

def write_result_summary(accuracy):
    output_string = globals().get("experiment_id") + ": " + str(accuracy) + "\n"
    with open("output/result_summary.txt", "a+") as output_file:
        output_file.write(output_string)
        output_file.flush()
    print(output_string, end="")

def calculate_dihedral_angles_over_minibatch(original_aa_sequence, atomic_coords):
    angles = []
    for idx, aa_sequence in enumerate(original_aa_sequence):
        res = calculate_dihedral_angels(atomic_coords[idx])
        actual_angles_t = torch.stack((res[0],res[1],res[2])).transpose(0,1)
        angles.append(actual_angles_t)
    return torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.pack_sequence(angles))

def protein_id_to_str(protein_id_list):
    _aa_dict_inverse = {v: k for k, v in AA_ID_DICT.items()}
    aa_list = []
    for a in protein_id_list:
        aa_symbol = _aa_dict_inverse[int(a)]
        aa_list.append(aa_symbol)
    return aa_list

def calculate_dihedral_angels(atomic_coords):

    assert int(atomic_coords.shape[1]) == 9
    atomic_coords = list([v for v in atomic_coords.contiguous().view(-1,3)])
    phi_list = [torch.tensor(0.0)]
    psi_list = []
    omega_list = []
    for i, coord in enumerate(atomic_coords):
        # TODO: This should be implemented in a GPU friendly way
        #if int(original_aa_sequence[int(i/3)]) == 0:
        #    print("ERROR: Reached end of protein, stopping")
        #    break

        if i % 3 == 0:
            if i != 0:
                phi_list.append(calculate_dihedral_pytorch(atomic_coords[i - 1],
                                                           atomic_coords[i],
                                                           atomic_coords[i + 1],
                                                           atomic_coords[i + 2]))
            if i+3 < len(atomic_coords):
                psi_list.append(calculate_dihedral_pytorch(atomic_coords[i],
                                                           atomic_coords[i + 1],
                                                           atomic_coords[i + 2],
                                                           atomic_coords[i + 3]))
                omega_list.append(calculate_dihedral_pytorch(atomic_coords[i + 1],
                                                             atomic_coords[i + 2],
                                                             atomic_coords[i + 3],
                                                             atomic_coords[i + 4]))
    psi_list.append(torch.tensor(0.0))
    omega_list.append(torch.tensor(0.0))
    return (torch.stack(phi_list), torch.stack(psi_list), torch.stack(omega_list))

def calculate_dihedral_pytorch(a, b, c, d):
    bc = c - b
    u = torch.cross(a - b, bc)
    v = torch.cross(d - c, bc)
    dihedral_angle = calc_angle_between_vec(u,v)
    try:
        if calc_angle_between_vec(bc,torch.cross(u, v)) > 0.00001:
            return -dihedral_angle
        else:
            return dihedral_angle
    except ZeroDivisionError:
        return dihedral_angle

def calc_angle_between_vec(a, b):
    return torch.acos(
        torch.min(
            torch.max(
                (torch.dot(a, b)) / (a.norm() * b.norm()),
                torch.tensor(-1.0)
            ),
            torch.tensor(1.0)
        )
    )

def get_structure_from_angles(aa_list, phi_list, psi_list, omega_list):
    assert len(aa_list) == len(phi_list)+1 == len(psi_list)+1 == len(omega_list)+1
    structure = PeptideBuilder.make_structure(aa_list,
                                              list(map(lambda x: math.degrees(x), phi_list)),
                                              list(map(lambda x: math.degrees(x), psi_list)),
                                              list(map(lambda x: math.degrees(x), omega_list)))
    return structure

def write_to_pdb(structure, prot_id):
    out = Bio.PDB.PDBIO()
    out.set_structure(structure)
    out.save("output/protein_" + str(prot_id) + ".pdb")

def calc_pairwise_distances(chain_a, chain_b, use_gpu):
    distance_matrix = torch.Tensor(chain_a.size()[0], chain_b.size()[0]).type(torch.float)
    # add small epsilon to avoid boundary issues
    epsilon = 10 ** (-4) * torch.ones(chain_a.size(0), chain_b.size(0))
    if use_gpu:
        distance_matrix = distance_matrix.cuda()
        epsilon = epsilon.cuda()

    for i, row in enumerate(chain_a.split(1)):
        distance_matrix[i] = torch.sum((row.expand_as(chain_b) - chain_b) ** 2, 1).view(1, -1)

    return torch.sqrt(distance_matrix + epsilon)

def calc_drmsd(chain_a, chain_b, use_gpu=False):
    assert len(chain_a) == len(chain_b)
    distance_matrix_a = calc_pairwise_distances(chain_a, chain_a, use_gpu)
    distance_matrix_b = calc_pairwise_distances(chain_b, chain_b, use_gpu)
    return torch.norm(distance_matrix_a - distance_matrix_b, 2) \
            / math.sqrt((len(chain_a) * (len(chain_a) - 1)))

# method for translating a point cloud to its center of mass
def transpose_atoms_to_center_of_mass(x):
    # calculate com by summing x, y and z respectively
    # and dividing by the number of points
    centerOfMass = np.matrix([[x[0, :].sum() / x.shape[1]],
                    [x[1, :].sum() / x.shape[1]],
                    [x[2, :].sum() / x.shape[1]]])
    # translate points to com and return
    return x - centerOfMass

def calc_rmsd(chain_a, chain_b):
    # move to center of mass
    a = chain_a.numpy().transpose()
    b = chain_b.numpy().transpose()
    X = transpose_atoms_to_center_of_mass(a)
    Y = transpose_atoms_to_center_of_mass(b)

    R = Y * X.transpose()
    # extract the singular values
    _, S, _ = np.linalg.svd(R)
    # compute RMSD using the formular
    E0 = sum(list(np.linalg.norm(x) ** 2 for x in X.transpose())
             + list(np.linalg.norm(x) ** 2 for x in Y.transpose()))
    TraceS = sum(S)
    RMSD = np.sqrt((1 / len(X.transpose())) * (E0 - 2 * TraceS))
    return RMSD

def calc_angular_difference(a1, a2):
    a1 = a1.transpose(0,1).contiguous()
    a2 = a2.transpose(0,1).contiguous()
    sum = 0
    for idx, _ in enumerate(a1):
        assert a1[idx].shape[1] == 3
        assert a2[idx].shape[1] == 3
        a1_element = a1[idx].view(-1, 1)
        a2_element = a2[idx].view(-1, 1)
        sum += torch.sqrt(torch.mean(
            torch.min(torch.abs(a2_element - a1_element),
                      2 * math.pi - torch.abs(a2_element - a1_element)
                      ) ** 2))
    return sum / a1.shape[0]

def structures_to_backbone_atoms_list(structures):
    backbone_atoms_list = []
    for structure in structures:
        backbone_atoms_list.append(structure_to_backbone_atoms(structure))
    return backbone_atoms_list

def structure_to_backbone_atoms(structure):
    predicted_coords = []
    for res in structure.get_residues():
        predicted_coords.append(torch.Tensor(res["N"].get_coord()))
        predicted_coords.append(torch.Tensor(res["CA"].get_coord()))
        predicted_coords.append(torch.Tensor(res["C"].get_coord()))
    return torch.stack(predicted_coords).view(-1,9)

def get_structures_from_prediction(original_aa_string, emissions, batch_sizes):
    predicted_pos_list = list(
        [a[:batch_sizes[idx], :] for idx, a in enumerate(emissions.transpose(0, 1))])
    structures = []
    for idx, predicted_pos in enumerate(predicted_pos_list):
        structure = get_structure_from_angles(protein_id_to_str(original_aa_string[idx]),
                                              predicted_pos.detach().transpose(0, 1)[0][1:],
                                              predicted_pos.detach().transpose(0, 1)[1][:-1],
                                              predicted_pos.detach().transpose(0, 1)[2][:-1])
        structures.append(structure)
    return structures


def calc_avg_drmsd_over_minibatch(backbone_atoms_list, actual_coords_list):
    drmsd_avg = 0
    for idx, backbone_atoms in enumerate(backbone_atoms_list):
        actual_coords = actual_coords_list[idx].transpose(0, 1).contiguous().view(-1, 3)
        drmsd_avg += calc_drmsd(backbone_atoms.transpose(0, 1).contiguous().view(-1, 3), actual_coords) / int(actual_coords.shape[0])
    return drmsd_avg / len(backbone_atoms_list)


def intial_pos_from_aa_string(batch_aa_string):
    structures = []
    for aa_string in batch_aa_string:
        structure = get_structure_from_angles(protein_id_to_str(aa_string),
                                              np.repeat([-120], len(aa_string)-1),
                                              np.repeat([140], len(aa_string)-1),
                                              np.repeat([-370], len(aa_string)-1))
        structures.append(structure)
    return structures