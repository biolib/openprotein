"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""

import collections
import math
import os
from datetime import datetime
import torch
import torch.utils.data
import torch.nn.functional as F
import h5py
import PeptideBuilder
import Bio.PDB
from Bio.Data.IUPACData import protein_letters_1to3
import numpy as np
from torch.nn.utils.rnn import pad_sequence

AA_ID_DICT = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9,
              'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17,
              'V': 18, 'W': 19, 'Y': 20}

PI_TENSOR = torch.tensor([3.141592])

def contruct_dataloader_from_disk(filename, minibatch_size):
    return torch.utils.data.DataLoader(H5PytorchDataset(filename),
                                       batch_size=minibatch_size,
                                       shuffle=True,
                                       collate_fn=merge_samples_to_minibatch)


class H5PytorchDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        super(H5PytorchDataset, self).__init__()

        self.h5pyfile = h5py.File(filename, 'r')
        self.num_proteins, self.max_sequence_len = self.h5pyfile['primary'].shape

    def __getitem__(self, index):
        mask = torch.Tensor(self.h5pyfile['mask'][index, :]).type(dtype=torch.bool)
        prim = torch.masked_select(
            torch.Tensor(self.h5pyfile['primary'][index, :]).type(dtype=torch.int),
            mask)
        tertiary = torch.Tensor(self.h5pyfile['tertiary'][index][:int(mask.sum())])# max length x 9
        return prim, tertiary, mask

    def __len__(self):
        return self.num_proteins


def merge_samples_to_minibatch(samples):
    samples_list = []
    for sample in samples:
        samples_list.append(sample)
    # sort according to length of aa sequence
    samples_list.sort(key=lambda x: len(x[0]), reverse=True)
    return zip(*samples_list)

def set_experiment_id(data_set_identifier, learning_rate, minibatch_size):
    output_string = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    output_string += "-" + str(os.getpid())
    output_string += "-" + data_set_identifier
    output_string += "-LR" + str(learning_rate).replace(".", "_")
    output_string += "-MB" + str(minibatch_size)
    globals().__setitem__("experiment_id", output_string)


def get_experiment_id():
    return globals().get("experiment_id")


def write_out(*args, end='\n'):
    output_string = datetime.now().strftime('%Y-%m-%d %H:%M:%S') \
                    + ": " + str.join(" ", [str(a) for a in args]) + end
    if globals().get("experiment_id") is not None:
        with open("output/" + globals().get("experiment_id") + ".txt", "a+") as output_file:
            output_file.write(output_string)
            output_file.flush()
    print(output_string, end="")


def write_model_to_disk(model):
    path = "output/models/" + globals().get("experiment_id") + ".model"
    torch.save(model, path)
    return path


def write_prediction_data_to_disk(prediction_data):
    filepath = "output/predictions/" + globals().get("experiment_id") + ".txt"
    output_file = open(filepath, 'w')
    output_file.write(prediction_data)
    output_file.close()


def draw_plot(fig, plt, validation_dataset_size, sample_num, train_loss_values,
              validation_loss_values):
    def draw_with_vars():
        ax = fig.gca()
        ax2 = ax.twinx()
        plt.grid(True)
        plt.title("Training progress (" + str(validation_dataset_size)
                  + " samples in validation set)")
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


def calculate_dihedral_angles_over_minibatch(atomic_coords_padded, batch_sizes, use_gpu):
    angles = []
    batch_sizes = torch.LongTensor(batch_sizes)
    atomic_coords = atomic_coords_padded.transpose(0, 1)

    for idx, coordinate in enumerate(atomic_coords.split(1, dim=0)):
        angles_from_coords = torch.index_select(
            coordinate.squeeze(0),
            0,
            torch.arange(int(batch_sizes[idx].item()))
        )
        angles.append(calculate_dihedral_angles(angles_from_coords, use_gpu))

    return torch.nn.utils.rnn.pad_sequence(angles), batch_sizes

def protein_id_to_str(protein_id_list):
    _aa_dict_inverse = {v: k for k, v in AA_ID_DICT.items()}
    aa_list = []
    for protein_id in protein_id_list:
        aa_symbol = _aa_dict_inverse[protein_id.item()]
        aa_list.append(aa_symbol)
    return aa_list


def calculate_dihedral_angles(atomic_coords, use_gpu):
    #assert atomic_coords.shape[1] == 9
    atomic_coords = atomic_coords.contiguous().view(-1, 3)

    zero_tensor = torch.zeros(1)
    if use_gpu:
        zero_tensor = zero_tensor.cuda()



    angles = torch.cat((zero_tensor,
                        zero_tensor,
                        compute_dihedral_list(atomic_coords),
                        zero_tensor)).view(-1, 3)
    return angles

def compute_cross(tensor_a, tensor_b, dim):

    result = []

    x = torch.zeros(1).long()
    y = torch.ones(1).long()
    z = torch.ones(1).long() * 2

    ax = torch.index_select(tensor_a, dim, x).squeeze(dim)
    ay = torch.index_select(tensor_a, dim, y).squeeze(dim)
    az = torch.index_select(tensor_a, dim, z).squeeze(dim)

    bx = torch.index_select(tensor_b, dim, x).squeeze(dim)
    by = torch.index_select(tensor_b, dim, y).squeeze(dim)
    bz = torch.index_select(tensor_b, dim, z).squeeze(dim)

    result.append(ay * bz - az * by)
    result.append(az * bx - ax * bz)
    result.append(ax * by - ay * bx)

    result = torch.stack(result, dim=dim)

    return result


def compute_atan2(y_coord, x_coord):
    # TODO: figure out of eps is needed here
    eps = 10 ** (-4)
    ans = torch.atan(y_coord / (x_coord + eps)) # x > 0
    ans = torch.where((y_coord >= 0) & (x_coord < 0), ans + PI_TENSOR, ans)
    ans = torch.where((y_coord < 0) & (x_coord < 0), ans - PI_TENSOR, ans)
    ans = torch.where((y_coord > 0) & (x_coord == 0), PI_TENSOR / 2, ans)
    ans = torch.where((y_coord < 0) & (x_coord == 0), -PI_TENSOR / 2, ans)
    return ans


def compute_dihedral_list(atomic_coords):
    # atomic_coords is -1 x 3
    ba = atomic_coords[1:] - atomic_coords[:-1]
    ba_normalized = ba / ba.norm(dim=1).unsqueeze(1)
    ba_neg = -1 * ba_normalized

    n1_vec = compute_cross(ba_normalized[:-2], ba_neg[1:-1], dim=1)
    n2_vec = compute_cross(ba_neg[1:-1], ba_normalized[2:], dim=1)

    n1_vec_normalized = n1_vec / n1_vec.norm(dim=1).unsqueeze(1)
    n2_vec_normalized = n2_vec / n2_vec.norm(dim=1).unsqueeze(1)

    m1_vec = compute_cross(n1_vec_normalized, ba_neg[1:-1], dim=1)

    x_value = torch.sum(n1_vec_normalized * n2_vec_normalized, dim=1)
    y_value = torch.sum(m1_vec * n2_vec_normalized, dim=1)
    return compute_atan2(y_value, x_value)


def write_pdb(file_name, aa_sequence, residue_coords):
    residue_names = list([protein_letters_1to3[l].upper() for l in aa_sequence])
    num_atoms = len(residue_coords)
    backbone_names = num_atoms * ["N", "CA", "C"]

    assert num_atoms == len(aa_sequence) * 3
    file = open(file_name, 'w')

    for i in range(num_atoms):
        atom_coordinates = list([str(l) for l in np.round(residue_coords[i], 3)])
        residue_position = int(i / 3)
        atom_id = str(i + 1)
        file.write(f"""\
ATOM  \
{atom_id.rjust(5)} \
{backbone_names[i].rjust(4)} \
{residue_names[residue_position - 1].rjust(3)} \
A\
{str(residue_position).rjust(4)}    \
{atom_coordinates[0].rjust(8)}\
{atom_coordinates[1].rjust(8)}\
{atom_coordinates[2].rjust(8)}\
\n""")
    file.close()

def get_structure_from_angles(aa_list_encoded, angles):
    aa_list = protein_id_to_str(aa_list_encoded)
    omega_list = angles[1:, 0]
    phi_list = angles[1:, 1]
    psi_list = angles[:-1, 2]
    assert len(aa_list) == len(phi_list) + 1 == len(psi_list) + 1 == len(omega_list) + 1
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

    for idx, row in enumerate(chain_a.split(1)):
        distance_matrix[idx] = torch.sum((row.expand_as(chain_b) - chain_b) ** 2, 1).view(1, -1)

    return torch.sqrt(distance_matrix + epsilon)


def calc_drmsd(chain_a, chain_b, use_gpu=False):
    assert len(chain_a) == len(chain_b)
    distance_matrix_a = calc_pairwise_distances(chain_a, chain_a, use_gpu)
    distance_matrix_b = calc_pairwise_distances(chain_b, chain_b, use_gpu)
    return torch.norm(distance_matrix_a - distance_matrix_b, 2) \
           / math.sqrt((len(chain_a) * (len(chain_a) - 1)))


# method for translating a point cloud to its center of mass
def transpose_atoms_to_center_of_mass(atoms_matrix):
    # calculate com by summing x, y and z respectively
    # and dividing by the number of points
    center_of_mass = np.matrix([[atoms_matrix[0, :].sum() / atoms_matrix.shape[1]],
                                [atoms_matrix[1, :].sum() / atoms_matrix.shape[1]],
                                [atoms_matrix[2, :].sum() / atoms_matrix.shape[1]]])
    # translate points to com and return
    return atoms_matrix - center_of_mass


def calc_rmsd(chain_a, chain_b):
    # move to center of mass
    chain_a_value = chain_a.cpu().numpy().transpose()
    chain_b_value = chain_b.cpu().numpy().transpose()
    X = transpose_atoms_to_center_of_mass(chain_a_value)
    Y = transpose_atoms_to_center_of_mass(chain_b_value)

    R = Y * X.transpose()
    # extract the singular values
    _, S, _ = np.linalg.svd(R)
    # compute RMSD using the formular
    E0 = sum(list(np.linalg.norm(x) ** 2 for x in X.transpose())
             + list(np.linalg.norm(x) ** 2 for x in Y.transpose()))
    TraceS = sum(S)
    RMSD = np.sqrt((1 / len(X.transpose())) * (E0 - 2 * TraceS))
    return RMSD


def calc_angular_difference(values_1, values_2):
    values_1 = values_1.transpose(0, 1).contiguous()
    values_2 = values_2.transpose(0, 1).contiguous()
    acc = 0
    for idx, _ in enumerate(values_1):
        assert values_1[idx].shape[1] == 3
        assert values_2[idx].shape[1] == 3
        a1_element = values_1[idx].view(-1, 1)
        a2_element = values_2[idx].view(-1, 1)
        acc += torch.sqrt(torch.mean(
            torch.min(torch.abs(a2_element - a1_element),
                      2 * math.pi - torch.abs(a2_element - a1_element)
                      ) ** 2))
    return acc / values_1.shape[0]


def structures_to_backbone_atoms_padded(structures):
    backbone_atoms_list = []
    for structure in structures:
        backbone_atoms_list.append(structure_to_backbone_atoms(structure))
    backbone_atoms_padded, batch_sizes_backbone = torch.nn.utils.rnn.pad_packed_sequence(
        torch.nn.utils.rnn.pack_sequence(backbone_atoms_list))
    return backbone_atoms_padded, batch_sizes_backbone


def structure_to_backbone_atoms(structure):
    predicted_coords = []
    for res in structure.get_residues():
        predicted_coords.append(torch.Tensor(res["N"].get_coord()))
        predicted_coords.append(torch.Tensor(res["CA"].get_coord()))
        predicted_coords.append(torch.Tensor(res["C"].get_coord()))
    return torch.stack(predicted_coords).view(-1, 9)

NUM_FRAGMENTS = torch.tensor(6)
def get_backbone_positions_from_angles(angular_emissions, batch_sizes, use_gpu):
    # angular_emissions -1 x minibatch size x 3 (omega, phi, psi)
    points = dihedral_to_point(angular_emissions, use_gpu)
    coordinates = point_to_coordinate(
        points,
        use_gpu,
        num_fragments=NUM_FRAGMENTS) / 100  # divide by 100 to angstrom unit
    return coordinates.transpose(0, 1).contiguous()\
               .view(len(batch_sizes), -1, 9).transpose(0, 1), batch_sizes


def calc_avg_drmsd_over_minibatch(backbone_atoms_padded, actual_coords_padded, batch_sizes):
    backbone_atoms_list = list(
        [backbone_atoms_padded[:batch_sizes[i], i] for i in range(int(backbone_atoms_padded
                                                                      .size(1)))])
    actual_coords_list = list(
        [actual_coords_padded[:batch_sizes[i], i] for i in range(int(actual_coords_padded
                                                                     .size(1)))])
    drmsd_avg = 0
    for idx, backbone_atoms in enumerate(backbone_atoms_list):
        actual_coords = actual_coords_list[idx].transpose(0, 1).contiguous().view(-1, 3)
        drmsd_avg += calc_drmsd(backbone_atoms.transpose(0, 1).contiguous().view(-1, 3),
                                actual_coords)
    return drmsd_avg / len(backbone_atoms_list)


def encode_primary_string(primary):
    return list([AA_ID_DICT[aa] for aa in primary])


def initial_pos_from_aa_string(batch_aa_string, use_gpu):
    arr_of_angles = []
    batch_sizes = []
    for aa_string in batch_aa_string:
        length_of_protein = aa_string.size(0)
        angles = torch.stack([-120*torch.ones(length_of_protein),
                              140*torch.ones(length_of_protein),
                              -370*torch.ones(length_of_protein)]).transpose(0, 1)
        arr_of_angles.append(angles)
        batch_sizes.append(length_of_protein)

    padded = pad_sequence(arr_of_angles).transpose(0, 1)
    return get_backbone_positions_from_angles(padded, batch_sizes, use_gpu)

def pass_messages(aa_features, message_transformation, use_gpu):
    # aa_features (#aa, #features) - each row represents the amino acid type
    # (embedding) and the positions of the backbone atoms
    # message_transformation: (-1 * 2 * feature_size) -> (-1 * output message size)
    feature_size = aa_features.size(1)
    aa_count = aa_features.size(0)

    arange2d = torch.arange(aa_count).repeat(aa_count).view((aa_count, aa_count))

    diagonal_matrix = (arange2d == arange2d.transpose(0, 1)).int()

    eye = diagonal_matrix.view(-1).expand(2, feature_size, -1)\
        .transpose(1, 2).transpose(0, 1)

    eye_inverted = torch.ones(eye.size(), dtype=torch.uint8) - eye
    if use_gpu:
        eye_inverted = eye_inverted.cuda()
    features_repeated = aa_features.repeat((aa_count, 1)).view((aa_count, aa_count, feature_size))
    # (aa_count^2 - aa_count) x 2 x aa_features     (all pairs except for reflexive connections)
    aa_messages = torch.stack((features_repeated.transpose(0, 1), features_repeated))\
        .transpose(0, 1).transpose(1, 2).view(-1, 2, feature_size)

    eye_inverted_location = eye_inverted.view(-1).nonzero().squeeze(1)

    aa_msg_pairs = aa_messages\
        .reshape(-1).gather(0, eye_inverted_location).view(-1, 2, feature_size)

    transformed = message_transformation(aa_msg_pairs).view(aa_count, aa_count - 1, -1)
    transformed_sum = transformed.sum(dim=1)  # aa_count x output message size
    return transformed_sum


def load_model_from_disk(path, force_cpu=True):
    if force_cpu:
        # load model with map_location set to storage (main mem)
        model = torch.load(path, map_location=lambda storage, loc: storage)
        # flattern parameters in memory
        model.flatten_parameters()
        # update internal state accordingly
        model.use_gpu = False
    else:
        # load model using default map_location
        model = torch.load(path)
        model.flatten_parameters()
    return model

# Constants
NUM_DIMENSIONS = 3
NUM_DIHEDRALS = 3
BOND_LENGTHS = torch.tensor([145.801, 152.326, 132.868], dtype=torch.float32)
BOND_ANGLES = torch.tensor([2.124, 1.941, 2.028], dtype=torch.float32)


def dihedral_to_point(dihedral, use_gpu, bond_lengths=BOND_LENGTHS,
                      bond_angles=BOND_ANGLES):
    """
    Takes triplets of dihedral angles (phi, psi, omega) and returns 3D points
    ready for use in reconstruction of coordinates. Bond lengths and angles
    are based on idealized averages.
    :param dihedral: [NUM_STEPS, BATCH_SIZE, NUM_DIHEDRALS]
    :return: Tensor containing points of the protein's backbone atoms.
    Shape [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
    """
    num_steps = dihedral.shape[0]
    batch_size = dihedral.shape[1]

    r_cos_theta = bond_lengths * torch.cos(PI_TENSOR - bond_angles)
    r_sin_theta = bond_lengths * torch.sin(PI_TENSOR - bond_angles)

    if use_gpu:
        r_cos_theta = r_cos_theta.cuda()
        r_sin_theta = r_sin_theta.cuda()

    point_x = r_cos_theta.view(1, 1, -1).repeat(num_steps, batch_size, 1)
    point_y = torch.cos(dihedral) * r_sin_theta
    point_z = torch.sin(dihedral) * r_sin_theta

    point = torch.stack([point_x, point_y, point_z])
    point_perm = point.permute(1, 3, 2, 0)
    point_final = point_perm.contiguous().view(num_steps * NUM_DIHEDRALS,
                                               batch_size,
                                               NUM_DIMENSIONS)
    return point_final

PNERF_INIT_MATRIX = [torch.tensor([-torch.sqrt(torch.tensor([1.0 / 2.0])),
                                   torch.sqrt(torch.tensor([3.0 / 2.0])), 0]),
                     torch.tensor([-torch.sqrt(torch.tensor([2.0])), 0, 0]),
                     torch.tensor([0, 0, 0])]

def point_to_coordinate(points, use_gpu, num_fragments):
    """
    Takes points from dihedral_to_point and sequentially converts them into
    coordinates of a 3D structure.

    Reconstruction is done in parallel by independently reconstructing
    num_fragments and the reconstituting the chain at the end in reverse order.
    The core reconstruction algorithm is NeRF, based on
    DOI: 10.1002/jcc.20237 by Parsons et al. 2005.
    The parallelized version is described in
    https://www.biorxiv.org/content/early/2018/08/06/385450.
    :param points: Tensor containing points as returned by `dihedral_to_point`.
    Shape [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
    :param num_fragments: Number of fragments in which the sequence is split
    to perform parallel computation.
    :return: Tensor containing correctly transformed atom coordinates.
    Shape [NUM_STEPS x NUM_DIHEDRALS, BATCH_SIZE, NUM_DIMENSIONS]
    """

    # Compute optimal number of fragments if needed
    total_num_angles = points.size(0)  # NUM_STEPS x NUM_DIHEDRALS
    if isinstance(total_num_angles, int):
        total_num_angles = torch.tensor(total_num_angles)

    # Initial three coordinates (specifically chosen to eliminate need for
    # extraneous matmul)
    Triplet = collections.namedtuple('Triplet', 'a, b, c')
    batch_size = points.shape[1]

    init_coords = []
    for row in PNERF_INIT_MATRIX:
        row_tensor = row\
                .repeat([num_fragments * batch_size, 1])\
                .view(num_fragments, batch_size, NUM_DIMENSIONS)
        if use_gpu:
            row_tensor = row_tensor.cuda()
        init_coords.append(row_tensor)

    init_coords = Triplet(*init_coords)  # NUM_DIHEDRALS x [NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS]

    # Pad points to yield equal-sized fragments
    # (NUM_FRAGS x FRAG_SIZE) - (NUM_STEPS x NUM_DIHEDRALS)
    padding = torch.fmod(num_fragments - (total_num_angles % num_fragments), num_fragments)

    # [NUM_FRAGS x FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
    padding_tensor = torch.zeros((padding, points.size(1), points.size(2)))
    points = torch.cat((points, padding_tensor))

    points = points.view(num_fragments, -1, batch_size,
                         NUM_DIMENSIONS)  # [NUM_FRAGS, FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
    points = points.permute(1, 0, 2, 3)  # [FRAG_SIZE, NUM_FRAGS, BATCH_SIZE, NUM_DIMENSIONS]

    # Extension function used for single atom reconstruction and whole fragment
    # alignment
    def extend(prev_three_coords, point, multi_m):
        """
        Aligns an atom or an entire fragment depending on value of `multi_m`
        with the preceding three atoms.
        :param prev_three_coords: Named tuple storing the last three atom
        coordinates ("a", "b", "c") where "c" is the current end of the
        structure (i.e. closest to the atom/ fragment that will be added now).
        Shape NUM_DIHEDRALS x [NUM_FRAGS/0, BATCH_SIZE, NUM_DIMENSIONS].
        First rank depends on value of `multi_m`.
        :param point: Point describing the atom that is added to the structure.
        Shape [NUM_FRAGS/FRAG_SIZE, BATCH_SIZE, NUM_DIMENSIONS]
        First rank depends on value of `multi_m`.
        :param multi_m: If True, a single atom is added to the chain for
        multiple fragments in parallel. If False, an single fragment is added.
        Note the different parameter dimensions.
        :return: Coordinates of the atom/ fragment.
        """
        bc = F.normalize(prev_three_coords.c - prev_three_coords.b, dim=-1)
        n = F.normalize(compute_cross(prev_three_coords.b - prev_three_coords.a,
                                      bc, dim=2 if multi_m else 1), dim=-1)
        if multi_m:  # multiple fragments, one atom at a time
            m = torch.stack([bc, compute_cross(n, bc, dim=2), n]).permute(1, 2, 3, 0)
        else:  # single fragment, reconstructed entirely at once.
            s = point.shape + (3,)
            m = torch.stack([bc, compute_cross(n, bc, dim=1), n]).permute(1, 2, 0)
            m = m.repeat(s[0], 1, 1).view(s)
        coord = torch.squeeze(torch.matmul(m, point.unsqueeze(3)),
                              dim=3) + prev_three_coords.c
        return coord

    # Loop over FRAG_SIZE in NUM_FRAGS parallel fragments, sequentially
    # generating the coordinates for each fragment across all batches
    coords_list = []
    prev_three_coords = init_coords

    for point in points.split(1, dim=0):  # Iterate over FRAG_SIZE
        coord = extend(prev_three_coords, point.squeeze(0), True)
        coords_list.append(coord)
        prev_three_coords = Triplet(prev_three_coords.b,
                                    prev_three_coords.c,
                                    coord)

    coords_pretrans = torch.stack(coords_list).permute(1, 0, 2, 3)

    # Loop backwards over NUM_FRAGS to align the individual fragments. For each
    # next fragment, we transform the fragments we have already iterated over
    # (coords_trans) to be aligned with the next fragment
    coords_trans = coords_pretrans[-1]
    for idx in torch.arange(end=-1, start=coords_pretrans.shape[0] - 2, step=-1).split(1, dim=0):
        # Transform the fragments that we have already iterated over to be
        # aligned with the next fragment `coords_trans`
        transformed_coords = extend(Triplet(*[di.index_select(0, idx).squeeze(0)
                                              for di in prev_three_coords]),
                                    coords_trans, False)
        coords_trans = torch.cat(
            [coords_pretrans.index_select(0, idx).squeeze(0), transformed_coords], 0)

    coords_to_pad = torch.index_select(coords_trans, 0, torch.arange(total_num_angles - 1))

    coords = F.pad(coords_to_pad, (0, 0, 0, 0, 1, 0))

    return coords
