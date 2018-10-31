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

AA_ID_DICT = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9,
              'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17,
              'V': 18, 'W': 19,'Y': 20}

def contruct_dataloader_from_disk(filename, minibatch_size):
    return torch.utils.data.DataLoader(H5PytorchDataset(filename), batch_size=minibatch_size, shuffle=True)


class H5PytorchDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        super(H5PytorchDataset, self).__init__()

        self.h5pyfile = h5py.File(filename, 'r')
        self.num_proteins, self.max_sequence_len = self.h5pyfile['primary'].shape

    def __getitem__(self, index):
        return self.h5pyfile['primary'][index,:] , self.h5pyfile['tertiary'][index,:] , self.h5pyfile['mask'][index,:]

    def __len__(self):
        return self.num_proteins

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
    for i, data in enumerate(data_loader, 0):
        primary_sequence, tertiary_positions, mask = data

        predicted_positions = model(primary_sequence)

        minibatch_data = list(zip(primary_sequence,
                                  tertiary_positions,
                                  predicted_positions,
                                  mask))
        data_total.extend(minibatch_data)
        for primary_sequence, tertiary_positions,predicted_positions, mask in minibatch_data:
            error = 1
            loss += error
    loss /= data_loader.dataset.__len__()
    return (loss, data_total)

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

def write_result_summary(accuracy):
    output_string = globals().get("experiment_id") + ": " + str(accuracy) + "\n"
    with open("output/result_summary.txt", "a+") as output_file:
        output_file.write(output_string)
        output_file.flush()
    print(output_string, end="")


def write_to_pdb(atomic_coords, aaSequence, prot_id):
    _aa_dict_inverse = {v: k for k, v in AA_ID_DICT.items()}
    atomic_coords = list([Vector(v) for v in atomic_coords.numpy()])
    aa_list = []
    phi_list = []
    psi_list = []
    omega_list = []
    for i, coord in enumerate(atomic_coords):
        if int(aaSequence[int(i/3)]) == 0:
            print("Reached end of protein, stopping")
            break

        if i % 3 == 0:
            aa_symbol = _aa_dict_inverse[int(aaSequence[int(i/3)])]
            aa_list.append(aa_symbol)

            if i != 0:
                phi_list.append(math.degrees(Bio.PDB.calc_dihedral(atomic_coords[i - 1],
                                                                   atomic_coords[i],
                                                                   atomic_coords[i + 1],
                                                                   atomic_coords[i + 2])))
            if i+3 < len(atomic_coords) and int(aaSequence[int(i/3)+1]) != 0:
                psi_list.append(math.degrees(Bio.PDB.calc_dihedral(atomic_coords[i],
                                                                   atomic_coords[i + 1],
                                                                   atomic_coords[i + 2],
                                                                   atomic_coords[i + 3])))
                omega_list.append(math.degrees(Bio.PDB.calc_dihedral(atomic_coords[i + 1],
                                                                     atomic_coords[i + 2],
                                                                     atomic_coords[i + 3],
                                                                     atomic_coords[i + 4])))

    out = Bio.PDB.PDBIO()
    structure = PeptideBuilder.make_structure(aa_list, phi_list, psi_list, omega_list)
    out.set_structure(structure)
    out.save("output/protein_" + str(prot_id) + ".pdb")
