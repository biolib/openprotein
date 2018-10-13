# This file is part of the OpenProtein project.
#
# @author Jeppe Hallgren
#
# For license information, please see the LICENSE file in the root directory.

import torch
import torch.utils.data
import h5py
from datetime import datetime
import numpy as np

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


def write_to_pdb(atomic_coords, aaSequence, identifier):
    aa_name_dict = {"A": "ALA",
                    "R": "ARG",
                    "N": "ASN",
                    "D": "ASP",
                    "C": "CYS",
                    "E": "GLU",
                    "Q": "GLN",
                    "G": "GLY",
                    "H": "HIS",
                    "I": "ILE",
                    "L": "LEU",
                    "K": "LYS",
                    "M": "MET",
                    "F": "PHE",
                    "P": "PRO",
                    "S": "SER",
                    "T": "THR",
                    "W": "TRP",
                    "Y": "TYR",
                    "V": "VAL",
                    }

    _aa_dict = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
                'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19,
                'Y': 20}
    _aa_dict_inverse = {v: k for k, v in _aa_dict.items()}

    atomic_coords = np.round(atomic_coords, 3)

    atom_names = ['N', 'CA', 'C'] * (len(atomic_coords / 3))
    out_file = open("protein_" + str(identifier) + ".pdb", 'w')

    res_number = 0
    for i, coord in enumerate(atomic_coords):
        if i % 3 == 0:
            res_number += 1
        if int(aaSequence[res_number - 1]) == 0:
            print("Reached end of protein, stopping")
            break
        amino_acid_name = aa_name_dict[_aa_dict_inverse[int(aaSequence[res_number - 1])]]
        out_file.write('ATOM' +
                       ' ' * 2 + ' ' * (5 - len(str(i + 1))) + ' ' +
                       str(i + 1) + ' ' * (4 - len(atom_names[i])) +
                       atom_names[i] + ' ' + amino_acid_name +
                       ' ' + "A" + ' ' * (4 - len(str(res_number))) +
                       str(res_number) + ' ' * 4 +
                       ' ' * (8 - len(str(float(coord[0])))) + str(float(coord[0])) +
                       ' ' * (8 - len(str(float(coord[1])))) + str(float(coord[1])) +
                       ' ' * (8 - len(str(float(coord[2])))) + str(float(coord[2])) +
                       '\n')
    out_file.close()