"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""

import glob
import os.path
import os
import platform
import re
import numpy as np
import h5py
import torch

from util import calculate_dihedral_angles_over_minibatch, \
    get_backbone_positions_from_angles, encode_primary_string, write_out


MAX_SEQUENCE_LENGTH = 2000

def process_raw_data(use_gpu, raw_data_path="data/raw/*", force_pre_processing_overwrite=True):
    write_out("Starting pre-processing of raw data...")
    input_files = glob.glob(raw_data_path)
    write_out(input_files)
    input_files_filtered = filter_input_files(input_files)
    for file_path in input_files_filtered:
        if platform.system() == 'Windows':
            filename = file_path.split('\\')[-1]
        else:
            filename = file_path.split('/')[-1]
        preprocessed_file_name = "data/preprocessed/" + filename + ".hdf5"

        # check if we should remove any previously processed files
        if os.path.isfile(preprocessed_file_name):
            write_out("Preprocessed file for " + filename + " already exists.")
            if force_pre_processing_overwrite:
                write_out("force_pre_processing_overwrite flag set to True, "
                          "overwriting old file...")
                os.remove(preprocessed_file_name)
            else:
                write_out("Skipping pre-processing for this file...")

        if not os.path.isfile(preprocessed_file_name):
            process_file(file_path, preprocessed_file_name, use_gpu)
    write_out("Completed pre-processing.")


def read_protein_from_file(file_pointer):
    """The algorithm Defining Secondary Structure of Proteins (DSSP) uses information on e.g. the
    position of atoms and the hydrogen bonds of the molecule to determine the secondary structure
    (helices, sheets...).
    """
    dict_ = {}
    _dssp_dict = {'L': 0, 'H': 1, 'B': 2, 'E': 3, 'G': 4, 'I': 5, 'T': 6, 'S': 7}
    _mask_dict = {'-': 0, '+': 1}

    while True:
        next_line = file_pointer.readline()
        if next_line == '[ID]\n':
            id_ = file_pointer.readline()[:-1]
            dict_.update({'id': id_})
        elif next_line == '[PRIMARY]\n':
            primary = encode_primary_string(file_pointer.readline()[:-1])
            dict_.update({'primary': primary})
        elif next_line == '[EVOLUTIONARY]\n':
            evolutionary = []
            for _residue in range(21):
                evolutionary.append(\
                    [float(step) for step in file_pointer.readline().split()])
            dict_.update({'evolutionary': evolutionary})
        elif next_line == '[SECONDARY]\n':
            secondary = list([_dssp_dict[dssp] for dssp in file_pointer.readline()[:-1]])
            dict_.update({'secondary': secondary})
        elif next_line == '[TERTIARY]\n':
            tertiary = []
            # 3 dimension
            for _axis in range(3):
                tertiary.append(\
                [float(coord) for coord in file_pointer.readline().split()])
            dict_.update({'tertiary': tertiary})
        elif next_line == '[MASK]\n':
            mask = list([_mask_dict[aa] for aa in file_pointer.readline()[:-1]])
            dict_.update({'mask': mask})
            mask_str = ''.join(map(str, mask))

            write_out("-------------")
            # Check for missing AA coordinates
            missing_internal_aa = False
            sequence_end = len(mask)           # for now, assume no C-terminal truncation needed
            write_out("Reading the protein " + id_)
            if re.search(r'1+0+1+', mask_str) is not None:       # indicates missing coordinates
                missing_internal_aa = True
                write_out("One or more internal coordinates missing. Protein is discarded.")
            elif re.search(r'^0*$', mask_str) is not None:       # indicates no coordinates at all
                missing_internal_aa = True
                write_out("One or more internal coordinates missing. It will be discarded.")
            else:
                if mask[0] == 0:
                    write_out("Missing coordinates in the N-terminal end. Truncating protein.")
                # investigate when the sequence with coordinates start and finish
                sequence_start = re.search(r'1', mask_str).start()
                if re.search(r'10', mask_str) is not None:   # missing coords in the C-term end
                    sequence_end = re.search(r'10', mask_str).start() + 1
                    write_out("Missing coordinates in the C-term end. Truncating protein.")
                write_out("Analyzing amino acids", sequence_start + 1, "-", sequence_end)

                # split lists in dict to have the seq with coords
                # separated from what should not be analysed
                if 'secondary' in dict_:
                    dict_.update({'secondary': secondary[sequence_start:sequence_end]})
                dict_.update({'primary': primary[sequence_start:sequence_end]})
                dict_.update({'mask': mask[sequence_start:sequence_end]})
                for elem in range(len(dict_['evolutionary'])):
                    dict_['evolutionary'][elem] = \
                        dict_['evolutionary'][elem][sequence_start:sequence_end]
                for elem in range(len(dict_['tertiary'])):
                    dict_['tertiary'][elem] = \
                        dict_['tertiary'][elem][sequence_start * 3:sequence_end * 3]

        elif next_line == '\n':
            return dict_, missing_internal_aa
        elif next_line == '':
            if dict_:
                return dict_, missing_internal_aa
            else:
                return None, False

def process_file(input_file, output_file, use_gpu):
    write_out("Processing raw data file", input_file)
    # create output file
    file = h5py.File(output_file, 'w')
    current_buffer_size = 1
    current_buffer_allocation = 0
    dset1 = file.create_dataset('primary', (current_buffer_size, MAX_SEQUENCE_LENGTH),
                                maxshape=(None, MAX_SEQUENCE_LENGTH), dtype='int32')
    dset2 = file.create_dataset('tertiary', (current_buffer_size, MAX_SEQUENCE_LENGTH, 9),
                                maxshape=(None, MAX_SEQUENCE_LENGTH, 9), dtype='float')
    dset3 = file.create_dataset('mask', (current_buffer_size, MAX_SEQUENCE_LENGTH),
                                maxshape=(None, MAX_SEQUENCE_LENGTH),
                                dtype='uint8')

    input_file_pointer = open(input_file, "r")

    while True:
        # while there's more proteins to process
        next_protein, missing_aa = read_protein_from_file(input_file_pointer)
        if next_protein is None: # no more proteins to process
            break

        sequence_length = len(next_protein['primary'])

        if sequence_length > MAX_SEQUENCE_LENGTH:
            write_out("Dropping protein as length too long:", sequence_length)
            continue
        if missing_aa is True:
            continue
        if current_buffer_allocation >= current_buffer_size:
            current_buffer_size = current_buffer_size + 1
            dset1.resize((current_buffer_size, MAX_SEQUENCE_LENGTH))
            dset2.resize((current_buffer_size, MAX_SEQUENCE_LENGTH, 9))
            dset3.resize((current_buffer_size, MAX_SEQUENCE_LENGTH))

        primary_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        tertiary_padded = np.zeros((9, MAX_SEQUENCE_LENGTH))
        mask_padded = np.zeros(MAX_SEQUENCE_LENGTH)

        # masking and padding here happens so that the stored dataset is of the same size.
        # when the data is loaded in this padding is removed again.
        primary_padded[:sequence_length] = next_protein['primary']
        t_transposed = np.ravel(np.array(next_protein['tertiary']).T)
        t_reshaped = np.reshape(t_transposed, (sequence_length, 9)).T
        tertiary_padded[:, :sequence_length] = t_reshaped
        mask_padded[:sequence_length] = next_protein['mask']
        mask = torch.Tensor(mask_padded).type(dtype=torch.bool)
        prim = torch.masked_select(torch.Tensor(primary_padded)\
                                   .type(dtype=torch.long), mask)
        pos = torch.masked_select(torch.Tensor(tertiary_padded), mask)\
                  .view(9, -1).transpose(0, 1).unsqueeze(1)
        pos_angstrom = pos / 100

        if use_gpu:
            pos_angstrom = pos_angstrom.cuda()

        # map to angles and back to tertiary
        angles, batch_sizes = calculate_dihedral_angles_over_minibatch(pos_angstrom,
                                                                       torch.tensor([len(prim)]),
                                                                       use_gpu=use_gpu)

        tertiary, _ = get_backbone_positions_from_angles(angles,
                                                         batch_sizes,
                                                         use_gpu=use_gpu)
        tertiary = tertiary.squeeze(1)

        # create variables to store padded sequences in
        primary_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        tertiary_padded = np.zeros((MAX_SEQUENCE_LENGTH, 9))
        mask_padded = np.zeros(MAX_SEQUENCE_LENGTH)

        # store padded sequences
        length_after_mask_removed = len(prim)
        primary_padded[:length_after_mask_removed] = prim.data.cpu().numpy()
        tertiary_padded[:length_after_mask_removed, :] = tertiary.data.cpu().numpy()
        mask_padded[:length_after_mask_removed] = np.ones(length_after_mask_removed)

        # save padded sequences on disk
        dset1[current_buffer_allocation] = primary_padded
        dset2[current_buffer_allocation] = tertiary_padded
        dset3[current_buffer_allocation] = mask_padded
        current_buffer_allocation += 1
    if current_buffer_allocation == 0:
        write_out("Preprocessing was selected but no proteins in the input file "
                  "were accepted. Please check your input.")
        os._exit(1)
    write_out("Wrote output to", current_buffer_allocation, "proteins to", output_file)


def filter_input_files(input_files):
    disallowed_file_endings = (".gitignore", ".DS_Store")
    return list(filter(lambda x: not x.endswith(disallowed_file_endings), input_files))
