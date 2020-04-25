"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""


import argparse
import glob
import os
import torch
import torch.onnx


from util import encode_primary_string, get_structure_from_angles, write_to_pdb, \
    calculate_dihedral_angles_over_minibatch


def prediction():

    list_of_files = glob.glob('output/models/*')
    default_model_path = max(list_of_files, key=os.path.getctime)

    parser = argparse.ArgumentParser(
        description="OpenProtein - Prediction CLI"
    )
    parser.add_argument('--input_sequence', dest='input_sequence')
    parser.add_argument('--model_path', dest='model_path', default=default_model_path)
    parser.add_argument('--use_gpu', dest='use_gpu', default=False, type=bool)

    args, _ = parser.parse_known_args()

    print("Using model:", args.model_path)

    model = torch.load(args.model_path)

    input_sequences = [args.input_sequence]

    input_sequences_encoded = list(torch.IntTensor(encode_primary_string(aa))
                                   for aa in input_sequences)

    predicted_dihedral_angles, predicted_backbone_atoms, batch_sizes = \
        model(input_sequences_encoded)

    if predicted_dihedral_angles == []:
        predicted_dihedral_angles, _ = calculate_dihedral_angles_over_minibatch(
            predicted_backbone_atoms,
            batch_sizes,
            args.use_gpu)
    write_to_pdb(
        get_structure_from_angles(input_sequences_encoded[0], predicted_dihedral_angles[:, 0]),
        "prediction"
    )

    print("Wrote prediction to output/protein_prediction.pdb")


prediction()
