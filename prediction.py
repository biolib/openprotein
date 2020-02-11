"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""

import torch

from util import encode_primary_string, get_structure_from_angles, write_to_pdb


def main():
    input_sequences = [
        "SRSLVISTINQISEDSKEFYFTLDNGKTMFPSNSQAWGGEKFENGQRAFVIFNELEQPVNGYDYNIQVRDITKVLTKEIVTMDDEE" \
        "NTEEKIGDDKINATYMWISKDKKYLTIEFQYYSTHSEDKKHFLNLVINNKDNTDDEYINLEFRHNSERDSPDHLGEGYVSFKLDKI" \
        "EEQIEGKKGLNIRVRTLYDGIKNYKVQFP"]
    model_path = "output/models/2019-01-30_00_38_46-TRAIN-LR0_01-MB1.model"

    model = torch.load(model_path)
    input_sequences_encoded = list(torch.LongTensor(encode_primary_string(aa))
                                   for aa in input_sequences)

    predicted_dihedral_angles, _predicted_backbone_atoms, _batch_sizes = \
        model(input_sequences_encoded)

    write_to_pdb(
        get_structure_from_angles(input_sequences_encoded[0], predicted_dihedral_angles[:, 0]),
        "myprediction"
    )

    print("Wrote prediction to output/protein_myprediction.pdb")


main()
