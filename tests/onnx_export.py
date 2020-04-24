"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""

import glob
import os
import torch
import torch.onnx

from util import encode_primary_string

def onnx_from_model(model, input_str, path):
    """Export to onnx"""
    torch.onnx.export(model, input_str, path, opset_version=10, verbose=True)

def predict():
    list_of_files = glob.glob('output/models/*')  # * means all if need specific format then *.csv
    model_path = max(list_of_files, key=os.path.getctime)

    print("Generating ONNX from model:", model_path)
    model = torch.load(model_path)

    input_sequences = [
        "SRSLVISTINQISEDSKEFYFTLDNGKTMFPSNSQAWGGEKFENGQRAFVIFNELEQPVNGYDYNIQVRDITKVLTKEIVTMDDEE" \
        "NTEEKIGDDKINATYMWISKDKKYLTIEFQYYSTHSEDKKHFLNLVINNKDNTDDEYINLEFRHNSERDSPDHLGEGYVSFKLDKI" \
        "EEQIEGKKGLNIRVRTLYDGIKNYKVQFP"]

    input_sequences_encoded = list(torch.IntTensor(encode_primary_string(aa))
                                   for aa in input_sequences)

    print("Exporting to ONNX...")

    output_path = "./tests/output/openprotein.onnx"
    onnx_from_model(model, input_sequences_encoded, output_path)

    print("Wrote ONNX to", output_path)

predict()
