"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""

import glob
import os
import torch
import torch.onnx
import numpy as np

from experiments.tmhmm3 import decode, decode_numpy
from util import load_model_from_disk


def onnx_from_model(model, input_str, path):
    """Export to onnx"""
    torch.onnx.export(model, input_str, path,
                      enable_onnx_checker=True, opset_version=10, verbose=True,
                      input_names=['embedded_sequences', 'mask'],   # the model's input names
                      output_names=['emissions',
                                    'crf_start_transitions',
                                    'crf_transitions',
                                    'crf_end_transitions'],  # the model's output names
                      dynamic_axes={
                          'mask': {0: 'batch_size'},
                          'embedded_sequences': {0: 'max_seq_length', 1: 'batch_size'},
                          'emissions': {0: 'max_seq_length', 1: 'batch_size'},
                      }
                      )

def predict():
    list_of_files = glob.glob('output/models/*')  # * means all if need specific format then *.csv
    model_path = max(list_of_files, key=os.path.getctime)

    print("Generating ONNX from model:", model_path)
    model = load_model_from_disk(model_path, force_cpu=True)

    input_sequences = [
        "AAAAAAA", "AAA"]

    input_sequences_embedded = [x for x in model.embed(input_sequences)]

    input_sequences_padded = torch.nn.utils.rnn.pad_sequence(input_sequences_embedded)

    batch_sizes_list = []
    for x in input_sequences:
        batch_sizes_list.append(len(x))

    batch_sizes = torch.IntTensor(batch_sizes_list)

    emmissions, start_transitions, transitions, end_transitions = model(input_sequences_padded)
    predicted_labels, predicted_types, predicted_topologies = decode(emmissions,
                                                                     batch_sizes,
                                                                     start_transitions,
                                                                     transitions,
                                                                     end_transitions)
    predicted_labels_2, predicted_types_2, predicted_topologies_2 = \
        decode_numpy(emmissions.detach().numpy(),
                     batch_sizes.detach().numpy(),
                     start_transitions.detach().numpy(),
                     transitions.detach().numpy(),
                     end_transitions.detach().numpy())
    for idx, val in enumerate(predicted_labels):
        assert np.array_equal(val.detach().numpy(), predicted_labels_2[idx])
    assert np.array_equal(predicted_types.detach().numpy(), predicted_types_2)
    for idx, val in enumerate(predicted_topologies):
        for idx2, val2 in enumerate(val):
            assert np.array_equal(val2.detach().numpy(), predicted_topologies_2[idx][idx2])

    print("Exporting to ONNX...")

    output_path = "./tests/output/tmhmm3.onnx"

    onnx_from_model(model, (input_sequences_padded), output_path)

    print("Wrote ONNX to", output_path)

predict()
