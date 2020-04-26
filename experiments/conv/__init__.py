"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""
import math
import time
import numpy as np
import torch
import torch.nn as nn
import openprotein
from preprocessing import process_raw_data
from training import train_model

from util import get_backbone_positions_from_angles, contruct_dataloader_from_disk
from util import calculate_dihedral_angles_over_minibatch, calc_angular_difference, \
    write_out, calculate_dihedral_angles, \
    get_structure_from_angles, write_to_pdb, calc_rmsd,\
    calc_drmsd, get_backbone_positions_from_angles

ANGLE_ARR = torch.tensor([[-120, 140, -370], [0, 120, -150], [25, -120, 150]]).float()

def run_experiment(parser, use_gpu):
    # parse experiment specific command line arguments
    parser.add_argument('--learning-rate', dest='learning_rate', type=float,
                        default=0.01, help='Learning rate to use during training.')

    parser.add_argument('--input-file', dest='input_file', type=str,
                        default='data/preprocessed/protein_net_testfile.txt.hdf5')

    args, _unknown = parser.parse_known_args()

    # pre-process data
    process_raw_data(use_gpu, force_pre_processing_overwrite=False)

    # run experiment
    training_file = args.input_file
    validation_file = args.input_file

    model = MyModel(21, use_gpu=use_gpu)  # embed size = 21

    train_loader = contruct_dataloader_from_disk(training_file, args.minibatch_size)
    validation_loader = contruct_dataloader_from_disk(validation_file, args.minibatch_size)

    train_model_path = train_model(data_set_identifier="TRAIN",
                                   model=model,
                                   train_loader=train_loader,
                                   validation_loader=validation_loader,
                                   learning_rate=args.learning_rate,
                                   minibatch_size=args.minibatch_size,
                                   eval_interval=args.eval_interval,
                                   hide_ui=args.hide_ui,
                                   use_gpu=use_gpu,
                                   minimum_updates=args.minimum_updates)

    print("Completed training, trained model stored at:")
    print(train_model_path)
    
class MyModel(openprotein.BaseModel):
    def __init__(self, embedding_size, use_gpu):
        super(MyModel, self).__init__(use_gpu, embedding_size)
        self.use_gpu = use_gpu
        self.number_angles = 3
        self.conv1= nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(5, embedding_size), padding=(0,0))
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 1))
        self.deconv = nn.ConvTranspose2d(in_channels=6, out_channels=1, kernel_size=(9, 1))
        self.top_hidden = nn.Linear(1, 100)
        self.input_to_angles = nn.Linear(100, self.number_angles)

    def _get_network_emissions(self, original_aa_string):
        batch_sizes = list([a.size() for a in original_aa_string])

        embedded_input = self.embed(original_aa_string)
        embedded_input = embedded_input.transpose(0,1)
        print('shape of input and batch_size: ', embedded_input.shape, len(batch_sizes))
        conv1_out = self.conv1(embedded_input.unsqueeze(1))
        conv2_out = self.conv2(conv1_out)
        deconv_out = self.deconv(conv2_out)
        print('conv1_out shape: ', conv1_out.shape)
        #hidden_out = self.top_hidden(conv_output)
        hidden_out = self.top_hidden(deconv_out)
        emissions_padded = self.input_to_angles(hidden_out)
        emissions_padded = (emissions_padded.squeeze(1)).transpose(0,1)
        print('emissions_padded shape: ', emissions_padded.shape)

        probabilities = torch.softmax(emissions_padded.transpose(0, 1), 2)

        output_angles = torch.matmul(probabilities, ANGLE_ARR).transpose(0, 1)

        return output_angles, [], batch_sizes
