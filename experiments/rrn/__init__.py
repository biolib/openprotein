"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""

from experiments.rrn.models import RrnModel
from experiments.example.models import *

from preprocessing import process_raw_data

from training import train_model
from util import contruct_dataloader_from_disk

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

    model = RrnModel(21, use_gpu=use_gpu)  # embed size = 21

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
