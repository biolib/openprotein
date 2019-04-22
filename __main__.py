# This file is part of the OpenProtein project.
#
# @author Jeppe Hallgren
#
# For license information, please see the LICENSE file in the root directory.

from preprocessing import process_raw_data
import argparse
from dashboard import start_dashboard_server

from models import *
from util import write_out
from training import train_model

print("------------------------")
print("--- OpenProtein v0.1 ---")
print("------------------------")

parser = argparse.ArgumentParser(description = "OpenProtein version 0.1")
parser.add_argument('--silent', dest='silent', action='store_true',
                    help='Dont print verbose debug statements.')
parser.add_argument('--hide-ui', dest = 'hide_ui', action = 'store_true',
                    default=False, help='Hide loss graph and visualization UI while training goes on.')
parser.add_argument('--evaluate-on-test', dest = 'evaluate_on_test', action = 'store_true',
                    default=False, help='Run model of test data.')
parser.add_argument('--eval-interval', dest = 'eval_interval', type=int,
                    default=5, help='Evaluate model on validation set every n minibatches.')
parser.add_argument('--min-updates', dest = 'minimum_updates', type=int,
                    default=5000, help='Minimum number of minibatch iterations.')
parser.add_argument('--minibatch-size', dest = 'minibatch_size', type=int,
                    default=1, help='Size of each minibatch.')
parser.add_argument('--learning-rate', dest = 'learning_rate', type=float,
                    default=0.01, help='Learning rate to use during training.')
args, unknown = parser.parse_known_args()

if args.hide_ui:
    write_out("Live plot deactivated, see output folder for plot.")

use_gpu = False
if torch.cuda.is_available():
    write_out("CUDA is available, using GPU")
    use_gpu = True

# start web server
start_dashboard_server()

process_raw_data(use_gpu, force_pre_processing_overwrite=False)

training_file = "data/preprocessed/sample.txt.hdf5"
validation_file = "data/preprocessed/sample.txt.hdf5"
testing_file = "data/preprocessed/testing.hdf5"

model = ExampleModel(21, args.minibatch_size, use_gpu=use_gpu)  # embed size = 21

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

print(train_model_path)
