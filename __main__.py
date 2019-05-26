# This file is part of the OpenProtein project.
#
# @author Jeppe Hallgren
#
# For license information, please see the LICENSE file in the root directory.
from graphviz import Digraph

from preprocessing import process_raw_data
import pickle
import argparse
from dashboard import start_dashboard_server

from models import *
import os
from tm_models import *
from tm_util import *
from util import write_out, set_experiment_id
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
                    default=25, help='Evaluate model on validation set every n minibatches.')
parser.add_argument('--min-updates', dest = 'minimum_updates', type=int,
                    default=500, help='Minimum number of minibatch iterations.')
parser.add_argument('--hidden-size', dest = 'hidden_size', type=int,
                    default=64, help='Hidden size.')
parser.add_argument('--minibatch-size', dest = 'minibatch_size', type=int,
                    default=50, help='Size of each minibatch.')
parser.add_argument('--learning-rate', dest = 'learning_rate', type=float,
                    default=0.01, help='Learning rate to use during training.')
args, unknown = parser.parse_known_args()

if args.hide_ui:
    write_out("Live plot deactivated, see output folder for plot.")

use_gpu = False
if torch.cuda.is_available():
    write_out("CUDA is available, using GPU")
    use_gpu = True

if not args.hide_ui:
    # start web server
    start_dashboard_server()

