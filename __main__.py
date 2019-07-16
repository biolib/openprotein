# This file is part of the OpenProtein project.
#
# @author Jeppe Hallgren
#
# For license information, please see the LICENSE file in the root directory.

import argparse
import importlib
from dashboard import start_dashboard_server

from util import *

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
                    default=50, help='Evaluate model on validation set every n minibatches.')
parser.add_argument('--min-updates', dest = 'minimum_updates', type=int,
                    default=200, help='Minimum number of minibatch iterations.')
parser.add_argument('--minibatch-size', dest = 'minibatch_size', type=int,
                    default=10, help='Size of each minibatch.')
parser.add_argument('--experiment-id', dest = 'experiment_id', type=str,
                    default="example", help='Which experiment to run.')
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

experiment = importlib.import_module("experiments." + args.experiment_id)
experiment.run_experiment(parser,use_gpu)
