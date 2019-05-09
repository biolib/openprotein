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
                    default=50, help='Size of each minibatch.')
parser.add_argument('--minibatch-size-validation', dest = 'minibatch_size_validation', type=int,
                    default=50, help='Size of each minibatch during evaluation.')
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

#process_raw_data(use_gpu, force_pre_processing_overwrite=False)

#training_file = "data/preprocessed/sample.txt.hdf5"
#validation_file = "data/preprocessed/sample.txt.hdf5"
#testing_file = "data/preprocessed/testing.hdf5"
#train_loader = contruct_dataloader_from_disk(training_file, args.minibatch_size)
#validation_loader = contruct_dataloader_from_disk(validation_file, args.minibatch_size)

# prepare data sets
train_set, val_set, test_set = load_data_from_disk()

# topology data set
train_set_TOPOLOGY = list(filter(lambda x: x[3] is 0 or x[3] is 1, train_set))
val_set_TOPOLOGY = list(filter(lambda x: x[3] is 0 or x[3] is 1, val_set))
test_set_TOPOLOGY = list(filter(lambda x: x[3] is 0 or x[3] is 1, test_set))

if not args.silent:
    print("Loaded ",
          len(train_set),"training,",
          len(val_set),"validation and",
          len(test_set),"test samples")

print("Processing data...")
if not os.path.isfile('data/preprocessed/preprocessed_data.pickle'):
    input_data_processed = list([TMDataset.from_disk(set, use_gpu) for set in [train_set, val_set, test_set, train_set_TOPOLOGY, val_set_TOPOLOGY, test_set_TOPOLOGY]])
    pickle.dump( input_data_processed, open("data/preprocessed/preprocessed_data.pickle", "wb"))
input_data_processed = pickle.load(open("data/preprocessed/preprocessed_data.pickle", "rb"))
train_preprocessed_set = input_data_processed[0]
validation_preprocessed_set = input_data_processed[1]
test_preprocessed_set = input_data_processed[2]
train_preprocessed_set_TOPOLOGY = input_data_processed[3]
validation_preprocessed_set_TOPOLOGY = input_data_processed[4]
test_preprocessed_set_TOPOLOGY = input_data_processed[5]
print("Completed preprocessing of data...")

train_loader = tm_contruct_dataloader_from_disk(train_preprocessed_set, args.minibatch_size, balance_classes=True)
validation_loader = tm_contruct_dataloader_from_disk(validation_preprocessed_set, args.minibatch_size_validation)

model_mode = TMHMM3Mode.LSTM_CTC

hidden_size = 128
embedding = "BLOSUM62"
use_marg_prob = False

if model_mode == TMHMM3Mode.LSTM_CRF_HMM:
    allowed_transitions = [
        (2, 2), (3, 3), (4, 4),
        (3, 5), (4, 45),
        (2, 5), (2, 45), (2, 3), (2, 4)]
    for i in range(5, 45 - 1):
        allowed_transitions.append((i, i + 1))
        if i > 8 and i < 43:
            allowed_transitions.append((8, i))
    allowed_transitions.append((44, 4))
    for i in range(45, 85 - 1):
        allowed_transitions.append((i, i + 1))
        if i > 48 and i < 83:
            allowed_transitions.append((48, i))
    allowed_transitions.append((84, 3))
else:
    allowed_transitions = [
        (0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
        (3, 0), (0, 4), (4, 1), (1, 3),
        (2, 0), (2, 1), (2, 3), (2, 4)]

#hmm_state_graph = Digraph(comment='HMM States')
#for (a, b) in allowed_transitions:
#    hmm_state_graph.edge(str(a), str(b))
#hmm_state_graph.render('output/hmm-states.gv', view=False)

model = TMHMM3(
    embedding,
    hidden_size,
    use_gpu,
    model_mode,
    use_marg_prob,
    allowed_transitions)

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
