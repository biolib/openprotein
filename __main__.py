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
parser.add_argument('--minibatch-size-validation', dest = 'minibatch_size_validation', type=int,
                    default=50, help='Size of each minibatch during evaluation.')
parser.add_argument('--learning-rate', dest = 'learning_rate', type=float,
                    default=0.01, help='Learning rate to use during training.')
parser.add_argument('--cv-partition', dest = 'cv_partition', type=int,
                    default=0, help='Run a particular cross validation rotation.')
parser.add_argument('--model-mode', dest = 'model_mode', type=int,
                    default=2, help='Which model to use.')
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

result_matrices = np.zeros((5,5), dtype=np.int64)

if args.model_mode == 0:
    model_mode = TMHMM3Mode.LSTM
elif args.model_mode == 1:
    model_mode = TMHMM3Mode.LSTM_CRF
elif args.model_mode == 2:
    model_mode = TMHMM3Mode.LSTM_CRF_HMM
elif args.model_mode == 3:
    model_mode = TMHMM3Mode.LSTM_CTC
else:
    print("ERROR: No model defined")

print("Using model:", model_mode)

embedding = "BLOSUM62"
use_marg_prob = False

for cv_partition in [0, 1, 2, 3, 4]:
    # prepare data sets
    train_set, val_set, test_set = load_data_from_disk(partition_rotation=cv_partition)

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
    pre_processed_path = "data/preprocessed/preprocessed_data_cv"+str(cv_partition)+".pickle"
    if not os.path.isfile(pre_processed_path):
        input_data_processed = list([TMDataset.from_disk(set, use_gpu) for set in [train_set, val_set, test_set, train_set_TOPOLOGY, val_set_TOPOLOGY, test_set_TOPOLOGY]])
        pickle.dump( input_data_processed, open(pre_processed_path, "wb"))
    input_data_processed = pickle.load(open(pre_processed_path, "rb"))
    train_preprocessed_set = input_data_processed[0]
    validation_preprocessed_set = input_data_processed[1]
    test_preprocessed_set = input_data_processed[2]
    train_preprocessed_set_TOPOLOGY = input_data_processed[3]
    validation_preprocessed_set_TOPOLOGY = input_data_processed[4]
    test_preprocessed_set_TOPOLOGY = input_data_processed[5]
    print("Completed preprocessing of data...")

    train_loader = tm_contruct_dataloader_from_disk(train_preprocessed_set, args.minibatch_size, balance_classes=True)
    validation_loader = tm_contruct_dataloader_from_disk(validation_preprocessed_set, args.minibatch_size_validation, balance_classes=True)
    test_loader = tm_contruct_dataloader_from_disk(validation_preprocessed_set, args.minibatch_size_validation) # TODO: replace this with test_preprocessed_set

    train_loader_TOPOLOGY = tm_contruct_dataloader_from_disk(train_preprocessed_set_TOPOLOGY, int(args.minibatch_size / 8)) # use smaller minibatch size for topology
    validation_loader_TOPOLOGY = tm_contruct_dataloader_from_disk(validation_preprocessed_set_TOPOLOGY, args.minibatch_size_validation)

    type_predictor_model_path = None

    for (experiment_id, train_data, validation_data) in [
        ("TRAIN_TYPE_CV"+str(cv_partition)+"-" + str(model_mode)+"-HS" + str(args.hidden_size), train_loader, validation_loader),
        ("TRAIN_TOPOLOGY_CV"+str(cv_partition)+"-" + str(model_mode)+"-HS" + str(args.hidden_size), train_loader_TOPOLOGY, validation_loader_TOPOLOGY)]:

        type_predictor = None
        if type_predictor_model_path is not None:
            type_predictor = load_model_from_disk(type_predictor_model_path, force_cpu=False)

        model = TMHMM3(
            embedding,
            args.hidden_size,
            use_gpu,
            model_mode,
            use_marg_prob,
            type_predictor)

        model_path = train_model(data_set_identifier=experiment_id,
                                 model=model,
                                 train_loader=train_data,
                                 validation_loader=validation_data,
                                 learning_rate=args.learning_rate,
                                 minibatch_size=args.minibatch_size,
                                 eval_interval=args.eval_interval,
                                 hide_ui=args.hide_ui,
                                 use_gpu=use_gpu,
                                 minimum_updates=args.minimum_updates)

        # let the GC collect the model
        del model

        write_out(model_path)

        # if we just trained a type predictor, save it for later
        if "TRAIN_TYPE" in experiment_id:
            type_predictor_model_path = model_path

    # test model
    if args.evaluate_on_test:
        write_out("Testing model of test set...")
        model = load_model_from_disk(model_path, force_cpu=False)
        loss, json_data, prediction_data = model.evaluate_model(test_loader)

        write_prediction_data_to_disk(model.post_process_prediction_data(prediction_data))
        result_matrix = json_data['confusion_matrix']
        result_matrices += result_matrix
        write_out(result_matrix)

set_experiment_id("TEST-" + str(model_mode)+"-HS" + str(args.hidden_size), args.learning_rate, args.minibatch_size)
write_out(result_matrices)