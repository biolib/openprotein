# This file is part of the OpenProtein project.
#
# @author Jeppe Hallgren
#
# For license information, please see the LICENSE file in the root directory.

from preprocessing import process_raw_data
import torch
import torch.utils.data
import h5py
import torch.autograd as autograd
import torch.optim as optim
import argparse
import numpy as np
import time
import requests
pymol_argv = ['pymol', '-i']
from pymol import cmd
from models import ExampleModel
from util import contruct_dataloader_from_disk, set_experiment_id, write_out, \
    evaluate_model, write_model_to_disk, draw_plot, write_result_summary, write_to_pdb

print("------------------------")
print("--- OpenProtein v0.1 ---")
print("------------------------")

parser = argparse.ArgumentParser(description = "OpenProtein version 0.1")
parser.add_argument('--silent', dest='silent', action='store_true',
                    help='Dont print verbose debug statements.')
parser.add_argument('--live-plot', dest = 'live_plot', action = 'store_true',
                    default=False, help='Draw loss graph while training goes on.')
parser.add_argument('--evaluate-on-test', dest = 'evaluate_on_test', action = 'store_true',
                    default=False, help='Run model of test data.')
parser.add_argument('--eval-interval', dest = 'eval_interval', type=int,
                    default=10, help='Evaluate model on validation set every n minibatches.')
parser.add_argument('--min-updates', dest = 'minimum_updates', type=int,
                    default=1000, help='Minimum number of minibatch iterations.')
parser.add_argument('--minibatch-size', dest = 'minibatch_size', type=int,
                    default=1, help='Size of each minibatch.')
parser.add_argument('--learning-rate', dest = 'learning_rate', type=float,
                    default=0.001, help='Learning rate to use during training.')
args, unknown = parser.parse_known_args()

if not args.live_plot:
    write_out("Live plot deactivated, see output folder for plot.")

use_gpu = False
if torch.cuda.is_available():
    write_out("CUDA is available, using GPU")
    use_gpu = True

process_raw_data(force_pre_processing_overwrite=False)

training_file = "data/preprocessed/testing.txt.hdf5"
validation_file = "data/preprocessed/testing.txt.hdf5"
testing_file = "data/preprocessed/testing.hdf5"

def train_model(data_set_identifier, train_file, val_file, learning_rate, minibatch_size):
    set_experiment_id(data_set_identifier, learning_rate, minibatch_size)

    train_loader = contruct_dataloader_from_disk(train_file, minibatch_size)
    validation_loader = contruct_dataloader_from_disk(val_file, minibatch_size)
    validation_dataset_size = validation_loader.dataset.__len__()

    model = ExampleModel(9, "ONEHOT", minibatch_size, use_gpu=use_gpu) # 3 x 3 coordinates for each aa

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    sample_num = list()
    train_loss_values = list()
    validation_loss_values = list()

    best_model_loss = 1.1
    best_model_minibatch_time = None
    best_model_path = None
    stopping_condition_met = False
    minibatches_proccesed = 0

    while not stopping_condition_met:
        optimizer.zero_grad()
        model.zero_grad()
        loss_tracker = np.zeros(0)
        for minibatch_id, training_minibatch in enumerate(train_loader, 0):
            minibatches_proccesed += 1
            primary_sequence, tertiary_positions, mask = training_minibatch
            start_compute_loss = time.time()
            loss = model.neg_log_likelihood(primary_sequence, tertiary_positions)
            write_out("Train loss:", float(loss))
            start_compute_grad = time.time()
            loss.backward()
            loss_tracker = np.append(loss_tracker, float(loss))
            end = time.time()
            write_out("Loss time:", start_compute_grad-start_compute_loss, "Grad time:", end-start_compute_grad)
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()

            # for every eval_interval samples, plot performance on the validation set
            if minibatches_proccesed % args.eval_interval == 0:

                train_loss = loss_tracker.mean()
                loss_tracker = np.zeros(0)
                validation_loss, data_total = evaluate_model(validation_loader, model)
                prim = data_total[0][0]
                pos = data_total[0][1].transpose(0,1).contiguous().view(-1,3)
                pos_predicted = data_total[0][2].transpose(0,1).contiguous().view(-1,3)
                write_to_pdb(pos, prim, "test")
                cmd.load("output/protein_test.pdb")
                write_to_pdb(pos_predicted.detach(), prim, "test_pred")
                if validation_loss < best_model_loss:
                    best_model_loss = validation_loss
                    best_model_minibatch_time = minibatches_proccesed
                    best_model_path = write_model_to_disk(model)

                write_out("Validation loss:", validation_loss, "Train loss:", train_loss)
                write_out("Best model so far (label loss): ", validation_loss, "at time", best_model_minibatch_time)
                write_out("Best model stored at " + best_model_path)
                write_out("Minibatches processed:",minibatches_proccesed)
                sample_num.append(minibatches_proccesed)
                train_loss_values.append(train_loss)
                validation_loss_values.append(validation_loss)
                if args.live_plot:
                    data = {}
                    data["validation_dataset_size"] = validation_dataset_size
                    data["sample_num"] = sample_num
                    data["train_loss_values"] = train_loss_values
                    data["validation_loss_values"] = validation_loss_values
                    res = requests.post('http://localhost:5000/graph', json=data)
                    if res.ok:
                        print(res.json())

                if minibatches_proccesed > args.minimum_updates and minibatches_proccesed > best_model_minibatch_time * 2:
                    stopping_condition_met = True
                    break
    write_result_summary(best_model_loss)
    return best_model_path


train_model_path = train_model("TRAIN", training_file, validation_file, args.learning_rate, args.minibatch_size)

print(train_model_path)
