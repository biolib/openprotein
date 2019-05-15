# This file is part of the OpenProtein project.
#
# @author Jeppe Hallgren
#
# For license information, please see the LICENSE file in the root directory.

from util import *
import torch.optim as optim
import requests

def train_model(data_set_identifier, model, train_loader, validation_loader, learning_rate, minibatch_size=64, eval_interval=50, hide_ui=False, use_gpu=False, minimum_updates=1000):
    set_experiment_id(data_set_identifier, learning_rate, minibatch_size)

    validation_dataset_size = validation_loader.dataset.__len__()

    if use_gpu:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    sample_num = list()
    train_loss_values = list()
    validation_loss_values = list()

    best_model_loss = 1e20
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
            start_compute_loss = time.time()
            loss = model.compute_loss(training_minibatch)
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
            if minibatches_proccesed % eval_interval == 0:

                write_out("Testing model on validation set...")

                train_loss = loss_tracker.mean()
                loss_tracker = np.zeros(0)
                validation_loss, json_data, prediction_data = model.evaluate_model(validation_loader)

                if validation_loss < best_model_loss:
                    best_model_loss = validation_loss
                    best_model_minibatch_time = minibatches_proccesed
                    best_model_path = write_model_and_data_to_disk(model, model.post_process_prediction_data(prediction_data))

                write_out("Validation loss:", validation_loss, "Train loss:", train_loss)
                write_out("Best model so far (validation loss): ", best_model_loss, "at time", best_model_minibatch_time)
                write_out("Best model stored at " + best_model_path)
                write_out("Minibatches processed:",minibatches_proccesed)
                sample_num.append(minibatches_proccesed)
                train_loss_values.append(float(train_loss))
                validation_loss_values.append(float(validation_loss))

                if not hide_ui:
                    json_data["validation_dataset_size"] = validation_dataset_size
                    json_data["sample_num"] = sample_num
                    json_data["train_loss_values"] = train_loss_values
                    json_data["validation_loss_values"] = validation_loss_values
                    res = requests.post('http://localhost:5000/graph', json=json_data)
                    if res.ok:
                        print(res.json())

                if minibatches_proccesed > minimum_updates and minibatches_proccesed > best_model_minibatch_time * 2:
                    stopping_condition_met = True
                    break
    write_result_summary(best_model_loss)
    return best_model_path