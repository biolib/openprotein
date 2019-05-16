# This file is part of the TMHMM3 project.
#
# @author Jeppe Hallgren
#
# For license information, please see the LICENSE file in the root directory.

import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from datetime import datetime
import math
import random

class TMDataset(Dataset):
    def __init__(self, aa_list, label_list, remapped_labels_list, type_list, topology_list, prot_name_list, original_aa_string_list, original_label_string):
        assert len(aa_list) == len(label_list)
        assert len(aa_list) == len(remapped_labels_list)
        assert len(aa_list) == len(type_list)
        assert len(aa_list) == len(topology_list)
        self.aa_list = aa_list
        self.label_list = label_list
        self.remapped_labels_list = remapped_labels_list
        self.type_list = type_list
        self.topology_list = topology_list
        self.prot_name_list = prot_name_list
        self.original_aa_string_list = original_aa_string_list
        self.original_label_string = original_label_string

    def __getitem__(self, index):
        return self.aa_list[index], \
               self.label_list[index], \
               self.remapped_labels_list[index], \
               self.type_list[index], \
               self.topology_list[index], \
               self.prot_name_list[index], \
               self.original_aa_string_list[index], \
               self.original_label_string[index]

    def __len__(self):
        return len(self.aa_list)

    def merge_samples_to_minibatch(samples):
        samples_list = []
        for s in samples:
            samples_list.append(s)
        # sort according to length of aa sequence
        samples_list.sort(key=lambda x: len(x[6]), reverse=True)
        aa_list, labels_list, remapped_labels_list, prot_type_list, prot_topology_list, prot_name, original_aa_string, original_label_string = zip(*samples_list)
        write_out(prot_type_list)
        return aa_list, labels_list, remapped_labels_list, prot_type_list, prot_topology_list, prot_name, original_aa_string, original_label_string

    def from_disk(dataset, use_gpu, re_map_labels=True):
        print("Constructing data set from disk...")
        aa_list = []
        labels_list = []
        remapped_labels_list = []
        prot_type_list = []
        prot_topology_list_all = []
        prot_aa_list_all = []
        prot_labels_list_all = []
        prot_name_list = []
        # sort according to length of aa sequence
        dataset.sort(key=lambda x: len(x[1]), reverse=True)
        for prot_name, prot_aa_list, prot_original_label_list, type_id, cluster_id in dataset:
            prot_name_list.append(prot_name)
            prot_aa_list_all.append(prot_aa_list)
            prot_labels_list_all.append(prot_original_label_list)
            aa_tmp_list_tensor = []
            labels = None
            remapped_labels = None
            last_non_membrane_position = None
            if prot_original_label_list is not None:
                labels = []
                for topology_label in prot_original_label_list:
                    if topology_label is "L":
                        topology_label = "I"
                    if topology_label is "I":
                        last_non_membrane_position = "I"
                        labels.append(3)
                    elif topology_label is "O":
                        last_non_membrane_position = "O"
                        labels.append(4)
                    elif topology_label is "S":
                        last_non_membrane_position = "S"
                        labels.append(2)
                    elif topology_label is "M":
                        if last_non_membrane_position is "I":
                            labels.append(0)
                        elif last_non_membrane_position is "O":
                            labels.append(1)
                        else:
                            print("Error: unexpected label found in last_non_membrane_position:", topology_label)
                    else:
                        print("Error: unexpected label found:", topology_label, "for protein", prot_name)
                labels = torch.LongTensor(labels)
                remapped_labels = []
                topology = label_list_to_topology(labels)
                # given topology, now calculate remapped labels
                for idx, (pos, l) in enumerate(topology):
                    if l == 0: # I -> O
                        membrane_length = topology[idx+1][0]-pos
                        mm_beginning = 4
                        for i in range(mm_beginning):
                            remapped_labels.append(5 + i)
                        for i in range(40-(membrane_length-mm_beginning), 40):
                            remapped_labels.append(5 + i)
                    elif l == 1: # O -> I
                        membrane_length = topology[idx + 1][0] - pos
                        mm_beginning = 4
                        for i in range(mm_beginning):
                            remapped_labels.append(45 + i)
                        for i in range(40 - (membrane_length - mm_beginning), 40):
                            remapped_labels.append(45 + i)
                    else:
                        if idx == (len(topology) - 1):
                            for i in range(len(labels)-pos):
                                remapped_labels.append(l)
                        else:
                            for i in range(topology[idx+1][0]-pos):
                                remapped_labels.append(l)
                remapped_labels = torch.LongTensor(remapped_labels)
                # check that protein was properly parsed
                assert remapped_labels.size() == labels.size()
            if use_gpu:
                if labels is not None:
                    labels = labels.cuda()
                if remapped_labels is not None:
                    remapped_labels = remapped_labels.cuda()
            aa_list.append(aa_tmp_list_tensor)
            labels_list.append(labels)
            remapped_labels_list.append(remapped_labels)
            prot_type_list.append(type_id)
            prot_topology_list_all.append(label_list_to_topology(labels))
        return TMDataset(aa_list, labels_list, remapped_labels_list, prot_type_list, prot_topology_list_all, prot_name_list, prot_aa_list_all, prot_labels_list_all)


def tm_contruct_dataloader_from_disk(tm_dataset, minibatch_size, balance_classes=False):
    if balance_classes:
        batch_sampler = RandomBatchClassBalancedSequentialSampler(tm_dataset, minibatch_size)
    else:
        batch_sampler = RandomBatchSequentialSampler(tm_dataset, minibatch_size)
    return torch.utils.data.DataLoader(tm_dataset,
                                       batch_sampler = batch_sampler,
                                       collate_fn = TMDataset.merge_samples_to_minibatch)


class RandomBatchClassBalancedSequentialSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, batch_size):
        self.sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        self.batch_size = batch_size
        self.dataset = dataset

    def sample_at_index(self, rows, offset, sample_num):
        assert sample_num < len(rows)
        sample_half = int(sample_num / 2)
        if offset - sample_half <= 0:
            # sample start has to be 0
            samples = rows[:sample_num]
        elif offset + sample_half + (sample_num % 2) > len(rows):
            # sample end has to be a end
            samples = rows[-(sample_num+1):-1]
        else:
            samples = rows[offset-sample_half:offset+sample_half+(sample_num % 2)]
        assert len(samples) == sample_num
        return samples

    def __iter__(self):
        data_class_map = {}
        data_class_map[0] = []
        data_class_map[1] = []
        data_class_map[2] = []
        data_class_map[3] = []

        for idx in self.sampler:
            data_class_map[self.dataset[idx][3]].append(idx)

        num_each_class = int(self.batch_size / 4)

        max_class_size = max([len(data_class_map[0]),len(data_class_map[1]),len(data_class_map[2]),len(data_class_map[3])])

        batch_num = int(max_class_size / num_each_class)
        if max_class_size % num_each_class != 0:
            batch_num += 1

        batch_relative_offset = (1.0 / float(batch_num)) / 2.0
        batches = []
        for i in range(batch_num):
            batch = []
            for class_id, data_rows in data_class_map.items():
                int_offset = int(batch_relative_offset * len(data_rows))
                batch.extend(self.sample_at_index(data_rows, int_offset, num_each_class))
            batch_relative_offset += 1.0 / float(batch_num)
            batches.append(batch)

        random.shuffle(batches)

        for batch in batches:
            write_out("Using minibatch from RandomBatchClassBalancedSequentialSampler")
            yield batch

    def __len__(self):
        length = 0
        for idx in self.sampler:
            length += 1
        return length

class RandomBatchSequentialSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, batch_size):
        self.sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        self.batch_size = batch_size

    def __iter__(self):
        data = []
        for idx in self.sampler:
            data.append(idx)

        batch_num = int(len(data) / self.batch_size)
        if len(data) % self.batch_size != 0:
            batch_num += 1

        batch_order = list(range(batch_num))
        random.shuffle(batch_order)

        batch = []
        for batch_id in batch_order:
            write_out("Accessing minibatch #" + str(batch_id))
            for i in range(self.batch_size):
                if i+(batch_id*self.batch_size) < len(data):
                    batch.append(data[i+(batch_id*self.batch_size)])
            yield batch
            batch = []

    def __len__(self):
        length = 0;
        for idx in self.sampler:
            length += 1
        return length

def label_list_to_topology(labels):
    top_list = []
    last_label = None
    for idx, label in enumerate(labels):
        if last_label is None or last_label != label:
            top_list.append((idx, label))
        last_label = label
    return top_list

def remapped_labels_to_orginal_labels(labels):
    for idx, pl in enumerate(labels):
        if pl >= 5 and pl < 45:
            labels[idx] = 0
        if pl >= 45:
            labels[idx] = 1
    return labels

def orginal_labels_to_fasta(label_list):
    sequence = ""
    for label in label_list:
        if label == 0:
            sequence = sequence + "M"
        if label == 1:
            sequence = sequence + "M"
        if label == 2:
            sequence = sequence + "S"
        if label == 3:
            sequence = sequence + "I"
        if label == 4:
            sequence = sequence + "O"
        if label == 5:
            sequence = sequence + "-"
    return sequence

def get_predicted_type_from_labels(labels):
    labels = list([int(i) for i in labels])
    if 0 in labels or 1 in labels:
        if labels[0] == 2:
            return 1
        else:
            return 0
    else:
        if labels[0] == 2:
            return 2
        else:
            return 3

def is_topologies_equal(topology_a, topology_b, minimum_seqment_overlap=5):
    if len(topology_a) != len(topology_b):
        return False
    for idx, (position_a, label_a) in enumerate(topology_a):
        if label_a != topology_b[idx][1]:
            return False
        if label_a == 0 or label_a == 1:
            overlap_segment_start = max(topology_a[idx][0],topology_b[idx][0])
            overlap_segment_end = min(topology_a[idx+1][0],topology_b[idx+1][0])
            if overlap_segment_end-overlap_segment_start < minimum_seqment_overlap:
                return False
    return True

def draw_plot(fig, plt, validation_dataset_size, sample_num, train_loss_values,
              zero_one_topology_loss_values, zero_one_loss_values,zero_one_type_loss_values, embedding):
    def draw_with_vars():
        ax = fig.gca()
        ax2 = ax.twinx()
        plt.grid(True)
        plt.title("Training progress with " + embedding + " embedding and marginal probabilities")
        train_loss_plot, = ax.plot(sample_num, train_loss_values)
        ax.set_ylabel('Negative log likelihood')
        ax.yaxis.labelpad = 0
        topology_loss_plot, = ax2.plot(sample_num, zero_one_topology_loss_values, color='black')
        loss_plot, = ax2.plot(sample_num, zero_one_loss_values, color='g')
        type_loss_plot, = ax2.plot(sample_num, zero_one_type_loss_values, color='y')
        ax2.set_ylabel('0-1 loss')
        ax2.set_ylim(bottom=0)
        plt.legend([train_loss_plot, loss_plot, type_loss_plot, topology_loss_plot],
                   ['Train loss on last batch', 'Validation 0-1 label loss',
                    'Validation 0-1 type loss', 'Validation 0-1 topology loss'])
        ax.set_xlabel('Minibatches processed (=network updates)', color='black')
    return draw_with_vars

def get_data_set_size_from_data_loader(data_loader):
    size = 0
    for minibatch_id, minibatch in enumerate(data_loader, 0):
        aa_list_sorted, labels_list, remapped_labels_list, prot_type_list, prot_name_list, original_aa_string_list = minibatch
        size += len(labels_list)
    return size

def parse_3line_format(lines):
    i = 0
    prot_list = []
    while(i < len(lines)):
        if lines[i].strip() is "":
            i += 1
            continue
        prot_name_comment = lines[i]
        type_string = None
        cluster_id = None
        if prot_name_comment.__contains__(">"):
            i += 1
            prot_name = prot_name_comment.split("|")[0].split(">")[1]
            type_string = prot_name_comment.split("|")[1]
            cluster_id = int(prot_name_comment.split("|")[2])
        else:
            # assume this is data
            prot_name = "> Unknown Protein Name"
        prot_aa_list = lines[i].upper()
        i += 1
        if len(prot_aa_list) > 2000:
            print("Discarding protein",prot_name,"as length larger than 2000:",len(prot_aa_list))
        else:
            if i < len(lines) and not lines[i].__contains__(">"):
                prot_topology_list = lines[i].upper()
                i += 1
                if prot_topology_list.__contains__("S"):
                    if prot_topology_list.__contains__("M"):
                        type_id = 1
                        assert type_string == "SP+TM"
                    else:
                        type_id = 2
                        assert type_string == "SP"
                else:
                    if prot_topology_list.__contains__("M"):
                        type_id = 0
                        assert type_string == "TM"
                    else:
                        type_id = 3
                        assert type_string == "GLOBULAR"
            else:
                type_id = None
                prot_topology_list = None
            prot_list.append((prot_name, prot_aa_list, prot_topology_list, type_id, cluster_id))

    return prot_list

def parse_datafile_from_disk(file):
    lines = list([line.strip() for line in open(file)])
    return parse_3line_format(lines)


def calculate_partitions(n_partitions, cluster_partitions, types):
    partition_distribution = np.ones((n_partitions,len(np.unique(types))))
    partition_assignments = np.zeros(cluster_partitions.shape[0])

    for i in np.unique(cluster_partitions):
        cluster_positions = np.where(cluster_partitions == i)
        cluster_types = types[cluster_positions]
        unique_types_in_cluster, type_count = np.unique(cluster_types, return_counts=True)
        unique_types_in_cluster = unique_types_in_cluster.astype(np.int32)
        tmp_distribution = np.copy(partition_distribution)
        tmp_distribution[:,unique_types_in_cluster] += type_count
        relative_distribution = partition_distribution/tmp_distribution
        min_relative_distribution_group = np.argmin(np.sum(relative_distribution,axis=1))
        partition_distribution[min_relative_distribution_group,unique_types_in_cluster] += type_count
        partition_assignments[cluster_positions] = min_relative_distribution_group

    write_out("Loaded data into the following partitions")
    write_out("[[  TM  SP+TM  SP Glob]")
    write_out(partition_distribution.astype(np.int32)-np.ones(partition_distribution.shape).astype(np.int32))
    return partition_assignments.astype(np.int32)

def load_data_from_disk(partition_rotation=0):
    print("Loading data from disk...")
    data = parse_datafile_from_disk('data/raw/TMHMM3.train.3line.clstr20')
    data_unzipped = list(zip(*data))
    n_partitions = 5
    partitions = calculate_partitions(cluster_partitions=np.array(data_unzipped[4]), types=np.array(data_unzipped[3]), n_partitions=n_partitions)
    train_set = []
    val_set = []
    test_set = []
    for idx, sample in enumerate(data):
        partition = int(partitions[idx]) # in range 0-4
        rotated = (partition + partition_rotation) % 5
        if int(rotated) <= 2:
            train_set.append(sample)
        elif int(rotated) == 3:
            val_set.append(sample)
        else:
            test_set.append(sample)

    print("Data splited as:",
          len(train_set), "train set",
          len(val_set), "validation set",
          len(test_set), "test set")
    return train_set, val_set, test_set

def set_experiment_id(data_set_identifier, use_hmm_model, learning_rate, minibatch_size,hidden_size):
    output_string = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    output_string += "-" + data_set_identifier
    output_string += "-LR" + str(learning_rate).replace(".","_")
    output_string += "-MB" + str(minibatch_size)
    output_string += "-HS" + str(hidden_size)
    if use_hmm_model:
        output_string += "-HMM"
    globals().__setitem__("experiment_id",output_string)

def write_out(*args, end='\n'):
    output_string = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + str.join(" ", [str(a) for a in args]) + end
    if globals().get("experiment_id") is not None:
        with open("output/"+globals().get("experiment_id")+".txt", "a+") as output_file:
            output_file.write(output_string)
            output_file.flush()
    print(output_string, end="")

def write_result_summary(topology_accuracy):
    output_string = globals().get("experiment_id") + ": " + str(topology_accuracy) + "\n"
    with open("output/result_summary.txt", "a+") as output_file:
        output_file.write(output_string)
        output_file.flush()
    print(output_string, end="")

def write_model_to_disk(model):
    path = "output/models/"+globals().get("experiment_id")+".model"
    torch.save(model,path)
    return path

def load_model_from_disk(path, force_cpu=True):
    if force_cpu:
        # load model with map_location set to storage (main mem)
        model = torch.load(path, map_location=lambda storage, loc: storage)
        # flattern parameters in memory
        model.flatten_parameters()
        # update internal TMHMM3 state accordingly
        model.use_gpu = False
    else:
        # load model using default map_location
        model = torch.load(path)
        model.flatten_parameters()
    return model

def evaluate_model(data_set_identifier, data_loader, data_set_size, type_predictor_model, tm_model, sptm_model):
    assert data_set_size == get_data_set_size_from_data_loader(data_loader)
    label_loss = 0
    type_error = 0
    type_count = np.zeros(4)
    topology_error = np.zeros(4)
    confusion_matrix = np.zeros((4, 4))
    validation_loss_tracker = []
    data_total = []
    for i, data in enumerate(data_loader, 0):
        _, labels_list, remapped_labels_list, prot_type_list, prot_name_list, original_aa_string = data
        use_hmm_model = True
        labels_to_use = remapped_labels_list if use_hmm_model else labels_list
        actual_labels = torch.nn.utils.rnn.pad_sequence([l for l in labels_to_use])
        validation_loss_tracker.append( tm_model.neg_log_likelihood(original_aa_string, actual_labels).detach())
        write_out("Starting viterbi decode...")
        predicted_tm = []
        predicted_sptm = []
        if type_predictor_model:
            predicted_labels_for_type, type_predictor_types = type_predictor_model(original_aa_string)
            for idx, predicted_type in enumerate(type_predictor_types):
                if predicted_type == 0:
                    predicted_tm.append(idx)
                elif predicted_type == 1:
                    predicted_sptm.append(idx)
        else:
            predicted_labels_for_type = None
            type_predictor_types = None

        predicted_types = type_predictor_types
        if tm_model is not None and sptm_model is not None:
            tm_predicted_labels_list, tm_predicted_types = \
                tm_model(original_aa_string, type_predictor_types if type_predictor_model else None)
            sptm_predicted_labels_list, sptm_predicted_types = \
                sptm_model(original_aa_string, type_predictor_types if type_predictor_model else None)

            predicted_labels_list = predicted_labels_for_type
            for idx in predicted_tm:
                predicted_labels_list[idx] = tm_predicted_labels_list[idx]
            for idx in predicted_sptm:
                predicted_labels_list[idx] = sptm_predicted_labels_list[idx]
        else:
            # train just on the first given model (tm_model)
            predicted_labels_list, predicted_types = \
                tm_model(original_aa_string, type_predictor_types if type_predictor_model else None)

        write_out("Completed viterbi decode")
        minibatch_data = list(zip(predicted_labels_list,
                                  #predicted_labels_for_type if type_predictor_model else predicted_labels_list,
                                  predicted_types,
                                  list(map(lambda x: list(x) if x is not None else None,labels_list)),
                                  list(map(lambda x: int(x) if x is not None else None,prot_type_list)),
                                  prot_name_list,
                                  original_aa_string))
        data_total.extend(minibatch_data)
        for predicted_labels, predicted_type, true_labels, actual_type, prot_name_list, original_aa_string in \
                minibatch_data:
            error = 0
            predicted_topology = label_list_to_topology(predicted_labels)
            if actual_type is not None:
                type_count[actual_type] += 1
            if true_labels is not None:
                for idx, label in enumerate(predicted_labels):
                    if true_labels[idx] != label:
                        error += 1
                true_topology = label_list_to_topology(true_labels)

                if not is_topologies_equal(true_topology, predicted_topology, 5):
                    topology_error[actual_type] += 1
                    #write_out(list([int(a) for a in true_labels]))
                    #write_out(list([int(a) for a in predicted_labels]))
                    #write_out(predicted_labels_type)
                    write_out(list([(a,int(b)) for (a,b) in true_topology]))
                    write_out(list([(a,int(b)) for (a,b) in predicted_topology]))
                    if predicted_type != actual_type:
                        type_error += 1
                        write_out("Predicted type:", predicted_type, "Actual type:", actual_type)
                else:
                    if predicted_type != actual_type:
                        type_error += 1
                        topology_error[actual_type] += 1
                        write_out("Predicted type:", predicted_type, "Actual type:", actual_type)
                confusion_matrix[actual_type][predicted_type] += 1.0
                label_loss += error / len(predicted_labels)
    label_loss /= data_set_size
    type_loss = type_error / data_set_size
    topology_loss = topology_error.sum() / data_set_size
    loss = torch.stack(validation_loss_tracker).mean()
    for i in range(4):
        sum = int(confusion_matrix[i].sum())
        for k in range(4):
            if sum != 0:
                confusion_matrix[i][k] /= sum
            else:
                confusion_matrix[i][k] = math.nan
    return (loss, label_loss, type_loss, type_count, topology_loss, topology_error, confusion_matrix, data_total)

def fast_eval_model(data_loader, data_set_size, model):
    assert data_set_size == get_data_set_size_from_data_loader(data_loader)
    validation_loss_tracker = []
    for i, data in enumerate(data_loader, 0):
        _, labels_list, remapped_labels_list, prot_type_list, prot_name_list, original_aa_string = data
        use_hmm_model = True
        labels_to_use = remapped_labels_list if use_hmm_model else labels_list
        actual_labels = torch.nn.utils.rnn.pad_sequence([l for l in labels_to_use])
        validation_loss_tracker.append( model.neg_log_likelihood(original_aa_string, actual_labels).detach())
    loss = torch.stack(validation_loss_tracker).mean()
    return loss
