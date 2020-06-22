"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""

import math
import random
from typing import List

import torch
from torch.utils.data.dataset import Dataset
import numpy as np

from pytorchcrf.torchcrf import CRF
from util import write_out


class TMDataset(Dataset):
    def __init__(self,
                 aa_list,
                 label_list,
                 remapped_labels_list_crf_hmm,
                 remapped_labels_list_crf_marg,
                 type_list,
                 topology_list,
                 prot_name_list,
                 original_aa_string_list,
                 original_label_string):
        assert len(aa_list) == len(label_list)
        assert len(aa_list) == len(type_list)
        assert len(aa_list) == len(topology_list)
        self.aa_list = aa_list
        self.label_list = label_list
        self.remapped_labels_list_crf_hmm = remapped_labels_list_crf_hmm
        self.remapped_labels_list_crf_marg = remapped_labels_list_crf_marg
        self.type_list = type_list
        self.topology_list = topology_list
        self.prot_name_list = prot_name_list
        self.original_aa_string_list = original_aa_string_list
        self.original_label_string = original_label_string

    @staticmethod
    def from_disk(dataset, use_gpu):
        print("Constructing data set from disk...")
        aa_list = []
        labels_list = []
        remapped_labels_list_crf_hmm = []
        remapped_labels_list_crf_marg = []
        prot_type_list = []
        prot_topology_list_all = []
        prot_aa_list_all = []
        prot_labels_list_all = []
        prot_name_list = []
        # sort according to length of aa sequence
        dataset.sort(key=lambda x: len(x[1]), reverse=True)
        for prot_name, prot_aa_list, prot_original_label_list, type_id, _cluster_id in dataset:
            prot_name_list.append(prot_name)
            prot_aa_list_all.append(prot_aa_list)
            prot_labels_list_all.append(prot_original_label_list)
            aa_tmp_list_tensor = []
            labels = None
            remapped_labels_crf_hmm = None
            last_non_membrane_position = None
            if prot_original_label_list is not None:
                labels = []
                for topology_label in prot_original_label_list:
                    if topology_label == "L":
                        topology_label = "I"
                    if topology_label == "I":
                        last_non_membrane_position = "I"
                        labels.append(3)
                    elif topology_label == "O":
                        last_non_membrane_position = "O"
                        labels.append(4)
                    elif topology_label == "S":
                        last_non_membrane_position = "S"
                        labels.append(2)
                    elif topology_label == "M":
                        if last_non_membrane_position == "I":
                            labels.append(0)
                        elif last_non_membrane_position == "O":
                            labels.append(1)
                        else:
                            print("Error: unexpected label found in last_non_membrane_position:",
                                  topology_label)
                    else:
                        print("Error: unexpected label found:", topology_label, "for protein",
                              prot_name)
                labels = torch.LongTensor(labels)
                remapped_labels_crf_hmm = []
                topology = label_list_to_topology(labels)
                # given topology, now calculate remapped labels
                for idx, (pos, l) in enumerate(topology):
                    if l == 0:  # I -> O
                        membrane_length = topology[idx + 1][0] - pos
                        mm_beginning = 4
                        for i in range(mm_beginning):
                            remapped_labels_crf_hmm.append(5 + i)
                        for i in range(40 - (membrane_length - mm_beginning), 40):
                            remapped_labels_crf_hmm.append(5 + i)
                    elif l == 1:  # O -> I
                        membrane_length = topology[idx + 1][0] - pos
                        mm_beginning = 4
                        for i in range(mm_beginning):
                            remapped_labels_crf_hmm.append(45 + i)
                        for i in range(40 - (membrane_length - mm_beginning), 40):
                            remapped_labels_crf_hmm.append(45 + i)
                    elif l == 2:  # S
                        signal_length = topology[idx + 1][0] - pos
                        remapped_labels_crf_hmm.append(2)
                        for i in range(signal_length - 1):
                            remapped_labels_crf_hmm.append(152 - ((signal_length - 1) - i))
                            if remapped_labels_crf_hmm[-1] == 85:
                                print("Too long signal peptide region found", prot_name)
                    else:
                        if idx == (len(topology) - 1):
                            for i in range(len(labels) - pos):
                                remapped_labels_crf_hmm.append(l)
                        else:
                            for i in range(topology[idx + 1][0] - pos):
                                remapped_labels_crf_hmm.append(l)
                remapped_labels_crf_hmm = torch.LongTensor(remapped_labels_crf_hmm)

                remapped_labels_crf_marg = list([l + (type_id * 5) for l in labels])
                remapped_labels_crf_marg = torch.LongTensor(remapped_labels_crf_marg)

                # check that protein was properly parsed
                assert remapped_labels_crf_hmm.size() == labels.size()
                assert remapped_labels_crf_marg.size() == labels.size()

            if use_gpu:
                if labels is not None:
                    labels = labels.cuda()
                remapped_labels_crf_hmm = remapped_labels_crf_hmm.cuda()
                remapped_labels_crf_marg = remapped_labels_crf_marg.cuda()
            aa_list.append(aa_tmp_list_tensor)
            labels_list.append(labels)
            remapped_labels_list_crf_hmm.append(remapped_labels_crf_hmm)
            remapped_labels_list_crf_marg.append(remapped_labels_crf_marg)
            prot_type_list.append(type_id)
            prot_topology_list_all.append(label_list_to_topology(labels))
        return TMDataset(aa_list, labels_list, remapped_labels_list_crf_hmm,
                         remapped_labels_list_crf_marg,
                         prot_type_list, prot_topology_list_all, prot_name_list,
                         prot_aa_list_all, prot_labels_list_all)

    def __getitem__(self, index):
        return self.aa_list[index], \
               self.label_list[index], \
               self.remapped_labels_list_crf_hmm[index], \
               self.remapped_labels_list_crf_marg[index], \
               self.type_list[index], \
               self.topology_list[index], \
               self.prot_name_list[index], \
               self.original_aa_string_list[index], \
               self.original_label_string[index]

    def __len__(self):
        return len(self.aa_list)


def merge_samples_to_minibatch(samples):
    samples_list = []
    for sample in samples:
        samples_list.append(sample)
    # sort according to length of aa sequence
    samples_list.sort(key=lambda x: len(x[7]), reverse=True)
    aa_list, labels_list, remapped_labels_list_crf_hmm, \
    remapped_labels_list_crf_marg, prot_type_list, prot_topology_list, \
    prot_name, original_aa_string, original_label_string = zip(
        *samples_list)
    write_out(prot_type_list)
    return aa_list, labels_list, remapped_labels_list_crf_hmm, remapped_labels_list_crf_marg, \
           prot_type_list, prot_topology_list, prot_name, original_aa_string, original_label_string

def tm_contruct_dataloader_from_disk(tm_dataset, minibatch_size, balance_classes=False):
    if balance_classes:
        batch_sampler = RandomBatchClassBalancedSequentialSampler(tm_dataset, minibatch_size)
    else:
        batch_sampler = RandomBatchSequentialSampler(tm_dataset, minibatch_size)
    return torch.utils.data.DataLoader(tm_dataset,
                                       batch_sampler=batch_sampler,
                                       collate_fn=merge_samples_to_minibatch)


class RandomBatchClassBalancedSequentialSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, batch_size):
        self.sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        self.batch_size = batch_size
        self.dataset = dataset

    def __iter__(self):
        data_class_map = {}
        data_class_map[0] = []
        data_class_map[1] = []
        data_class_map[2] = []
        data_class_map[3] = []

        for idx in self.sampler:
            data_class_map[self.dataset[idx][4]].append(idx)

        num_each_class = int(self.batch_size / 4)

        max_class_size = max(
            [len(data_class_map[0]), len(data_class_map[1]),
             len(data_class_map[2]), len(data_class_map[3])])

        batch_num = int(max_class_size / num_each_class)
        if max_class_size % num_each_class != 0:
            batch_num += 1

        batch_relative_offset = (1.0 / float(batch_num)) / 2.0
        batches = []
        for _ in range(batch_num):
            batch = []
            for _class_id, data_rows in data_class_map.items():
                int_offset = int(batch_relative_offset * len(data_rows))
                batch.extend(sample_at_index(data_rows, int_offset, num_each_class))
            batch_relative_offset += 1.0 / float(batch_num)
            batches.append(batch)

        random.shuffle(batches)

        for batch in batches:
            write_out("Using minibatch from RandomBatchClassBalancedSequentialSampler")
            yield batch

    def __len__(self):
        length = 0
        for _ in self.sampler:
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
                if i + (batch_id * self.batch_size) < len(data):
                    batch.append(data[i + (batch_id * self.batch_size)])
            yield batch
            batch = []

    def __len__(self):
        length = 0
        for _ in self.sampler:
            length += 1
        return length


def sample_at_index(rows, offset, sample_num):
    assert sample_num < len(rows)
    sample_half = int(sample_num / 2)
    if offset - sample_half <= 0:
        # sample start has to be 0
        samples = rows[:sample_num]
    elif offset + sample_half + (sample_num % 2) > len(rows):
        # sample end has to be an end
        samples = rows[-(sample_num + 1):-1]
    else:
        samples = rows[offset - sample_half:offset + sample_half + (sample_num % 2)]
    assert len(samples) == sample_num
    return samples

def label_list_to_topology(labels):
    if isinstance(labels, np.ndarray):
        top_list = []
        last_label = None
        for idx, label in enumerate(labels):
            if last_label is None or last_label != label:
                top_list.append((idx, label))
            last_label = label
        return top_list

    if isinstance(labels, list):
        labels = torch.LongTensor(labels)

    if isinstance(labels, torch.LongTensor):
        zero_tensor = torch.LongTensor([0])
        if labels.is_cuda:
            zero_tensor = zero_tensor.cuda()

        unique, count = torch.unique_consecutive(labels, return_counts=True)
        top_list = [torch.cat((zero_tensor, labels[0]))]
        prev_count = 0
        i = 0
        for _ in unique.split(1):
            if i == 0:
                i += 1
                continue
            prev_count += count[i - 1]
            top_list.append(torch.cat((prev_count.view(1), unique[i].view(1))))
            i += 1
        return top_list



def remapped_labels_hmm_to_orginal_labels(labels):

    if isinstance(labels, np.ndarray):
        zeros = np.zeros(labels.shape, dtype=np.long)
        ones = np.ones(labels.shape, dtype=np.long)
        twos = np.ones(labels.shape, dtype=np.long) * 2


        labels = np.where((labels >= 5) & (labels < 45), zeros, labels)
        labels = np.where((labels >= 45) & (labels < 85), ones, labels)
        labels = np.where(labels >= 85, twos, labels)

        return labels

    if isinstance(labels, list):
        labels = torch.LongTensor(labels)

    if isinstance(labels, torch.LongTensor):

        torch_zeros = torch.zeros(labels.size(), dtype=torch.long)
        torch_ones = torch.ones(labels.size(), dtype=torch.long)
        torch_twos = torch.ones(labels.size(), dtype=torch.long) * 2

        if labels.is_cuda:
            labels = labels.cuda()
            torch_zeros = labels.cuda()
            torch_ones = labels.cuda()
            torch_twos = labels.cuda()

        labels = torch.where((labels >= 5) & (labels < 45), torch_zeros, labels)
        labels = torch.where((labels >= 45) & (labels < 85), torch_ones, labels)
        labels = torch.where(labels >= 85, torch_twos, labels)

        return labels

def batch_sizes_to_mask(batch_sizes: torch.Tensor) -> torch.Tensor:
    arange = torch.arange(batch_sizes[0], dtype=torch.int32)
    if batch_sizes.is_cuda:
        arange = arange.cuda()
    res = (arange.expand(batch_sizes.size(0), batch_sizes[0])
           < batch_sizes.unsqueeze(1)).transpose(0, 1)
    return res

def original_labels_to_fasta(label_list):
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
    if isinstance(labels, np.ndarray):
        zero = np.zeros(1, dtype=np.long)

        contains_0 = (labels == 0).sum() > 0
        contains_1 = (labels == 1).sum() > 0
        contains_2 = np.where((labels == 2).sum() > 0, zero + 1, zero)

        is_tm = np.where(contains_0 | contains_1, zero + 1, zero)

        return is_tm * contains_2 \
               + ((is_tm - 1) * (is_tm - 1)) * (3 - contains_2)

    if isinstance(labels, torch.LongTensor):
        torch_zero = torch.zeros(1)

        if labels.is_cuda:
            torch_zero = torch_zero.cuda()

        contains_0 = (labels == 0).int().sum() > 0
        contains_1 = (labels == 1).int().sum() > 0
        contains_2 = torch.where((labels == 2).int().sum() > 0, torch_zero + 1, torch_zero)

        is_tm = torch.where(contains_0 | contains_1, torch_zero + 1, torch_zero)

        return is_tm * contains_2 \
               + ((is_tm - 1) * (is_tm - 1)) * (3 - contains_2)




def is_topologies_equal(topology_a, topology_b, minimum_seqment_overlap=5):
    if len(topology_a) != len(topology_b):
        return False
    for idx, (_position_a, label_a) in enumerate(topology_a):
        if label_a != topology_b[idx][1]:
            return False
        if label_a in (0, 1):
            overlap_segment_start = max(topology_a[idx][0], topology_b[idx][0])
            overlap_segment_end = min(topology_a[idx + 1][0], topology_b[idx + 1][0])
            if overlap_segment_end - overlap_segment_start < minimum_seqment_overlap:
                return False
    return True


def parse_3line_format(lines):
    i = 0
    prot_list = []
    while i < len(lines):
        if lines[i].strip() == "":
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
        if len(prot_aa_list) > 6000:
            print("Discarding protein", prot_name, "as length larger than 6000:",
                  len(prot_aa_list))
            if i < len(lines) and not lines[i].__contains__(">"):
                i += 1
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
            prot_list.append((prot_name, prot_aa_list, prot_topology_list,
                              type_id, cluster_id))

    return prot_list


def parse_datafile_from_disk(file):
    lines = list([line.strip() for line in open(file)])
    return parse_3line_format(lines)


def calculate_partitions(partitions_count, cluster_partitions, types):
    partition_distribution = torch.ones((partitions_count,
                                         len(torch.unique(types))),
                                        dtype=torch.long)
    partition_assignments = torch.zeros(cluster_partitions.shape[0],
                                        dtype=torch.long)

    for i in torch.unique(cluster_partitions):
        cluster_positions = (cluster_partitions == i).nonzero()
        cluster_types = types[cluster_positions]
        unique_types_in_cluster, type_count = torch.unique(cluster_types, return_counts=True)
        tmp_distribution = partition_distribution.clone()
        tmp_distribution[:, unique_types_in_cluster] += type_count
        relative_distribution = partition_distribution.double() / tmp_distribution.double()
        min_relative_distribution_group = torch.argmin(torch.sum(relative_distribution, dim=1))
        partition_distribution[min_relative_distribution_group,
                               unique_types_in_cluster] += type_count
        partition_assignments[cluster_positions] = min_relative_distribution_group

    write_out("Loaded data into the following partitions")
    write_out("[[  TM  SP+TM  SP Glob]")
    write_out(partition_distribution - torch.ones(partition_distribution.shape,
                                                  dtype=torch.long))
    return partition_assignments


def load_data_from_disk(filename, partition_rotation=0):
    print("Loading data from disk...")
    data = parse_datafile_from_disk(filename)
    data_unzipped = list(zip(*data))
    partitions = calculate_partitions(
        cluster_partitions=torch.LongTensor(np.array(data_unzipped[4])),
        types=torch.LongTensor(np.array(data_unzipped[3])),
        partitions_count=5)
    train_set = []
    val_set = []
    test_set = []
    for idx, sample in enumerate(data):
        partition = int(partitions[idx])  # in range 0-4
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


def normalize_confusion_matrix(confusion_matrix):
    confusion_matrix = confusion_matrix.astype(np.float64)
    for i in range(4):
        accumulator = int(confusion_matrix[i].sum())
        if accumulator != 0:
            confusion_matrix[4][i] /= accumulator * 0.01  # 0.01 to convert to percentage
        for k in range(5):
            if accumulator != 0:
                confusion_matrix[i][k] /= accumulator * 0.01  # 0.01 to convert to percentage
            else:
                confusion_matrix[i][k] = math.nan
    return confusion_matrix.round(2)

def decode(emissions, batch_sizes, start_transitions, transitions, end_transitions):
    mask = batch_sizes_to_mask(batch_sizes)

    if emissions.is_cuda:
        mask = mask.cuda()
    crf_model = CRF(int(start_transitions.size(0)))
    initialize_crf_parameters(crf_model,
                              start_transitions=start_transitions,
                              transitions=transitions,
                              end_transitions=end_transitions)
    labels_predicted = []
    for l in crf_model.decode(emissions, mask=mask):
        val = torch.tensor(l).unsqueeze(1)
        if emissions.is_cuda:
            val = val.cuda()
        labels_predicted.append(val)


    predicted_labels = []
    for l in labels_predicted:
        predicted_labels.append(remapped_labels_hmm_to_orginal_labels(l))

    predicted_types_list = []
    for p_label in predicted_labels:
        predicted_types_list.append(get_predicted_type_from_labels(p_label))
    predicted_types = torch.cat(predicted_types_list)



    if emissions.is_cuda:
        predicted_types = predicted_types.cuda()

    # if all O's, change to all I's (by convention)
    torch_zero = torch.zeros(1, dtype=torch.long)
    if emissions.is_cuda:
        torch_zero = torch_zero.cuda()
    for idx, labels in enumerate(predicted_labels):
        predicted_labels[idx] = \
            labels - torch.where(torch.eq(labels, 4).min() == 1, torch_zero + 1, torch_zero)

    return predicted_labels, predicted_types, list(map(label_list_to_topology, predicted_labels))


def decode_numpy(emissions, batch_sizes, start_transitions, transitions, end_transitions):
    labels_predicted = []
    for l in numpy_viterbi_decode(emissions,
                                  batch_sizes=batch_sizes,
                                  start_transitions=start_transitions,
                                  transitions=transitions,
                                  end_transitions=end_transitions):
        val = np.expand_dims(np.array(l), 1)
        labels_predicted.append(val)


    predicted_labels = []
    for l in labels_predicted:
        predicted_labels.append(remapped_labels_hmm_to_orginal_labels(l))

    predicted_types_list = []
    for p_label in predicted_labels:
        predicted_types_list.append(get_predicted_type_from_labels(p_label))
    predicted_types = np.array(predicted_types_list).squeeze(axis=1)

    # if all O's, change to all I's (by convention)
    zero = np.zeros(1, dtype=np.long)

    for idx, labels in enumerate(predicted_labels):
        predicted_labels[idx] = \
            labels - np.where((labels == 4).min() == 1, zero + 1, zero)

    return predicted_labels, \
           predicted_types, \
           list(map(label_list_to_topology, predicted_labels))

def numpy_viterbi_decode(emissions,
                         batch_sizes,
                         start_transitions,
                         transitions,
                         end_transitions) -> List[List[int]]:
    # emissions: (seq_length, batch_size, num_tags)
    # mask: (seq_length, batch_size)
    assert len(emissions.shape) == 3
    #assert emissions.shape[:2] == mask.shape
    #assert emissions.size(2) == self.num_tags
    #assert mask[0].all()

    seq_length = emissions.shape[0]
    batch_size = emissions.shape[1]

    # Start transition and first emission
    # shape: (batch_size, num_tags)
    score = start_transitions + emissions[0]
    history = []

    # score is a tensor of size (batch_size, num_tags) where for every batch,
    # value at column j stores the score of the best tag sequence so far that ends
    # with tag j
    # history saves where the best tags candidate transitioned from; this is used
    # when we trace back the best tag sequence

    # Viterbi algorithm recursive case: we compute the score of the best tag sequence
    # for every possible next tag

    l = []
    for i in batch_sizes:
        l.append(np.array([1] * i + [0] * (seq_length - i)))
    mask = np.array(l).T


    for i in range(1, seq_length):

        # Broadcast viterbi score for every possible next tag
        # shape: (batch_size, num_tags, 1)
        broadcast_score = np.expand_dims(score, 2)

        # Broadcast emission score for every possible current tag
        # shape: (batch_size, 1, num_tags)
        broadcast_emission = np.expand_dims(emissions[i], 1)

        # Compute the score tensor of size (batch_size, num_tags, num_tags) where
        # for each sample, entry at row i and column j stores the score of the best
        # tag sequence so far that ends with transitioning from tag i to tag j and emitting
        # shape: (batch_size, num_tags, num_tags)
        next_score = broadcast_score + transitions + broadcast_emission

        # Find the maximum score over all possible current tag
        # shape: (batch_size, num_tags)
        indices = next_score.argmax(axis=1)
        next_score = next_score.max(axis=1)


        # Set score to the next score if this timestep is valid (mask == 1)
        # and save the index that produces the next score
        # shape: (batch_size, num_tags)
        score = np.where(np.expand_dims(mask[i], 1), next_score, score) # pylint: disable=E1136
        history.append(indices)


    # End transition score
    # shape: (batch_size, num_tags)
    score += end_transitions

    # Now, compute the best path for each sample

    # shape: (batch_size,)
    seq_ends = batch_sizes - 1
    best_tags_list = []

    for idx in range(batch_size):
        # Find the tag which maximizes the score at the last timestep; this is our best tag
        # for the last timestep
        best_last_tag = score[idx].argmax(axis=0)
        best_tags = [best_last_tag.item()]

        # We trace back where the best last tag comes from, append that to our best tag
        # sequence, and trace it back again, and so on
        for hist in reversed(history[:seq_ends[idx]]):
            best_last_tag = hist[idx][best_tags[-1]]
            best_tags.append(best_last_tag.item())

        # Reverse the order because we start from the last timestep
        best_tags.reverse()
        best_tags_list.append(best_tags)

    return best_tags_list

def initialize_crf_parameters(crf_model,
                              start_transitions=None,
                              end_transitions=None,
                              transitions=None) -> None:
    """Initialize the transition parameters.

    The parameters will be initialized randomly from a uniform distribution
    between -0.1 and 0.1, unless given explicitly as an argument.
    """
    if start_transitions is None:
        torch.nn.init.uniform(crf_model.start_transitions, -0.1, 0.1)
    else:
        crf_model.start_transitions.data = start_transitions
    if end_transitions is None:
        torch.nn.init.uniform(crf_model.end_transitions, -0.1, 0.1)
    else:
        crf_model.end_transitions.data = end_transitions
    if transitions is None:
        torch.nn.init.uniform(crf_model.transitions, -0.1, 0.1)
    else:
        crf_model.transitions.data = transitions
