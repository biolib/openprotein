"""
This file is part of the OpenProtein project.

For license information, please see the LICENSE file in the root directory.
"""

import sys
from enum import Enum
import glob
import pickle
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import openprotein
from experiments.tmhmm3.tm_util import label_list_to_topology
from experiments.tmhmm3.tm_util import get_predicted_type_from_labels
from experiments.tmhmm3.tm_util import remapped_labels_hmm_to_orginal_labels
from experiments.tmhmm3.tm_util import is_topologies_equal
from experiments.tmhmm3.tm_util import original_labels_to_fasta
from pytorchcrf.torchcrf import CRF
from util import write_out, get_experiment_id

# seed random generator for reproducibility
torch.manual_seed(1)


class TMHMM3(openprotein.BaseModel):
    def __init__(self,
                 embedding,
                 hidden_size,
                 use_gpu,
                 model_mode,
                 use_marg_prob,
                 type_predictor_model,
                 profile_path):
        super(TMHMM3, self).__init__(embedding, use_gpu)

        # initialize model variables
        num_tags = 5
        num_labels = 5
        self.max_signal_length = 67
        if model_mode == TMHMM3Mode.LSTM_CRF_HMM:
            num_tags += 2 * 40 + self.max_signal_length
        elif model_mode == TMHMM3Mode.LSTM_CRF_MARG:
            num_tags = num_tags * 4  # 4 different types
            # num_labels = num_tags # 4 different types
        self.hidden_size = hidden_size
        self.use_gpu = use_gpu
        self.use_marg_prob = use_marg_prob
        self.model_mode = model_mode
        self.embedding = embedding
        self.profile_path = profile_path
        self.bi_lstm = nn.LSTM(self.get_embedding_size(),
                               self.hidden_size,
                               num_layers=1,
                               bidirectional=True)
        self.hidden_to_labels = nn.Linear(self.hidden_size * 2, num_labels)  # * 2 for bidirectional
        self.hidden_layer = None
        crf_start_mask = torch.ones(num_tags).byte()
        crf_end_mask = torch.ones(num_tags).byte()
        if model_mode == TMHMM3Mode.LSTM_CRF_HMM:
            allowed_transitions = [
                (3, 3), (4, 4),
                (3, 5), (4, 45)]
            for i in range(5, 45 - 1):
                allowed_transitions.append((i, i + 1))
                if 8 < i < 43:
                    allowed_transitions.append((8, i))
            allowed_transitions.append((44, 4))
            for i in range(45, 85 - 1):
                allowed_transitions.append((i, i + 1))
                if 48 < i < 83:
                    allowed_transitions.append((48, i))
            allowed_transitions.append((84, 3))
            for i in range(85, 151):
                allowed_transitions.append((i, i + 1))
                allowed_transitions.append((2, i))
            allowed_transitions.append((2, 151))
            allowed_transitions.append((2, 4))
            allowed_transitions.append((151, 4))

            crf_start_mask[2] = 0
            crf_start_mask[3] = 0
            crf_start_mask[4] = 0
            crf_end_mask[3] = 0
            crf_end_mask[4] = 0
        elif model_mode == TMHMM3Mode.LSTM_CRF_MARG:
            allowed_transitions = [
                (0, 0), (1, 1), (3, 3), (4, 4), (3, 0), (0, 4), (4, 1), (1, 3),
                (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (8, 5), (5, 9), (9, 6), (6, 8), (7, 9),
                (12, 12), (14, 14), (12, 14),
                (18, 18),
            ]
            crf_start_mask[3] = 0
            crf_start_mask[4] = 0
            crf_start_mask[7] = 0
            crf_start_mask[8] = 0
            crf_start_mask[9] = 0
            crf_start_mask[12] = 0
            crf_start_mask[18] = 0
            crf_end_mask[3] = 0
            crf_end_mask[4] = 0
            crf_end_mask[8] = 0
            crf_end_mask[9] = 0
            crf_end_mask[14] = 0
            crf_end_mask[18] = 0
        else:
            allowed_transitions = [
                (0, 0), (1, 1), (2, 2), (3, 3), (4, 4),
                (3, 0), (0, 4), (4, 1), (1, 3), (2, 4)]

            crf_start_mask[2] = 0
            crf_start_mask[3] = 0
            crf_start_mask[4] = 0
            crf_end_mask[3] = 0
            crf_end_mask[4] = 0
        self.allowed_transitions = allowed_transitions
        self.crf_model = CRF(num_tags)
        self.type_classifier = type_predictor_model
        self.type_tm_classier = None
        self.type_sp_classier = None
        crf_transitions_mask = torch.ones((num_tags, num_tags)).byte()

        self.label_01loss_values = []
        self.type_01loss_values = []
        self.topology_01loss_values = []

        # if on GPU, move state to GPU memory
        if self.use_gpu:
            self.crf_model = self.crf_model.cuda()
            self.bi_lstm = self.bi_lstm.cuda()
            self.hidden_to_labels = self.hidden_to_labels.cuda()
            crf_transitions_mask = crf_transitions_mask.cuda()
            crf_start_mask = crf_start_mask.cuda()
            crf_end_mask = crf_end_mask.cuda()

        # compute mask matrix from allow transitions list
        for i in range(num_tags):
            for k in range(num_tags):
                if (i, k) in self.allowed_transitions:
                    crf_transitions_mask[i][k] = 0

        # generate masked transition parameters
        crf_start_transitions, crf_end_transitions, crf_transitions = \
            generate_masked_crf_transitions(
                self.crf_model, (crf_start_mask, crf_transitions_mask, crf_end_mask)
            )

        # initialize CRF
        initialize_crf_parameters(self.crf_model,
                                  start_transitions=crf_start_transitions,
                                  end_transitions=crf_end_transitions,
                                  transitions=crf_transitions)

    def get_embedding_size(self):
        if self.embedding == "BLOSUM62":
            return 24  # bloom matrix has size 24
        elif self.embedding == "PROFILE":
            return 51  # protein profiles have size 51

    def flatten_parameters(self):
        self.bi_lstm.flatten_parameters()

    def encode_amino_acid(self, letter):
        if self.embedding == "BLOSUM62":
            # blosum encoding
            if not globals().get('blosum_encoder'):
                blosum = \
                    """4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0,-2,-1,0,-4
                    -1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3,-1,0,-1,-4
                    -2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3,3,0,-1,-4
                    -2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3,4,1,-1,-4
                    0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1,-3,-3,-2,-4
                    -1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2,0,3,-1,-4
                    -1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2,1,4,-1,-4
                    0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3,-1,-2,-1,-4
                    -2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3,0,0,-1,-4
                    -1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3,-3,-3,-1,-4
                    -1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1,-4,-3,-1,-4
                    -1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2,0,1,-1,-4
                    -1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1,-3,-1,-1,-4
                    -2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1,-3,-3,-1,-4
                    -1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2,-2,-1,-2,-4
                    1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2,0,0,0,-4
                    0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0,-1,-1,0,-4
                    -3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3,-4,-3,-2,-4
                    -2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1,-3,-2,-1,-4
                    0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4,-3,-2,-1,-4
                    -2,-1,3,4,-3,0,1,-1,0,-3,-4,0,-3,-3,-2,0,-1,-4,-3,-3,4,1,-1,-4
                    -1,0,0,1,-3,3,4,-2,0,-3,-3,1,-1,-3,-1,0,-1,-3,-2,-2,1,4,-1,-4
                    0,-1,-1,-1,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2,0,0,-2,-1,-1,-1,-1,-1,-4
                    -4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,1""" \
                        .replace('\n', ',')
                blosum_matrix = np.fromstring(blosum, sep=",").reshape(24, 24)
                blosum_key = "A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V,B,Z,X,U".split(",")
                key_map = {}
                for idx, value in enumerate(blosum_key):
                    key_map[value] = list([int(v) for v in blosum_matrix[idx].astype('int')])
                globals().__setitem__("blosum_encoder", key_map)
            return globals().get('blosum_encoder')[letter]
        elif self.embedding == "ONEHOT":
            # one hot encoding
            one_hot_key = "A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V,B,Z,X,U".split(",")
            arr = []
            for idx, k in enumerate(one_hot_key):
                if k == letter:
                    arr.append(1)
                else:
                    arr.append(0)
            return arr
        elif self.embedding == "PYTORCH":
            key_id = "A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V,B,Z,X,U".split(",")
            for idx, k in enumerate(key_id):
                if k == letter:
                    return idx

    def embed(self, prot_aa_list):
        embed_list = []
        for aa_list in prot_aa_list:
            if self.embedding == "PYTORCH":
                tensor = torch.LongTensor(tensor)
            elif self.embedding == "PROFILE":
                if not globals().get('profile_encoder'):
                    print("Load profiles...")
                    files = glob.glob(self.profile_path.strip("/") + "/*")
                    profile_dict = {}
                    for profile_file in files:
                        profile = pickle.load(open(profile_file, "rb")).popitem()[1]
                        profile_dict[profile["seq"]] = torch.from_numpy(profile["profile"]).float()
                    globals().__setitem__("profile_encoder", profile_dict)
                    print("Loaded profiles")
                tensor = globals().get('profile_encoder')[aa_list]
            else:
                tensor = list([self.encode_amino_acid(aa) for aa in aa_list])
                tensor = torch.FloatTensor(tensor)
            if self.use_gpu:
                tensor = tensor.cuda()
            embed_list.append(tensor)
        return embed_list

    def init_hidden(self, minibatch_size):
        # number of layers (* 2 since bidirectional), minibatch_size, hidden size
        initial_hidden_state = torch.zeros(1 * 2, minibatch_size, self.hidden_size)
        initial_cell_state = torch.zeros(1 * 2, minibatch_size, self.hidden_size)
        if self.use_gpu:
            initial_hidden_state = initial_hidden_state.cuda()
            initial_cell_state = initial_cell_state.cuda()
        self.hidden_layer = (autograd.Variable(initial_hidden_state),
                             autograd.Variable(initial_cell_state))

    def _get_network_emissions(self, input_sequences):
        batch_sizes = torch.LongTensor(list([i.size(0) for i in input_sequences]))
        pad_seq_embed = torch.nn.utils.rnn.pad_sequence(input_sequences)
        minibatch_size = len(input_sequences)
        self.init_hidden(minibatch_size)
        bi_lstm_out, self.hidden_layer = self.bi_lstm(pad_seq_embed, self.hidden_layer)
        emissions = self.hidden_to_labels(bi_lstm_out)
        if self.model_mode == TMHMM3Mode.LSTM_CRF_HMM:
            inout_select = torch.LongTensor([0])
            outin_select = torch.LongTensor([1])
            signal_select = torch.LongTensor([2])
            if self.use_gpu:
                inout_select = inout_select.cuda()
                outin_select = outin_select.cuda()
                signal_select = signal_select.cuda()
            inout = torch.index_select(emissions, 2, autograd.Variable(inout_select))
            outin = torch.index_select(emissions, 2, autograd.Variable(outin_select))
            signal = torch.index_select(emissions, 2, autograd.Variable(signal_select))
            emissions = torch.cat((emissions, inout.expand(-1, len(batch_sizes), 40),
                                   outin.expand(-1, len(batch_sizes), 40),
                                   signal.expand(-1, len(batch_sizes), self.max_signal_length)), 2)
        elif self.model_mode == TMHMM3Mode.LSTM_CRF_MARG:
            emissions = emissions.repeat(1, 1, 4)
        return emissions, batch_sizes

    def batch_sizes_to_mask(self, batch_sizes):
        mask = torch.autograd.Variable(torch.t(torch.ByteTensor(
            [[1] * int(batch_size) + [0] * (int(batch_sizes[0])
                                            - int(batch_size)) for batch_size in batch_sizes]
        )))
        if self.use_gpu:
            mask = mask.cuda()
        return mask

    def compute_loss(self, training_minibatch):
        _, labels_list, remapped_labels_list_crf_hmm, remapped_labels_list_crf_marg, \
        _prot_type_list, _prot_topology_list, _prot_name_list, original_aa_string, \
        _original_label_string = training_minibatch
        minibatch_size = len(labels_list)
        if self.model_mode == TMHMM3Mode.LSTM_CRF_MARG:
            labels_to_use = remapped_labels_list_crf_marg
        elif self.model_mode == TMHMM3Mode.LSTM_CRF_HMM:
            labels_to_use = remapped_labels_list_crf_hmm
        else:
            labels_to_use = labels_list
        input_sequences = [autograd.Variable(x) for x in self.embed(original_aa_string)]

        actual_labels = torch.nn.utils.rnn.pad_sequence([autograd.Variable(l)
                                                         for l in labels_to_use])
        emissions, batch_sizes = self._get_network_emissions(input_sequences)
        if self.model_mode == TMHMM3Mode.LSTM:
            prediction = emissions.transpose(0, 1).contiguous().view(-1, emissions.size(-1))
            target = actual_labels.transpose(0, 1).contiguous().view(-1, 1)
            losses = -torch.gather(nn.functional.log_softmax(prediction),
                                   dim=1, index=target).view(*actual_labels
                                                             .transpose(0, 1).size())
            mask_expand = torch.range(0, batch_sizes.data.max() - 1).long() \
                .unsqueeze(0).expand(batch_sizes.size(0), batch_sizes.data.max())
            if self.use_gpu:
                mask_expand = mask_expand.cuda()
                batch_sizes = batch_sizes.cuda()
            mask = mask_expand < batch_sizes.unsqueeze(1).expand_as(mask_expand)
            loss = (losses * mask.float()).sum() / batch_sizes.float().sum()
        else:
            mask = (self.batch_sizes_to_mask(batch_sizes))
            loss = -1 * self.crf_model(emissions, actual_labels, mask=mask) / minibatch_size
            if float(loss) > 100000: # if loss is this large, an invalid tx must have been found
                for idx, batch_size in enumerate(batch_sizes):
                    last_label = None
                    for i in range(batch_size):
                        label = int(actual_labels[i][idx])
                        write_out(str(label) + ",", end='')
                        if last_label is not None and (last_label, label) \
                                not in self.allowed_transitions:
                            write_out("Error: invalid transition found")
                            write_out((last_label, label))
                            sys.exit(1)
                        last_label = label
                    write_out(" ")
        return loss

    def forward(self, input_sequences, forced_types=None):
        emissions, batch_sizes = self._get_network_emissions(input_sequences)
        if self.model_mode == TMHMM3Mode.LSTM:
            output = torch.nn.functional.log_softmax(emissions, dim=2)
            _, predicted_labels = output[:, :, 0:5].max(dim=2)
            predicted_labels = list(
                [list(map(int, x[:batch_sizes[idx]])) for idx, x in enumerate(predicted_labels
                                                                              .transpose(0, 1))])
            predicted_labels = list(
                torch.cuda.LongTensor(l) if self.use_gpu else torch.LongTensor(l)
                for l in predicted_labels)
            predicted_topologies = list(map(label_list_to_topology, predicted_labels))
            predicted_types = torch.LongTensor(list(map(get_predicted_type_from_labels,
                                                        predicted_labels)))

        else:
            mask = self.batch_sizes_to_mask(batch_sizes)
            labels_predicted = list(torch.cuda.LongTensor(l) if self.use_gpu
                                    else torch.LongTensor(l) for l in
                                    self.crf_model.decode(emissions, mask=mask))

            if self.model_mode == TMHMM3Mode.LSTM_CRF_HMM:
                predicted_labels = list(map(remapped_labels_hmm_to_orginal_labels,
                                            labels_predicted))
                predicted_types = torch.LongTensor(list(map(get_predicted_type_from_labels,
                                                            predicted_labels)))
            elif self.model_mode == TMHMM3Mode.LSTM_CRF_MARG:
                alpha = self.crf_model._compute_log_alpha(emissions, mask, run_backwards=False)
                z_value = alpha[alpha.size(0) - 1] + self.crf_model.end_transitions
                types = z_value.view((-1, 4, 5))
                types = logsumexp(types, dim=2)
                _, predicted_types = torch.max(types, dim=1)
                predicted_labels = list([l % 5 for l in labels_predicted])  # remap
            else:
                predicted_labels = labels_predicted
                predicted_types = torch.LongTensor(list(map(get_predicted_type_from_labels,
                                                            predicted_labels)))

            if self.use_gpu:
                predicted_types = predicted_types.cuda()
            predicted_topologies = list(map(label_list_to_topology, predicted_labels))

        # if all O's, change to all I's (by convention)
        for idx, labels in enumerate(predicted_labels):
            if torch.eq(labels, 4).all():
                predicted_labels[idx] = labels - 1

        return predicted_labels, predicted_types if forced_types \
                                                    is None else forced_types, predicted_topologies

    def evaluate_model(self, data_loader):
        validation_loss_tracker = []
        validation_type_loss_tracker = []
        validation_topology_loss_tracker = []
        confusion_matrix = np.zeros((5, 5), dtype=np.int64)
        protein_names = []
        protein_aa_strings = []
        protein_label_actual = []
        protein_label_prediction = []
        for _, minibatch in enumerate(data_loader, 0):
            validation_loss_tracker.append(self.compute_loss(minibatch).detach())

            _, _, _, _, prot_type_list, prot_topology_list, \
            prot_name_list, original_aa_string, original_label_string = minibatch
            input_sequences = [x for x in self.embed(original_aa_string)]
            predicted_labels, predicted_types, predicted_topologies = self(input_sequences)

            protein_names.extend(prot_name_list)
            protein_aa_strings.extend(original_aa_string)
            protein_label_actual.extend(original_label_string)

            # if we're using an external type predictor
            if self.type_classifier is not None:
                predicted_labels_type_classifer, \
                predicted_types_type_classifier, \
                predicted_topologies_type_classifier = self.type_classifier(input_sequences)

            for idx, actual_type in enumerate(prot_type_list):

                predicted_type = predicted_types[idx]
                predicted_topology = predicted_topologies[idx]
                predicted_labels_for_protein = predicted_labels[idx]

                if self.type_classifier is not None:
                    if predicted_type != predicted_types_type_classifier[idx]:
                        # we must always use the type predicted by the type predictor if available
                        predicted_type = predicted_types_type_classifier[idx]
                        predicted_topology = predicted_topologies_type_classifier[idx]
                        predicted_labels_for_protein = predicted_labels_type_classifer[idx]

                    prediction_topology_match = is_topologies_equal(prot_topology_list[idx],
                                                                    predicted_topology, 5)

                    if actual_type == predicted_type:
                        validation_type_loss_tracker.append(0)
                        # if we guessed the type right for SP+GLOB or GLOB,
                        # count the topology as correct
                        if actual_type == 2 or actual_type == 3 or prediction_topology_match:
                            validation_topology_loss_tracker.append(0)
                            confusion_matrix[actual_type][4] += 1
                    else:
                        validation_topology_loss_tracker.append(1)
                        confusion_matrix[actual_type][predicted_type] += 1
                        # if the type was correctly guess 2 or 3 by the type classifier,
                        # use its topology prediction
                        if (actual_type in (2, 3)) and self.type_classifier is not None:
                            protein_label_prediction.append(predicted_labels_type_classifer[idx])
                        else:
                            protein_label_prediction.append(predicted_labels_for_protein)
                else:
                    confusion_matrix[actual_type][predicted_type] += 1
                    validation_type_loss_tracker.append(1)
                    validation_topology_loss_tracker.append(1)
                    protein_label_prediction.append(predicted_labels_for_protein)

        write_out(confusion_matrix)
        _loss = float(torch.stack(validation_loss_tracker).mean())

        type_loss = float(torch.FloatTensor(validation_type_loss_tracker).mean().detach())
        topology_loss = float(torch.FloatTensor(validation_topology_loss_tracker).mean().detach())

        self.type_01loss_values.append(type_loss)
        self.topology_01loss_values.append(topology_loss)

        if get_experiment_id() is not None and "TYPE" in get_experiment_id():
            # optimize for type
            validation_loss = type_loss
        else:
            # optimize for topology
            validation_loss = topology_loss

        data = {}
        data['type_01loss_values'] = self.type_01loss_values
        data['topology_01loss_values'] = self.topology_01loss_values
        data['confusion_matrix'] = confusion_matrix.tolist()

        return validation_loss, data, (
            protein_names, protein_aa_strings, protein_label_actual, protein_label_prediction)


def post_process_prediction_data(prediction_data):
    data = []
    for (name, aa_string, actual, prediction) in zip(*prediction_data):
        data.append("\n".join([">" + name,
                               aa_string,
                               actual,
                               original_labels_to_fasta(prediction)]))
    return "\n".join(data)


def logsumexp(data, dim):
    return data.max(dim)[0] + torch.log(torch.sum(
        torch.exp(data - data.max(dim)[0].unsqueeze(dim)), dim))


def initialize_crf_parameters(crf_model,
                              start_transitions=None,
                              end_transitions=None,
                              transitions=None) -> None:
    """Initialize the transition parameters.

    The parameters will be initialized randomly from a uniform distribution
    between -0.1 and 0.1, unless given explicitly as an argument.
    """
    if start_transitions is None:
        nn.init.uniform(crf_model.start_transitions, -0.1, 0.1)
    else:
        crf_model.start_transitions.data = start_transitions
    if end_transitions is None:
        nn.init.uniform(crf_model.end_transitions, -0.1, 0.1)
    else:
        crf_model.end_transitions.data = end_transitions
    if transitions is None:
        nn.init.uniform(crf_model.transitions, -0.1, 0.1)
    else:
        crf_model.transitions.data = transitions


def generate_masked_crf_transitions(crf_model, transition_mask):
    start_transitions_mask, transitions_mask, end_transition_mask = transition_mask
    start_transitions = crf_model.start_transitions.data.clone()
    end_transitions = crf_model.end_transitions.data.clone()
    transitions = crf_model.transitions.data.clone()
    if start_transitions_mask is not None:
        start_transitions.masked_fill_(start_transitions_mask, -100000000)
    if end_transition_mask is not None:
        end_transitions.masked_fill_(end_transition_mask, -100000000)
    if transitions_mask is not None:
        transitions.masked_fill_(transitions_mask, -100000000)
    return start_transitions, end_transitions, transitions


class TMHMM3Mode(Enum):
    LSTM = 1
    LSTM_CRF = 2
    LSTM_CRF_HMM = 3
    LSTM_CRF_MARG = 4
