# This file is part of the TMHMM3 project.
#
# @author Jeppe Hallgren
#
# For license information, please see the LICENSE file in the root directory.
from enum import Enum

import tensorflow as tf
import torch.autograd as autograd
import torch.nn as nn

import openprotein
from experiments.tmhmm3.tm_util import *
from pytorchcrf.torchcrf import CRF
from util import write_out
import os

# seed random generator for reproducibility
torch.manual_seed(1)

class TMHMM3(openprotein.BaseModel):
    def __init__(self, embedding, hidden_size, use_gpu, model_mode, use_marg_prob, type_predictor_model):
        super(TMHMM3, self).__init__(embedding, use_gpu)

        # initialize model variables
        num_tags = 5
        num_labels = 5
        if model_mode == TMHMM3Mode.LSTM_CRF_HMM:
            num_tags += 2 * 40 + 60
        elif model_mode == TMHMM3Mode.LSTM_CRF_MARG:
            num_tags = num_tags * 4 # 4 different types
            num_labels = num_tags # 4 different types
        elif model_mode == TMHMM3Mode.LSTM_CTC:
            num_tags += 1 # add extra class for blank
            num_labels += 1
        self.hidden_size = hidden_size
        self.use_gpu = use_gpu
        self.use_marg_prob = use_marg_prob
        self.model_mode = model_mode
        self.embedding = embedding
        self.embedding_function = nn.Embedding(24, self.get_embedding_size())
        self.bi_lstm = nn.LSTM(self.get_embedding_size(), self.hidden_size, num_layers=1, bidirectional=True)
        self.hidden_to_labels = nn.Linear(self.hidden_size * 2, num_labels) # * 2 for bidirectional

        crf_start_mask = torch.ones(num_tags).byte()
        crf_end_mask = torch.ones(num_tags).byte()
        if model_mode == TMHMM3Mode.LSTM_CRF_HMM:
            allowed_transitions = [
                (3, 3), (4, 4),
                (3, 5), (4, 45)]
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
            for i in range(85, 144):
                allowed_transitions.append((i, i + 1))
                allowed_transitions.append((2, i))
            allowed_transitions.append((2, 144))
            allowed_transitions.append((2, 4))
            allowed_transitions.append((144, 4))

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
        self.crfModel = CRF(num_tags)
        self.type_classifier = type_predictor_model
        self.type_tm_classier = None
        self.type_sp_classier = None
        crf_transitions_mask = torch.ones((num_tags, num_tags)).byte()

        self.label_01loss_values = []
        self.type_01loss_values = []
        self.topology_01loss_values = []

        # if on GPU, move state to GPU memory
        if self.use_gpu:
            self.embedding_function = self.embedding_function.cuda()
            self.crfModel = self.crfModel.cuda()
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
            self.generate_masked_crf_transitions(
                self.crfModel, (crf_start_mask, crf_transitions_mask, crf_end_mask)
            )

        # initialize CRF
        self.initialize_crf_parameters(self.crfModel,
                                       start_transitions=crf_start_transitions,
                                       end_transitions=crf_end_transitions,
                                       transitions=crf_transitions)

    def initialize_crf_parameters(self,
                                  crfModel,
                                  start_transitions=None,
                                  end_transitions=None,
                                  transitions=None) -> None:
        """Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1, unless given explicitly as an argument.
        """
        if start_transitions is None:
            nn.init.uniform(crfModel.start_transitions, -0.1, 0.1)
        else:
            crfModel.start_transitions.data = start_transitions
        if end_transitions is None:
            nn.init.uniform(crfModel.end_transitions, -0.1, 0.1)
        else:
            crfModel.end_transitions.data = end_transitions
        if transitions is None:
            nn.init.uniform(crfModel.transitions, -0.1, 0.1)
        else:
            crfModel.transitions.data = transitions

    def generate_masked_crf_transitions(self, crf_model, transition_mask):
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

    def get_embedding_size(self):
        return 24 # bloom matrix has size 24

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
-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,-4,1""".replace('\n', ',')
                blosum_matrix = np.fromstring(blosum, sep=",").reshape(24,24)
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
            t = list([self.encode_amino_acid(aa) for aa in aa_list])
            if self.embedding == "PYTORCH":
                t = torch.LongTensor(t)
            else:
                t= torch.FloatTensor(t)
            if self.use_gpu:
                t = t.cuda()
            embed_list.append(t)
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

        if self.embedding == "PYTORCH":
            pad_seq, seq_length = torch.nn.utils.rnn.pad_sequence(input_sequences), [v.size(0) for v in input_sequences]
            pad_seq_embed = self.embedding_function(pad_seq)
            packed = torch.nn.utils.rnn.pack_padded_sequence(pad_seq_embed, seq_length)
        else:
            packed = torch.nn.utils.rnn.pack_sequence(input_sequences)
        minibatch_size = len(input_sequences)
        self.init_hidden(minibatch_size)
        bi_lstm_out, self.hidden_layer = self.bi_lstm(packed, self.hidden_layer)
        data, batch_sizes, _, _ = bi_lstm_out
        emissions = self.hidden_to_labels(data)
        if self.model_mode == TMHMM3Mode.LSTM_CRF_HMM:
            inout_select = torch.LongTensor([0])
            outin_select = torch.LongTensor([1])
            signal_select = torch.LongTensor([2])
            if self.use_gpu:
                inout_select = inout_select.cuda()
                outin_select = outin_select.cuda()
                signal_select = signal_select.cuda()
            inout = torch.index_select(emissions, 1, autograd.Variable(inout_select))
            outin = torch.index_select(emissions, 1, autograd.Variable(outin_select))
            signal = torch.index_select(emissions, 1, autograd.Variable(signal_select))
            emissions = torch.cat((emissions, inout.expand(-1, 40), outin.expand(-1, 40), signal.expand(-1, 60)), 1)
        emissions_padded = torch.nn.utils.rnn.pad_packed_sequence(torch.nn.utils.rnn.PackedSequence(emissions,batch_sizes))
        return emissions_padded

    def batch_sizes_to_mask(self, batch_sizes):
        mask = torch.autograd.Variable(torch.t(torch.ByteTensor(
            [[1] * int(batch_size) + [0] * (int(batch_sizes[0]) - int(batch_size)) for batch_size in batch_sizes]
        )))
        if self.use_gpu:
            mask = mask.cuda()
        return mask

    def compute_loss(self, training_minibatch):
        _, labels_list, remapped_labels_list_crf_hmm, remapped_labels_list_crf_marg, prot_type_list, prot_topology_list, prot_name_list, original_aa_string, original_label_string = training_minibatch
        minibatch_size = len(labels_list)
        if self.model_mode == TMHMM3Mode.LSTM_CRF_MARG:
            labels_to_use = remapped_labels_list_crf_marg
        elif self.model_mode == TMHMM3Mode.LSTM_CRF_HMM:
            labels_to_use = remapped_labels_list_crf_hmm
        else:
            labels_to_use = labels_list
        input_sequences = [autograd.Variable(x) for x in self.embed(original_aa_string)]
        if self.model_mode == TMHMM3Mode.LSTM_CTC:
            # CTC loss
            emissions, batch_sizes = self._get_network_emissions(input_sequences)
            output = torch.nn.functional.log_softmax(emissions, dim=2)
            topologies = list([torch.LongTensor(list([label for (idx, label) in label_list_to_topology(a)])) for a in labels_list])
            if self.use_gpu:
                topologies = list([a.cuda() for a in topologies])
            targets, target_lengths = torch.nn.utils.rnn.pad_sequence(topologies).transpose(0,1), list([a.size()[0] for a in topologies])
            ctc_loss = nn.CTCLoss(blank=5)
            return ctc_loss(output, targets, tuple(batch_sizes), tuple(target_lengths))
        else:
            actual_labels = torch.nn.utils.rnn.pad_sequence([autograd.Variable(l) for l in labels_to_use])
            emissions, batch_sizes = self._get_network_emissions(input_sequences)
            if self.model_mode == TMHMM3Mode.LSTM:
                prediction = emissions.transpose(0,1).contiguous().view(-1, emissions.size(-1))
                target = actual_labels.transpose(0,1).contiguous().view(-1, 1)
                losses = -torch.gather(nn.functional.log_softmax(prediction), dim=1, index=target).view(*actual_labels.transpose(0,1).size())
                mask_expand = torch.range(0, batch_sizes.data.max() - 1).long().unsqueeze(0).expand(batch_sizes.size(0), batch_sizes.data.max())
                if self.use_gpu:
                    mask_expand = mask_expand.cuda()
                    batch_sizes = batch_sizes.cuda()
                mask = mask_expand < batch_sizes.unsqueeze(1).expand_as(mask_expand)
                loss = (losses * mask.float()).sum() / batch_sizes.float().sum()
            else:
                loss = -1 * self.crfModel(emissions, actual_labels, mask=self.batch_sizes_to_mask(batch_sizes)) / minibatch_size
                if float(loss) > 100000:
                    for idx, batch_size in enumerate(batch_sizes):
                        last_label = None
                        for i in range(batch_size):
                            label = int(actual_labels[i][idx])
                            write_out(str(label) + ",", end='')
                            if last_label is not None and (last_label, label) not in self.allowed_transitions:
                                write_out("Error: invalid transition found")
                                write_out((last_label, label))
                                exit()
                            last_label = label
                        write_out(" ")
            return loss

    def forward(self, original_aa_string, forced_types=None):
        input_sequences = [autograd.Variable(x) for x in self.embed(original_aa_string)]
        emissions, batch_sizes = self._get_network_emissions(input_sequences)
        if self.model_mode == TMHMM3Mode.LSTM_CTC or self.model_mode == TMHMM3Mode.LSTM:
            output = torch.nn.functional.log_softmax(emissions, dim=2)
            _, predicted_labels = output.max(dim=2)
            predicted_labels = list([list(map(int,x[:batch_sizes[idx]])) for idx, x in enumerate(predicted_labels.transpose(0,1))])
            predicted_topologies = list(map(label_list_to_topology, predicted_labels))
            if forced_types is None and self.model_mode == TMHMM3Mode.LSTM_CTC:
                tf_output = tf.placeholder(tf.float32, shape=emissions.size())
                tf_batch_sizes = tf.placeholder(tf.int32, shape=(emissions.size()[1]))
                beam_decoded, _ = tf.nn.ctc_beam_search_decoder(tf_output, sequence_length=tf_batch_sizes)
                decoded_topology = tf.sparse_tensor_to_dense(beam_decoded[0])
                tmp = '-1'
                if 'CUDA_VISIBLE_DEVICES' in os.environ:
                    tmp = os.environ['CUDA_VISIBLE_DEVICES']
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                with tf.Session() as session:
                    tf.global_variables_initializer().run()
                    decoded_topology = session.run(decoded_topology, feed_dict={tf_output: output.detach().cpu().numpy(), tf_batch_sizes: batch_sizes})
                    predicted_types = list(map(get_predicted_type_from_labels, decoded_topology))
                os.environ['CUDA_VISIBLE_DEVICES'] = tmp
            else:
                predicted_types = list(map(get_predicted_type_from_labels, predicted_labels))

        else:
            mask = self.batch_sizes_to_mask(batch_sizes)

            if self.model_mode == TMHMM3Mode.LSTM_CRF_HMM:
                labels_predicted = self.crfModel.decode(emissions, mask=mask)
                predicted_labels = list(map(remapped_labels_hmm_to_orginal_labels, labels_predicted))
                predicted_types = list(map(get_predicted_type_from_labels, predicted_labels))
            elif self.model_mode == TMHMM3Mode.LSTM_CRF_MARG:
                alpha = self.crfModel._compute_log_alpha(emissions, mask, run_backwards=False)
                z = alpha[alpha.size(0)-1] + self.crfModel.end_transitions
                type = z.view((-1, 4, 5))
                type = torch.logsumexp(type,dim=2)
                max, predicted_types = torch.max(type, dim=1)

                labels_predicted = list(torch.cuda.LongTensor(l) if self.use_gpu else torch.LongTensor(l) for l in self.crfModel.decode(emissions, mask=mask))

                predicted_labels = list([l % 5 for l in labels_predicted]) # remap

            else:
                predicted_labels = self.crfModel.decode(emissions, mask=mask)
                predicted_types = list(map(get_predicted_type_from_labels, predicted_labels))

            predicted_topologies = list(map(label_list_to_topology, predicted_labels))

        return predicted_labels, predicted_types if forced_types is None else forced_types, predicted_topologies

    def evaluate_model(self, data_loader):
        validation_loss_tracker = []
        validation_type_loss_tracker = []
        validation_topology_loss_tracker = []
        confusion_matrix = np.zeros((5,5), dtype=np.int64)
        protein_names = []
        protein_aa_strings = []
        protein_label_actual = []
        protein_label_prediction = []
        for i, minibatch in enumerate(data_loader, 0):
            validation_loss_tracker.append(self.compute_loss(minibatch).detach())

            _, labels_list, _, _, prot_type_list, prot_topology_list, prot_name_list, original_aa_string, original_label_string = minibatch
            predicted_labels, predicted_types, predicted_topologies = self(original_aa_string)

            protein_names.extend(prot_name_list)
            protein_aa_strings.extend(original_aa_string)
            protein_label_actual.extend(original_label_string)

            # if we're using an external type predictor
            if self.type_classifier is not None:
                predicted_labels_type_classifer, predicted_types, _ = self.type_classifier(original_aa_string)

            for idx, actual_type in enumerate(prot_type_list):
                predicted_type = predicted_types[idx]
                prediction_topology_match = is_topologies_equal(prot_topology_list[idx], predicted_topologies[idx], 5)
                if actual_type == predicted_type:
                    validation_type_loss_tracker.append(0)
                    # if we guessed the type right for SP+GLOB or GLOB, we count the topology as correct
                    if actual_type == 2 or actual_type == 3 or prediction_topology_match:
                        validation_topology_loss_tracker.append(0)
                        confusion_matrix[actual_type][4] += 1
                    else:
                        validation_topology_loss_tracker.append(1)
                        confusion_matrix[actual_type][predicted_type] += 1
                    # if the type was correctly guess 2 or 3 by the type classifier, use its topology prediction
                    if (actual_type == 2 or actual_type == 3) and self.type_classifier is not None:
                        protein_label_prediction.append(predicted_labels_type_classifer[idx])
                    else:
                        protein_label_prediction.append(predicted_labels[idx])
                else:
                    confusion_matrix[actual_type][predicted_type] += 1
                    validation_type_loss_tracker.append(1)
                    validation_topology_loss_tracker.append(1)
                    if self.type_classifier is not None:
                        # if the type prediction is wrong, we must use labels predicted by type predictor if available
                        protein_label_prediction.append(predicted_labels_type_classifer[idx])
                        if prediction_topology_match:
                            confusion_matrix[4][actual_type] += 1
                    else:
                        protein_label_prediction.append(predicted_labels[idx])

        write_out(confusion_matrix)
        loss = float(torch.stack(validation_loss_tracker).mean())

        type_loss = float(torch.FloatTensor(validation_type_loss_tracker).mean().detach())
        topology_loss = float(torch.FloatTensor(validation_topology_loss_tracker).mean().detach())

        self.type_01loss_values.append(type_loss)
        self.topology_01loss_values.append(topology_loss)

        if self.type_classifier is None:
            # optimize for type
            validation_loss = type_loss
        else:
            # optimize for topology
            validation_loss = topology_loss

        data = {}
        data['type_01loss_values'] = self.type_01loss_values
        data['topology_01loss_values'] = self.topology_01loss_values
        data['confusion_matrix'] = confusion_matrix.tolist()
        write_out(data)

        return validation_loss, data, (protein_names, protein_aa_strings, protein_label_actual, protein_label_prediction)

    def post_process_prediction_data(self, prediction_data):
        data = []
        for (name, aa_string, actual, prediction) in zip(*prediction_data):
            data.append("\n".join([">" + name, aa_string, actual, original_labels_to_fasta(prediction)]))
        return "\n".join(data)

class TMHMM3Mode(Enum):
    LSTM = 1
    LSTM_CRF = 2
    LSTM_CRF_HMM = 3
    LSTM_CRF_MARG = 4
    LSTM_CTC = 5