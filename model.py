import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from allennlp.modules.conditional_random_field import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.modules.seq2seq_encoders.multi_head_self_attention import MultiHeadSelfAttention


from modules import M_ELMO
from modules import init_lstm


class EventLabelCN(nn.Module):

    def __init__(self, w_num, w_dim, w_hidden, w_layer, y_num, droprate, unit='lstm'):
        super(EventLabelCN, self).__init__()

        rnnunit_map = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}

        self.word_embed = nn.Embedding(w_num, w_dim)

        tmp_rnn_dropout = droprate if w_layer > 1 else 0
        self.word_rnn = rnnunit_map[unit](w_dim, w_hidden // 2, w_layer, dropout=tmp_rnn_dropout,
                                          bidirectional=True, batch_first=True)

        self.y_num = y_num
        self.self_attention = MultiHeadSelfAttention(30, w_hidden, 60, 60) #########
        self.hidden2tag = nn.Linear(w_hidden, self.y_num)
        self.crf = ConditionalRandomField(y_num)  # add constraint
        self.drop = nn.Dropout(p=droprate)

        self.rand_init()

    def set_batch_seq_size(self, sentence):
        """
        set batch size and sequence length
        """
        tmp = sentence.size()
        self.word_seq_length = tmp[0]
        self.batch_size = tmp[1]

    def load_pretrained_word_embedding(self, pre_word_embeddings, updated=False):
        """
        Load pre-trained word embedding.
        """
        self.word_embed.weight = nn.Parameter(pre_word_embeddings)
        self.word_embed.weight.requires_grad = updated

    def rand_init(self):
        def _init_lstm(input_lstm):
            """
            random initialize lstms
            """
            for ind in range(0, input_lstm.num_layers):
                weight = eval('input_lstm.weight_ih_l' + str(ind))
                bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
                nn.init.uniform_(weight, -bias, bias)
                weight = eval('input_lstm.weight_hh_l' + str(ind))
                bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
                nn.init.uniform_(weight, -bias, bias)

            if input_lstm.bias:
                for ind in range(0, input_lstm.num_layers):
                    weight = eval('input_lstm.bias_ih_l' + str(ind))
                    weight.data.zero_()
                    weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                    weight = eval('input_lstm.bias_hh_l' + str(ind))
                    weight.data.zero_()
                    weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

        _init_lstm(self.word_rnn)
        ## self.crf.rand_init()  crf has been initialized ...

    def compute_logits(self, f_w):
        self.set_batch_seq_size(f_w)
        w_e = self.word_embed(f_w)
        out = self.drop(w_e)
        out, _ = self.word_rnn(out)
        out = self.self_attention(out)
        logits = self.hidden2tag(out)
        return logits

        # tmp = f_tags > 0
        # alpha = 0.3
        # tmp = alpha * tmp
        # f_mask = tmp.long() + f_mask
        # likelihood = -sequence_cross_entropy_with_logits(logits, f_tags, f_mask)


    def forward(self, f_w, f_tags, f_mask):
        logits = self.compute_logits(f_w)

        class_dim = logits.size(-1)
        logits = logits.view(-1, class_dim)
        f_tags = f_tags.view(-1, class_dim)
        f_m = f_mask.float().view(-1).unsqueeze(-1)

        loss_func = nn.MultiLabelSoftMarginLoss(reduction='none')
        likelihood = -loss_func(logits, f_tags)
        likelihood = likelihood * f_m
        likelihood = torch.sum(likelihood)

        # likelihood = -sequence_cross_entropy_with_logits(logits, f_tags)
        #  = self.crf(logits, f_tags, f_mask)
        return likelihood


    def decode(self, f_w, f_mask):
        logits = self.compute_logits(f_w)
        lens = get_lengths_from_binary_sequence_mask(f_mask)
        lens = list(lens.cpu().data.numpy())

        out = torch.sigmoid(logits)
        out = list(out.cpu().data.numpy())

        result = []
        for o, l in zip(out, lens):
            temp = []
            for t in range(0, l):
                r = []
                for i in range(0, len(o[t])):
                    if o[t][i] > 0.35:
                        r.append(i)
                if len(r) == 0:
                    r = [0]
                temp.append(r)
            result.append(temp)

        return result
