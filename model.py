import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.modules.seq2seq_encoders.multi_head_self_attention import MultiHeadSelfAttention


class BasicCausalModel(nn.Module):

    def __init__(self, w_num, w_dim, w_hidden, y_num):
        super(BasicCausalModel, self).__init__()

        self.y_num = y_num
        self.word_embed = nn.Embedding(w_num, w_dim)
        self.feature2hidden = nn.Linear(w_dim * 2, w_hidden)
        self.hidden2tag = nn.Linear(w_hidden, self.y_num)

    def forward(self, data_x1, mask_x1, data_x2, mask_x2):

        x1_emb = self.word_embed(data_x1)
        x2_emb = self.word_embed(data_x2)

        m1 = mask_x1.unsqueeze(-1).expand_as(x1_emb).float()
        m2 = mask_x2.unsqueeze(-1).expand_as(x2_emb).float()

        x1_emb = x1_emb * m1
        x2_emb = x2_emb * m2

        opt1 = torch.mean(x1_emb, dim=1)
        opt2 = torch.mean(x2_emb, dim=1)

        opt = torch.cat((opt1, opt2), 1)
        opt = self.feature2hidden(opt)
        opt = self.hidden2tag(opt)


        return opt


