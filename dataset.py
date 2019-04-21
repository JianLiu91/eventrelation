import sys
import torch
import pickle
import random

from tqdm import tqdm
from allennlp.common.util import pad_sequence_to_length
from allennlp.nn.util import get_mask_from_sequence_lengths
from allennlp.modules.elmo import batch_to_ids

class SeqDatasetCN(object):
    def __init__(self, batch_size, ds='Train'):
        super(SeqDatasetCN, self).__init__()

        self.batch_size = batch_size

        input_map = 'zh_dataset.pickle'
        with open(input_map, 'rb') as f:
            p_data = pickle.load(f)
            self.word_map = p_data['word_map']
            self.word_list = p_data['word_list']
            self.word_vectors = p_data['word_vectors']
            self.label_map = p_data['label_map']
            self.idx2map = {v:k for k, v in self.label_map.items()}

            self.train = p_data['train']
            self.dev = p_data['dev']
            self.test = p_data['test']

            if ds == 'All':
                self.construct_index(self.train + self.dev)
                self.shuffle()
            elif ds == 'Train':
                self.construct_index(self.train)
                self.shuffle()
            elif ds == 'Dev':
                self.construct_index(self.dev)
            elif ds == 'Test':
                self.construct_index(self.test)


    def construct_index(self, dataset):
        self.dataset = dataset
        self.index_length = len(dataset)
        self.shuffle_list = list(range(0, self.index_length))


    def shuffle(self):
        random.shuffle(self.shuffle_list)

    def get_tqdm(self, device, shuffle=True):
        return tqdm(self.reader(device, shuffle), mininterval=2, total=self.index_length // self.batch_size, leave=False, file=sys.stdout, ncols=80)

    def reader(self, device, shuffle):
        cur_idx = 0
        while cur_idx < self.index_length:
            end_index = min(cur_idx + self.batch_size, self.index_length)
            batch = [self.dataset[self.shuffle_list[index]] for index in range(cur_idx, end_index)]
            cur_idx = end_index
            yield self.batchify(batch, device)
        if shuffle:
            self.shuffle()

    def batchify(self, batch, device):
        lens = [len(tup[0]) for tup in batch]
        word_padded_len = max(lens)
        data_x, data_y = list(), list()
        for x, y in batch:
            data_x.append(x)
            data_y.append(y)

        data_x = list(map(lambda x: pad_sequence_to_length(x, word_padded_len), data_x))
        data_y = list(map(lambda x: pad_sequence_to_length(x, word_padded_len, default_value=lambda: [self.label_map['O']] * len(self.label_map)), data_y))
        mask = get_mask_from_sequence_lengths(torch.LongTensor(lens), word_padded_len)
        return [torch.LongTensor(data_x).to(device), torch.Tensor(data_y).to(device), mask.to(device)]
