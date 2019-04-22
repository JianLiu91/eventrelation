import sys
import torch
import pickle
import random

from tqdm import tqdm

def pad_sequence_to_length(sequence,
                           desired_length,
                           default_value = lambda: 0,
                           padding_on_right = True):
    """
    Take a list of objects and pads it to the desired length, returning the padded list.  The
    original list is not modified.
    Parameters
    ----------
    sequence : List
        A list of objects to be padded.
    desired_length : int
        Maximum length of each sequence. Longer sequences are truncated to this length, and
        shorter ones are padded to it.
    default_value: Callable, default=lambda: 0
        Callable that outputs a default value (of any type) to use as padding values.  This is
        a lambda to avoid using the same object when the default value is more complex, like a
        list.
    padding_on_right : bool, default=True
        When we add padding tokens (or truncate the sequence), should we do it on the right or
        the left?
    Returns
    -------
    padded_sequence : List
    """
    # Truncates the sequence to the desired length.
    if padding_on_right:
        padded_sequence = sequence[:desired_length]
    else:
        padded_sequence = sequence[-desired_length:]
    # Continues to pad with default_value() until we reach the desired length.
    for _ in range(desired_length - len(padded_sequence)):
        if padding_on_right:
            padded_sequence.append(default_value())
        else:
            padded_sequence.insert(0, default_value())
    return padded_sequence

def get_mask_from_sequence_lengths(sequence_lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    """
    Given a variable of shape ``(batch_size,)`` that represents the sequence lengths of each batch
    element, this function returns a ``(batch_size, max_length)`` mask variable.  For example, if
    our input was ``[2, 2, 3]``, with a ``max_length`` of 4, we'd return
    ``[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]``.
    We require ``max_length`` here instead of just computing it from the input ``sequence_lengths``
    because it lets us avoid finding the max, then copying that value from the GPU to the CPU so
    that we can use it to construct a new tensor.
    """
    # (batch_size, max_length)
    ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
    range_tensor = ones.cumsum(dim=1)
    return (sequence_lengths.unsqueeze(1) >= range_tensor).long()


class Dataset(object):
    def __init__(self, batch_size, ds='Train'):
        super(Dataset, self).__init__()

        self.batch_size = batch_size
        self.y_label = {
            'NULL': 0,
            'null': 0,
            'FALLING_ACTION': 1,
            'PRECONDITION': 2
        }

        input_map = 'training_data.txt'
        dataset = []
        with open(input_map) as f:
            for line in f:
                field = line.strip().split('\t')
                dataset.append(field)

        self.construct_index(dataset)
        self.shuffle()

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
        lens1 = [len(tup[4].split('_')) for tup in batch]
        lens2 = [len(tup[8].split('_')) for tup in batch]
        word_padded_len = 3
        data_x1, data_x2, data_y = list(), list(), list()
        for data in batch:
            data_x1.append(list(map(int, data[4].split('_'))))
            data_x2.append(list(map(int, data[8].split('_'))))
            data_y.append(self.y_label[data[-1]])

        data_x1 = list(map(lambda x: pad_sequence_to_length(x, word_padded_len), data_x1))
        data_x2 = list(map(lambda x: pad_sequence_to_length(x, word_padded_len), data_x2))

        mask_x1 = get_mask_from_sequence_lengths(torch.LongTensor(lens1), word_padded_len)
        mask_x2 = get_mask_from_sequence_lengths(torch.LongTensor(lens2), word_padded_len)

        return [torch.LongTensor(data_x1).to(device), mask_x1,
                torch.LongTensor(data_x2).to(device), mask_x2,
                torch.Tensor(data_y).to(device)]


if __name__ == '__main__':
    dataset = Dataset(10)
    for batch in dataset.reader('cpu', True):
        data_x1, mask_x1, data_x2, mask_x2, data_y = batch
        print(data_x1.size(0))
