import torch
import pickle
from dataset import Dataset
from model import BasicCausalModel

def split_train_test(dataset):
    train_set = []
    test_set = []

    test_topic = ['37', '41']
    for data in dataset:
        t = data[0]
        if t.split('/')[-2] in test_topic:
            test_set.append(data)
        else:
            train_set.append(data)
    return train_set, test_set

def compute_f1(gold, predicted):
    c_predict = 0
    c_correct = 0
    c_gold = 0

    for g, p in zip(gold, predicted):
        if g != 0:
            c_gold += 1
        if p != 0:
            c_predict += 1
        if g != 0 and p != 0:
            c_correct += 1

    p = c_correct / c_predict
    r = c_correct / c_gold
    f = 2 * p * r / (p + r)
    return p, r, f


if __name__ == '__main__':

    with open('data.pickle', 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        data = pickle.load(f)
        documents = data['data']
        word_map = data['word_map']
        word_list = data['word_list']
        word_vector = data['word_vector']

    dataset = []
    input_map = 'training_data.txt'
    dataset = []
    with open(input_map) as f:
        for line in f:
            field = line.strip().split('\t')
            dataset.append(field)
    train_set, test_set = split_train_test(dataset)
    print(len(train_set))
    print(len(test_set))

    train_dataset = Dataset(200, train_set)
    test_dataset = Dataset(200, test_set)

    model = BasicCausalModel(len(word_list), 300, 100, 3)
    model.word_embed.wight = word_vector
    model.word_embed.weight.requires_grad = False

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    while True:
        for batch in train_dataset.get_tqdm('cpu'):
            data_x1, mask_x1, data_x2, mask_x2, data_y = batch
            opt = model(data_x1, mask_x1, data_x2, mask_x2)
            loss = loss_fn(opt, data_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            predicted_all = []
            gold_all = []
            for batch in test_dataset.get_tqdm('cpu', False):
                data_x1, mask_x1, data_x2, mask_x2, data_y = batch
                opt = model(data_x1, mask_x1, data_x2, mask_x2)
                predicted = torch.argmax(opt, -1)
                predicted = list(predicted.cpu().numpy())
                predicted_all += predicted

                gold = list(data_y.cpu().numpy())
                gold_all += gold

            p, r, f = compute_f1(gold_all, predicted_all)
            print(p, r, f)