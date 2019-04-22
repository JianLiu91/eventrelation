from dataset import Dataset
from model import BasicCausalModel


if __name__ == '__main__':
    dataset = Dataset(10)
    model = BasicCausalModel(10000, 300, 100, 3)
    for batch in dataset.reader('cpu', True):
        data_x1, mask_x1, data_x2, mask_x2, data_y = batch
        opt = model(data_x1, mask_x1, data_x2, mask_x2)
        print(opt.size())
        break
