from .. import Dataset, InMemoryDataset
import gzip
import numpy as np
import os

INPUT_SHAPE = (96, 96)
INPUT_SIZE = 96*96
N_CLASSES = 5
N_TRAIN_TOTAL = 24300
N_TEST_TOTAL = 24300

def create(config):
    return SmallNorbDataset(config)

class SmallNorbDataset(InMemoryDataset):

    def __init__(self, config):
        self.config = config

        config.define('dataset.path', default='data/smallnorb')
        config.define('dataset.n_train', type=int, default=0)
        config.define('dataset.n_valid', type=int, default=0)
        config.set('dataset.n_test', 24300)

        (train_x, train_y), (test_x, test_y) = self._load(train=True), self._load(train=False)

        data = InMemoryDataset.create_standard_data(
            config, N_CLASSES, (train_x, train_y), (test_x, test_y), 
            config.dataset.n_train, config.dataset.n_valid,
            auto_flatten=True, auto_cv=True)

        super().__init__(config, data)

    def _get_dataset_path(self, *, train=True):

        base_path = self.config.dataset.path

        path = 'smallnorb-5x46789x9x18x6x2x96x96-training-{}.mat.gz' if train \
            else 'smallnorb-5x01235x9x18x6x2x96x96-testing-{}.mat.gz'

        return tuple(os.path.join(base_path, path.format(part)) for part in ['dat','cat'])

    def _load(self, *, train=True):

        DATASET_SIZE = 24300
        data_path, label_path = self._get_dataset_path(train=train)

        with gzip.open(data_path, 'rb') as fd, gzip.open(label_path, 'rb') as fl:
            fd.read(24)
            fl.read(20)
            data = (np.fromstring(fd.read(96*96*2*DATASET_SIZE), np.int8).reshape(DATASET_SIZE,2,96*96).astype('float32')+128)/255.
            labels = np.fromstring(fl.read(DATASET_SIZE*4), np.int32)
            data = data[:,0,:].squeeze()
            return data, labels