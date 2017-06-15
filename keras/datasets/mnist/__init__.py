from ....datasets import Dataset, InMemoryDataset
from keras.datasets import mnist

INPUT_SHAPE = (28, 28)
INPUT_SIZE = 28*28
N_CLASSES = 10
N_TRAIN_TOTAL = 60000
N_TEST_TOTAL = 10000

def create(config):
    return MnistDataset(config)

class MnistDataset(InMemoryDataset):

    def __init__(self, config):
        self.config = config

        config.define('dataset.path', default='mnist.npz')
        config.define('dataset.n_train', type=int, default=0)
        config.define('dataset.n_valid', type=int, default=0)
        config.set('dataset.n_test', 10000)

        (train_x, train_y), (test_x, test_y) = mnist.load_data(config.dataset.path)
        train_x = train_x.astype('float32')/255.
        test_x = test_x.astype('float32')/255.
        
        data = InMemoryDataset.create_standard_data(
            config, N_CLASSES, (train_x, train_y), (test_x, test_y), 
            config.dataset.n_train, config.dataset.n_valid,
            auto_flatten=True, auto_cv=True)

        super().__init__(config, data)