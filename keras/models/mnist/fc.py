from ...models import fc
from ...datasets import mnist

def create(config):

    config.define('model.n_units', type=int, is_list=True, default=[1000,300,50])

    return fc.create(config, mnist.INPUT_SIZE, mnist.N_CLASSES)