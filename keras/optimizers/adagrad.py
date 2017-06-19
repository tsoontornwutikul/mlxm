from keras.optimizers import Adagrad

def create(config):

    config.define('optimizer.adagrad.lr', type=float, default=1e-2)
    config.define('optimizer.adagrad.epsilon', type=float, default=1e-8)
    config.define('optimizer.adagrad.decay', type=float, default=0.)

    return Adagrad(config.optimizer.adagrad.lr, config.optimizer.adagrad.epsilon, config.optimizer.adagrad.decay)