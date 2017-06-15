from keras.optimizers import SGD

def create(config):

    config.define('optimizer.sgd.lr', type=float, default=1e-2)
    config.define('optimizer.sgd.momentum', type=float, default=0.)
    config.define('optimizer.sgd.decay', type=float, default=0.)
    config.define('optimizer.sgd.nesterov', type=int, default=0)

    return SGD(config.optimizer.sgd.lr, config.optimizer.sgd.momentum, 
               config.optimizer.sgd.decay, config.optimizer.sgd.nesterov)