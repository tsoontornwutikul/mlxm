from keras.optimizers import Adadelta

def create(config):

    config.define('optimizer.adadelta.lr', type=float, default=1e-1)
    config.define('optimizer.adadelta.rho', type=float, default=0.95)
    config.define('optimizer.adadelta.epsilon', type=float, default=1e-8)
    config.define('optimizer.adadelta.decay', type=float, default=0.)

    return Adadelta(config.optimizer.adadelta.lr, config.optimizer.adadelta.rho, 
                    config.optimizer.adadelta.epsilon, config.optimizer.adadelta.decay)