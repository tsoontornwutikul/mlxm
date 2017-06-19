from keras.optimizers import RMSprop

def create(config):

    config.define('optimizer.rmsprop.lr', type=float, default=1e-3)
    config.define('optimizer.rmsprop.rho', type=float, default=0.9)
    config.define('optimizer.rmsprop.epsilon', type=float, default=1e-8)
    config.define('optimizer.rmsprop.decay', type=float, default=0.)

    return RMSprop(config.optimizer.rmsprop.lr, config.optimizer.rmsprop.rho, 
                   config.optimizer.rmsprop.epsilon, config.optimizer.rmsprop.decay)