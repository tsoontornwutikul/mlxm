from keras.optimizers import Adam

def create(config):

    config.define('optimizer.adam.lr', type=float, default=1e-3)
    config.define('optimizer.adam.beta1', type=float, default=0.9)
    config.define('optimizer.adam.beta2', type=float, default=0.999)
    config.define('optimizer.adam.epsilon', type=float, default=1e-8)
    config.define('optimizer.adam.decay', type=float, default=0.)

    return Adam(config.optimizer.adam.lr, config.optimizer.adam.beta1, config.optimizer.adam.beta2, 
                config.optimizer.adam.epsilon, config.optimizer.adam.decay)