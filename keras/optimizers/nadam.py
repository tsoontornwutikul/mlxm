from keras.optimizers import Nadam

def create(config):

    config.define('optimizer.nadam.lr', type=float, default=2e-3)
    config.define('optimizer.nadam.beta1', type=float, default=0.9)
    config.define('optimizer.nadam.beta2', type=float, default=0.999)
    config.define('optimizer.nadam.epsilon', type=float, default=1e-8)
    config.define('optimizer.nadam.decay', type=float, default=4e-3)

    return Nadam(config.optimizer.nadam.lr, config.optimizer.nadam.beta1, config.optimizer.nadam.beta2, 
                 config.optimizer.nadam.epsilon, config.optimizer.nadam.decay)