from keras.optimizers import Adamax

def create(config):

    config.define('optimizer.adamax.lr', type=float, default=2e-3)
    config.define('optimizer.adamax.beta1', type=float, default=0.9)
    config.define('optimizer.adamax.beta2', type=float, default=0.999)
    config.define('optimizer.adamax.epsilon', type=float, default=1e-8)
    config.define('optimizer.adamax.decay', type=float, default=0.)

    return Adamax(config.optimizer.adamax.lr, config.optimizer.adamax.beta1, config.optimizer.adamax.beta2, 
                  config.optimizer.adamax.epsilon, config.optimizer.adamax.decay)