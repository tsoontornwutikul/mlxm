from keras import backend as K
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout
from keras.models import Model, Sequential
from keras.constraints import maxnorm
from keras.regularizers import l1, l2, l1_l2

def create(config, input_size, n_classes):

    """
    Creates a fully-connected feed-forward network.

    :param env: An EnvironmentDictionary containing the following optional arguments:
        model.activation.default : the default activation function to be applied after each layer
        model.batchnorm          : set to 1 to include batch normalization layers before activation function
        model.dropinput          : the percentage of the inputs to be set to zero
        model.dropout            : the percentage of dropout on intermediate layers
        model.initializer.weight : the initialization method for the weights
        model.L1.weight          : the factor of L1 regularization on the weights
        model.L2.weight          : the factor of L2 regularization on the weights
        model.maxnorm.weight     : the maximum norm of the weights
        model.n_units            : the number of hidden units for each layer
        
    """
    
    config.require('model.n_units', type=int, is_list=True)
    config.define('model.activation.default', 'model.activation', default='relu')
    config.define('model.initializer.weight', 'model.initializer', default='glorot_uniform')
    config.define('model.L1.weight', 'model.L1', type=float, default=0.)
    config.define('model.L2.weight', 'model.L2', type=float, default=0.)
    config.define('model.maxnorm.weight', 'model.maxnorm', type=float, default=0.)
    config.define('model.dropinput', type=float, default=0.)
    config.define('model.dropout', type=float, default=0.)
    config.define('model.batchnorm', 'model.bn', type=int, default=0)

    weight_regularizer = l1_l2(config.model.L1.weight, config.model.L2.weight)
    weight_constraint = maxnorm(config.model.maxnorm.weight) if config.model.maxnorm.weight > 0. else None
    
    y = x = Input((input_size,), name='input')
    if config.model.dropinput > 0.:
        y = Dropout(config.model.dropinput, name='input/dropinput')(y)
    for i, n_unit in enumerate(config.model.n_units):
        y = Dense(n_unit, use_bias=(not config.model.batchnorm),
                  kernel_initializer=config.model.initializer.weight, 
                  kernel_regularizer=weight_regularizer,
                  kernel_constraint=weight_constraint,
                  name='classifier/{}/hidden'.format(i+1))(y)
        if config.model.batchnorm:
            y = BatchNormalization(name='classifier/{}/bn'.format(i+1))(y)
        y = Activation(config.model.activation.default, name='classifier/{}/act'.format(i+1))(y)
        if config.model.dropout > 0.:
            y = Dropout(config.model.dropout, name='classifier/{}/dropout'.format(i+1))(y)
    y = Dense(n_classes, use_bias=(not config.model.batchnorm),
              kernel_initializer=config.model.initializer.weight,
              kernel_regularizer=weight_regularizer,
              kernel_constraint=weight_constraint,
              name='prediction')(y)
    if config.model.batchnorm:
        y = BatchNormalization(name='prediction/bn')(y)
    y = Activation('softmax', name='prediction/act')(y)
    classifier = Model(inputs=x, outputs=y, name='classifier')

    return classifier