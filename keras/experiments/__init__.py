import retrying
import keras.callbacks
from ... import experiments
from ...helpers import categorical_error, get_attribute, save_pickle_gz

class BaseTrainClassifierExperiment(experiments.Experiment):

    def _setup(self, config, status):
        super()._setup(config, status)

        config.require('dataset.name.cls', 'dataset.name', 'dataset')

        config.define('verbose', type=int, default=0)
        config.define('train.batch_size', 'batch_size', type=int, default=100)
        config.define('train.batches.cls', 'train.batches', type=int, default=1000)
        config.define('train.loss.cls', 'classifier_loss', 'train.loss', default='categorical_crossentropy')
        config.define('optimizer.cls', 'classifier_optimizer', 'optimizer', default='adam')
        config.define('valid.every.cls', 'valid.every', type=int, default=config.train.batches.cls//100)
        config.define('valid.batch_size', 'evaluate_batch_size', type=int, default=10000)
        config.define('history.enabled', 'record_history', type=int, default=1)
        config.define('history.track.cls', 'record_history_classifier', 'history.track', type=int, default=1)
        config.define('history.save.every', type=int, default=5000)
        config.define('history.histogram.weight', type=int, default=0)
        config.define('history.histogram.output', type=int, default=0)
        config.define('history.histogram.models', type=str, is_list=True, default=['classifier/test'])
        
        self.ds = self.create_dataset(config)

        status.trained_batches = 0

    def create_best_checkpoint_callback(self):
        return keras.callbacks.ModelCheckpoint(self.get_results_path('model-best.h5'), save_best_only=True)
    
    def create_dataset(self, config, module=None):
        module = module or config.dataset.name.cls
        return get_attribute(module, 'create', ['datasets','mlxm.datasets','mlxm.keras.datasets'])(config)

    def create_optimizer(self, config, module=None):
        module = module or config.optimizer.cls
        return get_attribute(module, 'create', ['optimizers','mlxm.optimizers','mlxm.keras.optimizers'])(config)
        
    def compile_classifier(self, config, model):
        loss = config.get('train.loss.cls', 'categorical_crossentropy')
        model.compile(optimizer=self.create_optimizer(config), 
                      loss=config.train.loss.cls, metrics=['acc', categorical_error])

    @retrying.retry
    def save_keras_history(self, histories, name='history-keras'):
        """Saves a list of histories returned by Keras model's fit() or fit_generator().

        # Parameters
            histories: list of History.
            name: str (optional). Name of the file to be saved.
        """

        self.log('Saving keras histories "{}" ... '.format(name), end='')

        combined = keras.callbacks.History()
        combined.epochs = []
        combined.history = {}

        for history in histories:
            for epoch in history.epoch:
                combined.epochs.append(epoch)
                for k, v in history.history.items():
                    combined.history.setdefault(k, []).extend(v)

        save_pickle_gz(combined, self.get_results_path(name))

        self.log('Done.')