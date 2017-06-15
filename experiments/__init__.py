import os
import retrying
import sys
import time
from .. import BaseAction, ActionStatus
from ..helpers import get_attribute, get_multiprint, save_pickle_gz

class BaseExperiment(BaseAction):
    pass

class MetaExperiment(BaseExperiment):
    pass

class Experiment(BaseExperiment):

    DEFAULT_RESULTS_ROOT = os.path.join('output', 'experiment')
    DEFAULT_TENSORBOARD_ROOT = os.path.join('output', 'experiment-tb')

    def _setup(self, config, status):

        config.require('name')
        config.require('model.name', 'model')
        config.define('iteration', default=None)

        config.model.name = os.path.basename(config.model.name)
        self._init_results_directory(config)
        self.model = self.create_model(config)

    def _init_results_directory(self, config):

        config.define('forced', 'force_replace', type=int, default=0)
        config.define('path.result.main.base', 'path.result.base', default=self.DEFAULT_RESULTS_ROOT)
        config.define('path.result.tensorboard.base', 'path.result.base.tensorboard', default=self.DEFAULT_TENSORBOARD_ROOT)
        
        if config.iteration is not None:
            config.name = '+' + config.name
        self.relative_results_path = os.path.join(config.model.name, config.name)
        if config.iteration is not None:
            self.relative_results_path = os.path.join(self.relative_results_path, config.iteration)
        self.relative_tensorboard_path = '{}@{}'.format(self.relative_results_path, int(time.time()))

        config.define('path.result.main.relative', 'path.result.relative', default=self.relative_results_path)
        config.define('path.result.tensorboard.relative', 'path.result.relative.tensorboard', default=self.relative_tensorboard_path)

        self.results_path = os.path.join(config.path.result.main.base, config.path.result.main.relative)
        self.tensorboard_path = os.path.join(config.path.result.tensorboard.base, config.path.result.tensorboard.relative)
        
        if os.path.exists(self.results_path):
            if not config.forced:
                raise RuntimeError('Experiment in \'{}\' already exists'.format(self.results_path))
        else:
            os.makedirs(self.results_path)
        os.makedirs(self.tensorboard_path)
            
        self.log = get_multiprint(self.get_results_path('__log.txt'))
        self.log('Results path: ' + self.results_path)
        self.log('Tensorboard log path: ' + self.tensorboard_path)


    def _teardown(self, config, status, success):

        if success:

            self.save_model()
            self.save_metadata()
            self.mark_as_completed()

    def create_model(self, config, module=None):
        module = module or config.model.name
        return get_attribute(module, 'create', ['models','mlxm.models','mlxm.keras.models'])(config)
    
    @retrying.retry
    def save_metadata(self):

        self.log('Saving metadata... ', end='')

        config_dict = self.config.to_dict()

        save_pickle_gz(sys.argv, self.get_results_path('__args'))
        save_pickle_gz(config_dict, self.get_results_path('__env'))

        with open(self.get_results_path('__args.txt'), 'w') as f:
            f.write(' '.join(sys.argv))
        with open(self.get_results_path('__env.txt'), 'w') as f:
            longest_key_length = max(len(key) for key in config_dict.keys())
            f.write('\n'.join(['{: <{width}} = {}'.format(key.upper(), config_dict[key], width=longest_key_length) for key in sorted(config_dict.keys())]))

        with open(self.get_results_path('__model.yaml'), 'w') as f:
            f.write(self.model.to_yaml())

        self.log('Done.')

    @retrying.retry
    def save_model(self, name='model-final'):

        self.log('Saving model "{}" ... '.format(name), end='')
        self.model.save(self.get_results_path(name + '.h5'))
        self.log('Done.')

    @retrying.retry
    def save_history(self, history, name='history'):

        self.log('Saving history "{}" ... '.format(name), end='')

        with open(self.get_results_path(name + '.txt'), 'w') as f:
            f.write('step')
            for model_name, metrics_names in history.metrics_names.items():
                f.write('\t' + '\t'.join(model_name + '/' + metric for metric in metrics_names))
            f.write('\n')
            for epoch, losses in enumerate(zip(*history.metrics.values())):
                f.write(str(epoch))
                for loss in losses:
                    if isinstance(loss, list):
                        f.write('\t' + '\t'.join('{:0.4f}'.format(float(L)) for L in loss))
                    else:
                        f.write('\t{:0.4f}'.format(float(loss)))
                f.write('\n')

        save_pickle_gz(history.metrics, self.get_results_path(name))
        save_pickle_gz(history.metrics_names, self.get_results_path(name + '-metrics'))

        self.log('Done.')

    @retrying.retry
    def mark_as_completed(self, message=''):
        with open(self.get_results_path('__completed.txt'), 'w') as f:
            f.write(message)

    def get_results_path(self, filename=None):
        return os.path.join(self.results_path, filename) if filename is not None else self.results_path