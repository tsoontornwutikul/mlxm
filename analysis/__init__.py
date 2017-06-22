import os
import retrying
import sys
from .. import BaseAction
from .. import experiments
from ..helpers import get_first_existing_path, get_multiprint, save_pickle_gz

class Analysis(BaseAction):

    DEFAULT_RESULTS_ROOT = os.path.join('output', 'analysis')

    def _setup(self, config, status):
        
        config.define('name', default=os.path.splitext(os.path.basename(sys.argv[0]))[0])
        config.define('title', default='')

        self._init_results_directory(config)

    def _init_results_directory(self, config):

        relative_results_path = config.name
        if config.title:
            relative_results_path = os.path.join(relative_results_path, config.title)

        config.define('path.result.analysis.base', default=Analysis.DEFAULT_RESULTS_ROOT)
        config.define('path.result.analysis.relative', default=relative_results_path)

        self.results_path = os.path.join(config.path.result.analysis.base, config.path.result.analysis.relative)
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

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

        self.log('Done.')

    def get_results_path(self, filename=None):
        return os.path.join(self.results_path, filename) if filename is not None else self.results_path

class InPlaceAnalysis(Analysis):

    def _setup(self, config, status):
        
        config.require('experiment')

        self.experiment_path = get_first_existing_path(
            config.experiment,
            experiments.get_results_path(config, config.experiment))

        if not self.experiment_path:
            raise FileNotFoundError('Cannot find a valid experiment path for: ' + config.experiment)

        config.define('path.result.analysis.base', default=os.path.join(self.experiment_path, 'analysis'))

        super()._setup(config, status)