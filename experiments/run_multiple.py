import os
from . import MetaExperiment
from ..helpers import get_filtered_argv

class Experiment(MetaExperiment):

    def _setup(self, config, status):

        config.require('experiment')
        config.require('n', type=int)
        config.define('n_skip', type=int, default=0)

    def _run(self, config, status):
        
        if not config.experiment.endswith('.py'):
            config.experiment = '-m ' + config.experiment

        for i in range(config.n_skip, config.n):
            filtered_argv = ['iteration=' + str(i+1)] + get_filtered_argv('experiment','n','n_skip','iteration')
            os.system('python {} {}'.format(config.experiment, ' '.join(filtered_argv)))

if __name__ == '__main__':
    Experiment().run()