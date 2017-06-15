import itertools
import os
from . import MetaExperiment
from ..helpers import get_filtered_argv

class Experiment(MetaExperiment):

    def _run(self, config, status):

        config.define('dryrun', type=int, default=0)
        config.define('n', type=int, default=0)

        run_multiple = (config.n >= 1)
        varying_keys = list(sorted(key for key in config.to_dict().keys() if key.startswith('~')))
        variations = dict((key, config[key].split('/')) for key in varying_keys)
        variations_product = itertools.product(*(variations[key] for key in varying_keys))

        for i, variation in enumerate(variations_product):
            if run_multiple:
                filtered_argv = get_filtered_argv('name', *varying_keys)
                cmd = 'python -m mlxm.experiments.run_multiple '
            else:
                filtered_argv = get_filtered_argv('experiment', 'name', *varying_keys)
                cmd = 'python '
                if not config.experiment.endswith('.py'):
                    cmd += '-m '
                cmd += config.experiment + ' '
            cmd += ' '.join(filtered_argv) + ' ' + ' '.join(k[1:] + '=' + v for k,v in zip(varying_keys,variation))
            cmd += ' name=' + config.name + '-' + '_'.join(variation)
            print(cmd, '\n')
            if not config.dryrun:
                os.system(cmd)

if __name__ == '__main__':
    Experiment().run()