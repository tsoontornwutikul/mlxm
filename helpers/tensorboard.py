import glob
import numpy as np
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from . import get_first_existing_path, get_nth_matching_path
from ..experiments import Experiment

class TensorboardLogs(object):

    def __init__(self, path):
        self.path = path
        self.ea = EventAccumulator(self.path)
        self.ea.Reload()

    def get_scalars(self, name):
        events = self.ea.Scalars(name)
        scalars = np.array([(event.wall_time, event.step, event.value) for event in events])
        return (scalars[:,0], scalars[:,1].astype('int'), scalars[:,2])

def find_log_path(config, main_path=None):

    config.define('path.result.main.base', 'path.result.base', default='')
    config.define('path.result.main.relative', 'path.result.relative', default='')
    config.define('path.result.tensorboard.base', 'path.result.base.tensorboard', default='')
    config.define('path.result.tensorboard.relative', 'path.result.relative.tensorboard', default='')

    candidates = [os.path.join(config('path.result.tensorboard.base'), config('path.result.tensorboard.relative')),
                  os.path.join(config('path.result.main.base').replace('experiment', 'experiment-tb'), config('path.result.tensorboard.relative')),
                  os.path.join(Experiment.DEFAULT_TENSORBOARD_ROOT, config('path.result.tensorboard.relative')),
                  get_nth_matching_path(os.path.join(config('path.result.tensorboard.base'), config('path.result.main.relative')) + '@*', -1, ''),
                  get_nth_matching_path(os.path.join(config('path.result.main.base').replace('experiment', 'experiment-tb'), config('path.result.main.relative')) + '@*', -1, ''),
                  get_nth_matching_path(os.path.join(Experiment.DEFAULT_TENSORBOARD_ROOT, config('path.result.main.relative')) + '@*', -1, '')]

    if main_path:
        candidates.append(get_nth_matching_path(glob.escape(main_path.replace('experiment','experiment-tb')) + '@*', -1, ''))

    path = get_first_existing_path(*candidates)
    if not path:
        raise FileNotFoundError('Tensorboard log directory is not found.')

    return path