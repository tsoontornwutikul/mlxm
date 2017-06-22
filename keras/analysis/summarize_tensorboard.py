import numpy as np
import os
from ... import experiments
from ...analysis import InPlaceAnalysis
from ...config import Config
from ...helpers import get_multiprint, load_pickle_gz, seconds_to_str
from ...helpers import tensorboard as tensorboard

class SummarizeTensorboard(InPlaceAnalysis):

    def _run(self, config, status):

        config.define('tb.key.valid_error', default=None)
        config.define('tb.key.test_error', default=None)
        
        self.log = get_multiprint(self.get_results_path('__log.txt'))
        self.log('Analyzing ' + self.experiment_path)
        child_experiments = list(experiments.get_child_experiments(self.experiment_path))

        envs = [load_pickle_gz(os.path.join(path, '__env'), raiseerror=False) for path in child_experiments]

        for i, (path, env) in enumerate(zip(child_experiments, envs)):

            if not env:
                continue

            env = Config(env)

            self.log('Analyzing sub-experiment: ' + os.path.basename(path))
            if not os.path.exists(os.path.join(path, '__completed.txt')):
                self.log('[WARNING] This experiment has not been marked as completed yet.')

            tensorboard_path = tensorboard.find_log_path(env, path)
            self.log('Using tensorboard logs in ' + tensorboard_path)
            logs = tensorboard.TensorboardLogs(tensorboard_path)

            if config.tb.key.valid_error is not None:
                key_valid_error = config.tb.key.valid_error
            else:
                key_valid_error = 'classifier/valid/prediction/act_categorical_error' if env.hybrid == 0 else \
                    'classifier/valid/categorical_error'

            if config.tb.key.test_error is not None:
                key_test_error = config.tb.key.test_error
            else:
                key_test_error = 'classifier/test/prediction/act_categorical_error' if env.hybrid == 0 else \
                    'classifier/test/categorical_error'

            _, valid_steps, valid_errors = logs.get_scalars(key_valid_error)
            test_walltime, test_steps, test_errors = logs.get_scalars(key_test_error)
            assert np.array_equal(test_steps, valid_steps)

            min_valid_error_idx = np.argmin(valid_errors)
            min_test_error_idx = np.argmin(test_errors)
            self.log('\tLowest validation error: {:.2%} (step #{})'.format(valid_errors[min_valid_error_idx], valid_steps[min_valid_error_idx]))
            self.log('\tFinal validation error: {:.2%} (step #{})'.format(valid_errors[-1], valid_steps[-1]))
            self.log('\tTest error corresponding to lowest validation error: {:.2%} (step #{})'.format(
                test_errors[min_valid_error_idx], test_steps[min_valid_error_idx]))
            self.log('\tLowest test error: {:.2%} (step #{})'.format(test_errors[min_test_error_idx], test_steps[min_test_error_idx]))
            self.log('\tFinal test error: {:.2%} (step #{})'.format(test_errors[-1], test_steps[-1]))
            self.log('\tTotal time: {}'.format(seconds_to_str(test_walltime[-1]-test_walltime[0])))

    def _teardown(self, config, status, success):
        if success:
            self.save_metadata()

if __name__ == '__main__':
    SummarizeTensorboard().run()