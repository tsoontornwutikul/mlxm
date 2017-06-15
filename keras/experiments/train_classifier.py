from keras.models import Model
from ...helpers import categorical_error, seconds_to_str, save_pickle_gz, Timer
from ..callbacks import BatchEndCallback, TrainingHistory
from . import BaseTrainClassifierExperiment

class TrainClassifierExperiment(BaseTrainClassifierExperiment):

    def _setup(self, config, status):
        super()._setup(config, status)

        self.classifier = Model(inputs=self.model.input, outputs=self.model.outputs[0])
        self.compile_classifier(config, self.classifier)
        self.classifier.summary()

        self.best_checkpoint_callback = self.create_best_checkpoint_callback()
        self.callbacks_classifier = [self.best_checkpoint_callback]
        
        self.status.timer = Timer(config.train.batches.cls)
        self.status.evaluation_time = 0.

        if config.history.enabled:
            self._setup_history(config, status)

        self.save_metadata()

    def _setup_history(self, config, status):

        _models, _test_data = {}, {}

        if config.history.track.cls:
            if config.dataset.n_valid > 0:
                _models['classifier/valid'] = self.classifier
                _test_data['classifier/valid'] = self.ds.get_full_batch(
                'valid', (self.ds.parts['input'],), (self.ds.parts['label-onehot'],), config.dataset.n_valid)
            _models['classifier/test'] = self.classifier
            _test_data['classifier/test'] = self.ds.get_full_batch(
                'test', (self.ds.parts['input'],), (self.ds.parts['label-onehot'],), config.dataset.n_test)

        def finished_batch_callback(experiment, batch):
            experiment.status.trained_batches += 1

        def should_evaluate_predicate(experiment, batch):
            return ((experiment.config.valid.every.cls > 0 and 
                     experiment.status.trained_batches % experiment.config.valid.every.cls == 0) or 
                     experiment.status.trained_batches == experiment.config.train.batches.cls)

        def report_callback(experiment, batch):
            if should_evaluate_predicate(experiment, batch):
                experiment.log('Trained on {} batches, {} remaining...'.format(
                               experiment.status.trained_batches,
                               seconds_to_str(experiment.status.timer.remaining(experiment.status.trained_batches))))

        def save_history_callback(experiment, batch):
            if (experiment.status.trained_batches % config.history.save.every == 0 or
                experiment.status.trained_batches == experiment.config.train.batches.cls):
                    experiment.log('Saving history after {} batches...'.format(experiment.status.trained_batches))
                    experiment.save_history(experiment.status.history)
                    experiment.save_keras_history(experiment.status.history_train_classifier, 'history-train-classifier')

        finished_batch_callback = BatchEndCallback(self, finished_batch_callback)
        report_callback = BatchEndCallback(self, report_callback)
        save_history_callback = BatchEndCallback(self, save_history_callback)

        status.history = TrainingHistory(self, _models, _test_data, evaluate_batch_size=config.valid.batch_size, 
                                         verbose=config.verbose, should_evaluate_predicate=should_evaluate_predicate)
        status.history.setup_tensorboard(weight_histogram=config.history.histogram.weight, 
                                         output_histogram=config.history.histogram.output, 
                                         histogram_models=config.history.histogram.models)

        status.history_train_classifier = []
        self.callbacks_classifier.extend([finished_batch_callback, status.history, report_callback, save_history_callback])

    def _run(self, config, status):
        super()._run(config, status)

        # Record model performance before any training.
        status.history.on_batch_end(0)

        self.log('Training classifier...')
        self.ds_classifier_iterator = self.ds.create_batch_iterator(which='train', batch_size=config.train.batch_size)
        h = self.classifier.fit_generator(self.ds_classifier_iterator, steps_per_epoch=1, epochs=config.train.batches.cls,
                                          callbacks=self.callbacks_classifier, verbose=config.verbose)
        status.history_train_classifier.append(h)

    def _teardown(self, config, status, success):
        super()._teardown(config, status, success)
        
        if success:

            if config.history.enabled:
                self.save_history(status.history)
                self.save_keras_history(status.history_train_classifier, 'history-train-classifier')
                status.history.close_tensorboard()

            self.log('Completed training in {} ({} used in validation).'.format(
                     seconds_to_str(status.timer.elapsed()), 
                     seconds_to_str(status.evaluation_time)))

if __name__ == '__main__':
    TrainClassifierExperiment().run()