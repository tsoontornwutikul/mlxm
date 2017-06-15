import sys
import time
import keras.backend as K
import keras.callbacks
import tensorflow as tf
from collections import defaultdict, Sequence

class TrainingHistory(keras.callbacks.Callback):

    def __init__(self, experiment, models, test_data, evaluate_batch_size=256, verbose=1, should_evaluate_predicate=None):
        self.experiment = experiment
        self.models = models
        self.data = test_data
        self.evaluate_batch_size = evaluate_batch_size
        self.metrics = {}
        self.metrics_names = {model_name: model.metrics_names for model_name, model in self.models.items()}
        self.verbose = verbose
        self.should_evaluate_predicate = should_evaluate_predicate
        self.tb_writer = None

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        if not self.should_evaluate_predicate or self.should_evaluate_predicate(self.experiment, batch):
            start = time.time()
            for model_name, model in self.models.items():
                metrics = model.evaluate(*self.data[model_name], batch_size=self.evaluate_batch_size, verbose=self.verbose)
                if not isinstance(metrics, Sequence):
                    metrics = [metrics]
                self.metrics.setdefault(model_name, []).append(metrics)
                if self.tb_writer:
                    self.write_tensorboard(model_name, metrics)
                    if (self.tb_weight_histogram or self.tb_output_histogram) and model_name in self.tb_histogram_models:
                        feed_dict = dict(zip(model.inputs, self.data[model_name]))
                        if model.uses_learning_phase:
                            feed_dict[K.learning_phase()] = 0
                        result = self.tb_session.run([self.tb_merged], feed_dict=feed_dict)
                        summary_str = result[0]
                        self.tb_writer.add_summary(summary_str, self.experiment.status.trained_batches)
                    self.tb_writer.flush()

                self.experiment.log('\t' + ' | '.join('{}/{}={:0.4f}'.format(model_name,k,v) for k,v in zip(self.metrics_names[model_name], metrics)))
            self.experiment.status.evaluation_time += (time.time()-start)

    def setup_tensorboard(self, weight_histogram=False, weight_image=False, output_histogram=False, histogram_models=None):
        self.tb_logdir = self.experiment.tensorboard_path
        self.tb_session = K.get_session()
        self.tb_weight_histogram = weight_histogram
        self.tb_output_histogram = output_histogram
        self.tb_weight_image = weight_image
        self.tb_histogram_models = histogram_models or []
        if self.tb_weight_histogram or self.tb_output_histogram:
            for model_name, model in self.models.items():
                if model_name in self.tb_histogram_models:
                    for layer in model.layers:
                        if self.tb_weight_histogram:
                            for weight in layer.weights:
                                tf.summary.histogram('{}/{}'.format(model_name, weight.name), weight)
                                if self.tb_weight_image:
                                    w_img = tf.squeeze(weight)
                                    shape = w_img.get_shape()
                                    if len(shape) > 1 and shape[0] > shape[1]:
                                        w_img = tf.transpose(w_img)
                                    if len(shape) == 1:
                                        w_img = tf.expand_dims(w_img, 0)
                                    w_img = tf.expand_dims(tf.expand_dims(w_img, 0), -1)
                                    tf.summary.image(weight.name, w_img)
                        if self.tb_output_histogram and hasattr(layer, 'output'):
                            tf.summary.histogram('{}/{}/output'.format(model_name, layer.name), layer.output)
        self.tb_merged = tf.summary.merge_all()
        self.tb_writer = tf.summary.FileWriter(self.tb_logdir, self.tb_session.graph)

    def write_tensorboard(self, model_name, metrics):
        for metric_name, metric in zip(self.metrics_names[model_name], metrics):
            summary = tf.Summary()
            value = summary.value.add()
            value.simple_value = metric
            value.tag = model_name + '/' + metric_name
            self.tb_writer.add_summary(summary, self.experiment.status.trained_batches)

    def close_tensorboard(self):
        self.tb_writer.flush()
        self.tb_writer.close()

class BatchEndCallback(keras.callbacks.Callback):

    def __init__(self, experiment, callback, **kwargs):
        self.experiment = experiment
        self.callback = callback
        self.kwargs = kwargs

    def on_batch_end(self, batch, logs=None):
        self.callback(self.experiment, batch, **self.kwargs)