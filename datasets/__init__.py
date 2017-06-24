import abc
import itertools
import numpy as np
from ..helpers import to_categorical

class Dataset(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create_batch_iterator(self, which, batch_size, input_parts=None, output_parts=None):
        """Creates a batch iterator for the dataset.

        # Parameters
            which: str. The subset of the dataset to use. Usually, this is one of "train", "valid", or "test".
            batch_size: int or BatchSizeScheduler. The batch size or a scheduler for batch size.
            input_parts: tuple of int (optional). The parts to be returned as inputs. 
                         If not specified, it defaults to `(parts['input'])`.
            output_parts: tuple of int (optional). The parts to be returned as outputs.
                          If not specified, it defaults to `(parts['label-onehot'])`.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def get_full_batch(self, which, batch_size_limit=None, input_parts=None, output_parts=None):
        """Gets the full batch of the dataset.

        # Parameters
            which: str. The subset of the dataset to use. Usually, this is one of "train", "valid", or "test".
            batch_size_limit: int (optional). The maximum size of the batch, used in case of an infinite dataset.
            input_parts: tuple of int (optional). The parts to be returned as inputs. 
                         If not specified, it defaults to `(parts['input'])`.
            output_parts: tuple of int (optional). The parts to be returned as outputs.
                          If not specified, it defaults to `(parts['label-onehot'])`.
        """

        raise NotImplementedError()

    @staticmethod
    def flatten(*xs):
        return tuple(x.reshape(x.shape[0], -1) for x in xs)

    @staticmethod
    def slice_train_valid(train_data, n_train, n_valid):

        if n_train == 0:
            n_train = train_data[0].shape[0] - n_valid

        valid_data = tuple(part[-n_valid:] if n_valid > 0 else np.empty((0,*part.shape[1:])) for part in train_data)
        train_data = tuple(part[:n_train] if n_train > 0 else part for part in train_data)
        return train_data, valid_data

    @staticmethod
    def make_onehot(n_classes, *ys):
        return tuple(to_categorical(y, n_classes) for y in ys)

    @staticmethod
    def shuffle(*xy, seed=None):

        idx = np.arange(xy[0].shape[0])
        random = np.random.RandomState(seed)
        random.shuffle(idx)
        return tuple(part[idx] for part in xy)

class InMemoryDataset(Dataset):

    def __init__(self, config, data, parts=None):
        super().__init__()

        self.config = config
        self.data = data
        self.data_shape = {k:v[0].shape for k,v in self.data.items()}
        self.parts = parts or {'input':0, 'label':1, 'label-onehot':2}

        self.config.dataset.n_train = self.data_shape['train'][0]
        self.config.dataset.n_valid = self.data_shape['valid'][0]
        self.config.dataset.n_test = self.data_shape['test'][0]

        print('Train data:', self.config.dataset.n_train)
        print('Validation data:', self.config.dataset.n_valid)
        print('Test data:', self.config.dataset.n_test)

    @staticmethod
    def create_standard_data(config, n_classes, train_xy, test_xy=None, 
                             n_train=None, n_valid=None, auto_flatten=False, auto_cv=False):
        """Transforms raw train and test data into the format used by this class.

        # Parameters
            config: Config. The configuration data to use when auto_flatten or auto_cv is used.
            n_classes: int. The number of classes used to transform labels to the one-hot format.
            train_xy: tuple of Array. Train input-label tuple.
            test_xy: tuple of Array (optional). Test input-label tuple.
                If not specified, test data is taken from a cross-validation fold instead.
                Supply this if not doing cross-validation.
            n_train: int (optional). The maximum number of train data to use.
                All available data is used if not specified or is zero.
            n_valid: int (optional). The number of validation data to use.
                This value is not used when doing cross-validation.
                In this case, the number of validation data equals that of one fold.
            auto_flatten: bool (default=False). Set to True to flatten all inputs if config specifies so.
            auto_cv: bool (default=False). Set to True to create cross-validation folds if config specifies so.

        # Returns
            dict of tuple containing "train", "valid", and "test" data tuples with "input", "label", "label-onehot" parts

        # Notes
            Call this before any kind of standardization across both train and test data.

        """

        config.define('dataset.cv.folds', type=int, default=0)
        config.define('dataset.flatten', type=int, default=0)

        if auto_flatten and config.dataset.flatten:

            train_xy = (*Dataset.flatten(train_xy[0]), *train_xy[1:])
            if test_xy is not None:
                test_xy = (*Dataset.flatten(test_xy[0]), *test_xy[1:])

        if auto_cv and config.dataset.cv.folds > 1:

            config.define('dataset.cv.index', type=int, default=config.get('iteration',1))

            data = InMemoryDataset.create_cv_data(
                train_xy, test_xy, config.dataset.cv.folds, config.dataset.cv.index)

            # NOTE: All validation data is used if doing cross-validation.

        else:

            train_xy, valid_xy = Dataset.slice_train_valid(train_xy, n_train, n_valid)
            data = {'train': train_xy, 'valid': valid_xy, 'test': test_xy}

            if n_valid is not None and n_valid > 0:
                data['valid'] = tuple(part[:n_valid] for part in data['valid'])

        # Use only specified amount of training data.
        if n_train is not None and n_train > 0:
            data['train'] = tuple(part[:n_train] for part in data['train'])

        data['train'] = tuple((*data['train'], *Dataset.make_onehot(n_classes, data['train'][-1])))
        data['valid'] = tuple((*data['valid'], *Dataset.make_onehot(n_classes, data['valid'][-1])))
        data['test'] = tuple((*data['test'], *Dataset.make_onehot(n_classes, data['test'][-1])))

        return data

    @staticmethod
    def create_cv_data(train_data, test_data=None, cv_folds=5, cv_index=1):

        cv_index -= 1  # Make zero-based.

        N = train_data[0].shape[0]
        valid_index = cv_index
        test_index = (cv_index+1) % cv_folds if not test_data else None
        train_index = (cv_index+2) % cv_folds if not test_data else (cv_index+1) % cv_folds
        train_folds = cv_folds-2 if not test_data else cv_folds-1
        valid_slice = slice(N//cv_folds*valid_index, N//cv_folds*(valid_index+1))
        test_slice = slice(N//cv_folds*test_index, N//cv_folds*(test_index+1)) if not test_data else None

        valid_data = (part[valid_slice] for part in train_data)
        test_data = (part[test_slice] for part in train_data) if not test_data else test_data

        if train_index + train_folds <= cv_folds:
            train_slice = slice(N//cv_folds*train_index, N//cv_folds*(train_index+train_folds))
            train_data = (part[train_slice] for part in train_data)
            print('Train data slice: {}'.format(train_slice))
        else:
            train_slices = (slice(N//cv_folds*train_index, N//cv_folds*cv_folds), 
                            slice(N//cv_folds*(train_folds-(cv_folds-train_index))))
            train_data = (np.concatenate((part[train_slices[0]],part[train_slices[1]])) for part in train_data)
            print('Train data slices: {}, {}'.format(train_slices[0],train_slices[1]))

        return {'train': tuple(train_data), 'valid': tuple(valid_data), 'test': tuple(test_data)}


    def create_batch_iterator(self, which, batch_size, input_parts=None, output_parts=None, 
                              first_batch=0, infinite=True, cutoff=False):

        if type(batch_size) == int:
            batch_size = FixedBatchSizeScheduler(batch_size)
        batch_size = batch_size.create_batch_size_iterator()

        data, data_shape = self.data[which], self.data_shape[which]
        
        input_parts = input_parts or (self.parts['input'],)
        output_parts = output_parts or (self.parts['label-onehot'],)

        input_data = [data[p] for p in input_parts]
        output_data = [data[p] for p in output_parts]
        
        i = sum(next(batch_size) for _ in range(first_batch)) % data_shape[0]
        
        while True:
            current_batch_size = next(batch_size)
            j = i + current_batch_size
            if j <= data_shape[0]:
                yield ([part[i:j] for part in input_data], [part[i:j] for part in output_data])
                i = j % data_shape[0]
            elif not infinite:
                break
            elif cutoff:
                i = 0
            else:
                j = j % data_shape[0]
                yield ([np.concatenate((part[i:], part[:j])) for part in input_data],
                       [np.concatenate((part[i:], part[:j])) for part in output_data])
                i = j

    def get_full_batch(self, which, input_parts=None, output_parts=None, batch_size_limit=None):

        if batch_size_limit is not None and batch_size_limit == 0:
            return np.empty((0,*self.data_shape[which][1:]))

        else:
            batch_size_limit = batch_size_limit or self.data_shape[which][0]
            iterator = self.create_batch_iterator(which, batch_size_limit, input_parts, output_parts)
            return next(iterator)

class BatchSizeScheduler(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def create_batch_size_iterator(self):
        """Creates a batch size iterator.

        # Returns
            An iterator yielding a batch size for each batch infinitely.
        """
        
        raise NotImplementedError()

class FixedBatchSizeScheduler(BatchSizeScheduler):

    def __init__(self, batch_size):
        if batch_size < 1:
            raise ValueError('Batch size cannot be less than 1.')
        self.batch_size = batch_size

    def create_batch_size_iterator(self):
        yield from itertools.repeat(self.batch_size)

class BatchSizeListScheduler(BatchSizeScheduler):

    def __init__(self, batch_sizes):
        self.batch_sizes = batch_sizes

    def create_batch_size_iterator(self):
        yield from self.batch_sizes
        yield from itertools.repeat(self.batch_sizes[-1])