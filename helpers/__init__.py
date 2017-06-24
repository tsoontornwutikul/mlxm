import datetime
import glob
import gzip
import numpy as np
import os
import pickle
import sys
import time
from importlib import import_module

attribute_cache = dict()

def seconds_to_str(seconds):
    """Formats the specified number of seconds to a string.

    # Parameters
        seconds: int or float. The number of seconds. Fractions of a second are ignored.
    """
    return str(datetime.timedelta(seconds=int(seconds)))

def categorical_error(y_true, y_pred):
    """Defines the categorical error metric for Keras."""

    from keras import metrics
    return 1. - metrics.categorical_accuracy(y_true, y_pred)

def get_filtered_argv(*keys, argv=None):
    """Gets the argument list without the specified keys.

    # Parameters
        keys: list. The list of argument keys to be removed.

        All keys are case-insensitive.
        Each key can be in either of the following formats (without quotes):
         - "key_name": remove the key with the exact name "key_name" if it exists
         - "key_name.": remove all keys starting with "key_name."

        argv: list (optional). The argument list. If not specified, `sys.argv[1:]` is used.
    """

    if not argv:
        argv = sys.argv[1:]

    return [_ for _ in sys.argv[1:] \
        if all(not _.lower().startswith(key + '=') and (not key.endswith('.') or not _.lower().startswith(key)) \
        for key in keys)]

def get_multiprint(logfile, append=False):
    """Gets a method used to print to both the standard output and the specified log file.

    # Parameters
        logfile: str. The filename of the log file.
        append: bool (optional). Set to true to append to the log file instead of replacing.
    """

    f = open(logfile, 'a' if append else 'w')
    def multiprint(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=f)
        f.flush()

    return multiprint

def save_pickle_gz(obj, filename, protocol=None):
    """Pickles an object to a gzip file.

    # Parameters
        filename: str. The filename of the pickle file. The .pkl.gz extension is added automatically.
        protocol: int (optional). The pickle protocol to use.
    """

    if not filename.lower().endswith('.gz'):
        if not filename.lower().endswith('.pkl'):
            filename += '.pkl'
        filename += '.gz'

    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def load_pickle_gz(filename, protocol=None, raiseerror=True):

    if not filename.lower().endswith('.gz'):
        if not filename.lower().endswith('.pkl'):
            filename += '.pkl'
        filename += '.gz'

    try:
        with gzip.open(filename, 'rb') as f:
            return pickle.load(f)
    except:
        print('[WARNING] Unpickle failed: {}'.format(filename))
        if raiseerror:
            raise
        else:
            return None

def get_attribute(module, name, packages=None, use_cache=True):
    """Gets a named attribute of a module.

    # Parameters
        module: str. Module name.
        name: str. Attribute name.
        packages: list of str (optional). Packages to be searched in, ordered by priority.
    """

    if use_cache and (module,name,tuple(packages)) in attribute_cache:
        return attribute_cache[(module,name,tuple(packages))]

    packages = packages or []
    packages.insert(0, None)

    print('Finding {}.{} ...'.format(module, name))

    for package in packages:
        try:
            attr = getattr(import_module((package + '.' if package else '') + module), name)
            attribute_cache[(module, name)] = attr
            print('  Found in {}'.format(package if package else '<root>'))
            return attr
        except (ImportError, NameError, AttributeError):
            print('  Not found in {}'.format(package if package else '<root>'))

    raise AttributeError(name)

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.

    Borrowed from keras.utils.np_utils to avoid strong dependency.

    # Arguments
        y: list or Array. Class vector to be converted into a matrix (integers from 0 to num_classes).
        num_classes: int. Total number of classes.

    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

def get_first_existing_path(*paths, default=None):
    return next(filter(lambda path: path and os.path.exists(path), paths), default)

def get_nth_matching_path(path_pattern, nth=0, default=None):
    matched_paths = glob.glob(path_pattern)
    return matched_paths[nth] if matched_paths else default

class Timer(object):

    def __init__(self, total_steps):
        self.time_start = time.time()
        self.total_steps = total_steps

    def elapsed(self):
        return time.time()-self.time_start

    def remaining(self, elapsed_steps):
        return self.elapsed()/elapsed_steps*(self.total_steps-elapsed_steps)