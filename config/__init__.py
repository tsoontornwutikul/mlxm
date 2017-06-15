import argparse

class TreatAsError(object): pass

class Config(object):

    def __init__(self, data=None, datatype=None, keyaliases=None, prefix=None):
        self.__dict__['data'] = data or dict()
        self.__dict__['datatype'] = datatype or dict()
        self.__dict__['keyaliases'] = keyaliases or dict()
        self.__dict__['prefix'] = prefix

    def __call__(self, key, default=None):
        return self.get(key, default)

    def __getattr__(self, key):
        return self.get(key)

    def __getitem__(self, key):
        return self.get(key)

    def __setattr__(self, key, value):
        return self.set(key, value)

    def __setitem__(self, key, value):
        return self.set(key, value)

    def define(self, key, *aliases, type=str, is_list=False, default=None):
        """Defines a configuration key, along with its expected type and default value.

        If a value already exists for this key (e.g. as supplied by the user),
        then the value is converted to the type as specified by the type parameter.

        # Parameters
            key: str. The name of the key to define. Case-insensitive.
                This name should be the one that will be used later on in the code.
                If a prefix is set, it will be defined relative to that prefix.
            aliases: list of str (optional). The list of aliases for the key.
                This can be used to define alternative names to the key,
                each of which could be a shorter version of the key or 
                the old key names used in previous verions for backward-compatibility.
            type: callable object. The expected type of the value.
                This can be one of the primitive types, such as int, float or str.
                When a value is set for this key, `type(value)` will be used as the new value.
                Do not specify the list type here. Instead, set is_list to True.
            is_list: bool. Whether the expected value will be a list of the specified type.
                Setting this to True will allow comma-separated strings to be converted
                to a list of the specified type when they are set as the value of this key.
                For example, if key is "data", type is int and is_list is set to True,
                then when the user specifies "data=1,2,3", the value [1,2,3] will be used.
            default: object. The default value to use if this key has not already been set.
        """

        key = self.normalize_key(key)
        for alias in aliases:
            alias = self.normalize_key(alias)
            if alias in self.data:
                self.data[key] = self.data.pop(alias)
            self.keyaliases[alias] = key
        self.datatype[key] = self.list_of(type) if is_list else type
        self.set(key, self.get(key) if key in self.data.keys() else default)

    def require(self, key, *aliases, type=str, is_list=False):
        """Defines a configuration key to be supplied by the user, along with its expected type.

        A KeyError will be raised if a value has not already been set for the specified key.

        # Parameters
            See define().
        """

        if not any(self.expand_prefix(k) in self.data for k in (key,)+aliases):
            raise KeyError(key)
        self.define(key, *aliases, type=type, is_list=is_list)

    def get(self, key, default=TreatAsError, resolve_alias=False):
        """Gets the value corresponding to the specified key.

        If a value of the exact key does not exist, but the key is a prefix to another key,
        a Config object of the specified key prefix is returned instead.
        This allows clean-looking chaining when getting a value from a prefixed key.
        For example, instead of calling `config.get('dataset.cv.index')`, 
        you can use the equivalent `config.dataset.cv.index` to get the same value.
        This trick can also be used when setting a new value to a prefixed key 
        (i.e. `config.dataset.cv.index = 1` instead of `config.set('dataset.cv.index', 1)`).
        However, if another key is defined with the same name as the prefix (e.g. "dataset" or "dataset.cv"),
        then this chaining trick will not work and you will have to fall back to get().

        # Parameters
            key: str. The name of the key to get value from.
                If a prefix is set, the name should be relative to that prefix.
            default: object (optional). The default value if this key has not already been set nor defined.
                If this is not set, a KeyError will be raised instead.
            resolve_alias: bool. Whether to treat the key as a potential alias of another key 
                when searching for its corresponding value.

        # Returns
            The value corresponding to the key, or a Config object with the specified prefix.
        """

        key = self.normalize_key(key, resolve_alias=resolve_alias)
        if key not in self.data:
            if any(name.startswith(key + '.') for name in self.data):
                return Config(self.data, self.datatype, self.keyaliases, key)
            if default is TreatAsError:
                raise KeyError(key)
            return default
        return self.data[key]

    def set(self, key, value):
        """Sets a value to the specified key.

        # Parameters
            key: str. The name of the key to set a value to.
                If a prefix is set, the name should be relative to that prefix.
            value: object. The value to be set.
                The value will automatically be transformed to the defined type.
                If the value is None or the key has not been defined yet, no transformation will be applied.
        """

        key = self.normalize_key(key)
        self.data[key] = self.datatype.get(key, lambda _:_)(value) if value is not None else None

    def expand_prefix(self, key):
        key = key.lower()
        return self.prefix + '.' + key if self.prefix else key
    
    def normalize_key(self, key, resolve_alias=True):
        key = self.expand_prefix(key)
        return self.keyaliases.get(key, key) if resolve_alias else key

    def to_dict(self):
        """Converts this object to a dict.
        If a prefix is set, only keys with the prefix will be included.
        """
        if not self.prefix:
            return self.data.copy()
        return dict((k,v) for k,v in self.data.items() if k.startswith(self.prefix + '.'))

    def __str__(self):
        return str(self.to_dict())

    @property
    def value(self):
        return self.get('')

    @staticmethod
    def list_of(datatype, split_at=','):
        def f(input):
            if not input:
                return []
            if type(input) == str:
                return [datatype(_) for _ in input.split(split_at)]
            return input
        return f

    @classmethod
    def from_args(cls, args=None, namespace=None):
        parser = argparse.ArgumentParser()
        parser.add_argument('args', nargs='+', metavar='key=value', default=[])
        args = parser.parse_args(args, namespace)
        return cls(data=dict((arg.split('=')[0].lower(), arg.split('=')[1]) for arg in args.args))

    