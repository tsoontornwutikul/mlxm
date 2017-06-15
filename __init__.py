import abc
from .config import Config

class BaseAction(object, metaclass=abc.ABCMeta):

    def __init__(self, config=None):
        
        config = config or Config.from_args()
        self.config = config
        self.status = ActionStatus()

    def run(self):
        success = False
        try:
            self._setup(self.config, self.status)
            self._run(self.config, self.status)
            success = True
        finally:
            self._teardown(self.config, self.status, success=success)

    def _setup(self, config, status):
        pass

    @abc.abstractmethod
    def _run(self, config, status):
        pass

    def _teardown(self, config, status, success):
        pass

class ActionStatus(object):
    pass