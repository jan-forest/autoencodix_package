import abc

from autoencodix.utils._result import Result

class BaseEvaluator(abc.ABC):
    def __init__(self):
        pass

    def evaluate(self, *args):
        pass

    @staticmethod
    def _expand_reference_methods(reference_methods: list, result: Result) -> list:
        # Blank, no expansion needed for standard evaluator
        return reference_methods
