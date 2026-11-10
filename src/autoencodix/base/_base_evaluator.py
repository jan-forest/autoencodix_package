import abc

from autoencodix.utils._result import Result


class BaseEvaluator(abc.ABC):
    """Interface for evaluating learned embeddings on downstream ML tasks.

    Subclasses implement ``evaluate``, which trains machine-learning models on
    latent spaces (and reference embeddings) and stores the scores in the
    ``Result``. Used by ``BasePipeline.evaluate``.

    (LLM-generated docstring, pending human review.)
    """

    @abc.abstractmethod
    def evaluate(self, *args):
        """
        Evaluate the Autoencodix pipeline on defined machine learning tasks.

        Subclasses must implement this method to perform evaluation using the provided arguments.

        Args:
            *args: Variable length argument list for evaluation parameters.

        Returns:
            Result: The evaluation result.
        """
        pass

    @staticmethod
    def _expand_reference_methods(reference_methods: list, result: Result) -> list:
        """
        Expands the list of reference methods if needed for evaluation.

        Args:
            reference_methods (list): The list of reference methods to potentially expand.
            result (Result): The evaluation result object.

        Returns:
            list: The (possibly expanded) list of reference methods.
        """
        return reference_methods
