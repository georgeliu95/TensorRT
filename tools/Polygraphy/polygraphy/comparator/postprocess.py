import numpy as np


class PostprocessFunc(object):
    """
    Provides functions that can apply post-processing to `IterationResult` s.
    """

    @staticmethod
    # This function returns a top_k function that can be used as a postprocess_func.
    def topk_func(k=10, axis=1, exclude_outputs=set()):
        """
        Creates a function that applies a Top-K operation to a IterationResult.
        Top-K will return the indices of the k largest values in the array.

        Args:
            k (int): The number of indices to keep. Defaults to 10.
            axis (int): The axis along which to apply the topk. Defaults to 1.
            exclude_outputs (Set[str]):
                    Names of outputs to exclude. Top-K will not be applied to these outputs.

        Returns:
            Callable(IterationResult) -> IterationResult: The top-k function.
        """
        # Top-K implementation.
        def topk(run_result):
            for name, output in run_result.items():
                if name not in exclude_outputs:
                    indices = np.argsort(-output, axis=axis)
                    run_result[name] = np.take(indices, np.arange(0, k), axis=axis)
            return run_result
        return topk
