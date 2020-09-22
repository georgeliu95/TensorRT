import numpy as np

from polygraphy.util import misc

class PostprocessFunc(object):
    """
    Provides functions that can apply post-processing to `IterationResult` s.
    """

    @staticmethod
    # This function returns a top_k function that can be used as a postprocess_func.
    def topk_func(k=10, axis=-1, outputs=None, exclude=None):
        """
        Creates a function that applies a Top-K operation to a IterationResult.
        Top-K will return the indices of the k largest values in the array.

        Args:
            k (int):
                    The number of indices to keep.
                    If this exceeds the axis length, it will be clamped.
                    Defaults to 10.
            axis (int):
                    The axis along which to apply the topk.
                    Defaults to -1.
            outputs (Sequence[str]):
                    Names of outputs to apply top-k to.
                    Defaults to all outputs.
            exclude (Sequence[str]):
                    Names of outputs to exclude. Top-K will not be applied to these outputs.

        Returns:
            Callable(IterationResult) -> IterationResult: The top-k function.
        """
        exclude = set(misc.default_value(exclude, []))

        # Top-K implementation.
        def topk(run_result):
            nonlocal outputs
            outputs = set(misc.default_value(outputs, run_result.keys()))

            for name, output in run_result.items():
                if name in outputs and name not in exclude:
                    indices = np.argsort(-output, axis=axis)
                    axis_len = indices.shape[axis]
                    run_result[name] = np.take(indices, np.arange(0, min(k, axis_len)), axis=axis)
            return run_result
        return topk
