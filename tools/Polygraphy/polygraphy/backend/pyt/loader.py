from polygraphy.backend.base import BaseLoadModel

class BaseLoadPyt(BaseLoadModel):
    def __call__(self):
        """
        Returns a torch.nn.Module or subclass.
        """
        raise NotImplementedError("PyTorch model loaders must be implemented on a per-model basis")
