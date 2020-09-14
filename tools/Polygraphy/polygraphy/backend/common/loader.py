from polygraphy.backend.base import BaseLoadModel


class BytesFromPath(BaseLoadModel):
    def __init__(self, path):
        """
        Functor that can load a file in binary mode ('rb').

        Args:
            path (str): The file path.
        """
        self.path = path


    def __call__(self):
        """
        Loads a file in binary mode ('rb').

        Returns:
            bytes: The contents of the file.
        """
        with open(self.path, "rb") as f:
            return f.read()
