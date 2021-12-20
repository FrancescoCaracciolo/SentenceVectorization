import numpy as np
#  from numpy import linalg as lg
import os


class SentVE:
    def __init__(self):
        super().__init__()
        self.vectors = {}

    def load_model_from_file(self, filename: str, limit: int = None) -> bool:
        if not os.path.exists(filename):
            return False
        loaded = 0
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                val = line.split()
                word = val[0]
                try:
                    vector = np.asarray(val[1:], "float32")
                    self.vectors[word] = vector
                except Exception:
                    pass
                else:
                    loaded += 1
                    if limit is not None and loaded >= limit:
                        return True
        return True

    def sum_vectors(self, *vectors: np.ndarray) -> np.ndarray:
        """Returns the sum of multiple vectors

        Returns:
            np.ndarray: Resulting vector
        """
        return self.sum_vectors_from_list(vectors)

    def sum_vectors_from_list(self, vectors: list) -> np.ndarray:
        """Returns the sum of vectors in a list

        Args:
            vectors (list): List of vectors

        Returns:
            np.ndarray: Sum of vectors
        """
        result = vectors[0] - vectors[0]  # Start from the null vector
        for vector in vectors:
            result += vector
        return result
