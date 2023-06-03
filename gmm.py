import numpy as np


class GmmComponent:
    def __init__(self, pixels, size):
        # Calculate the mean of each column
        self.mean = np.mean(pixels, axis=0)

        # Compute the covariance matrix
        covariance_matrix = np.cov(pixels.T)
        # Compute the determinant of the covariance matrix
        det = np.linalg.det(covariance_matrix)

        if det <= 0:
            self.covariance_matrix = covariance_matrix + np.eye(3) * 1e-6
        else:
            self.covariance_matrix = covariance_matrix

        self.determinant_covariance_matrix = np.linalg.det(self.covariance_matrix)

        # Compute the inverse of the covariance matrix
        self.inverse_covariance_matrix = np.linalg.inv(self.covariance_matrix)

        self.weight = len(pixels) / size
        assert self.determinant_covariance_matrix > 0, "determinant_cov <= 0"
        assert self.covariance_matrix.shape == (3, 3), "covariance.shape is not (3,3)"
        assert 0 < self.weight < 1, "weight is not between 0 to 1"
        assert self.inverse_covariance_matrix.shape == (3, 3), "inverse_covariance_matrix.shape != (3,3)"

    def get_mean(self):
        return self.mean

    def get_covariance_matrix(self):
        return self.covariance_matrix

    def get_inverse_covariance_matrix(self):
        return self.inverse_covariance_matrix

    def get_determinant_covariance_matrix(self):
        return self.determinant_covariance_matrix

    def get_weight(self):
        return self.weight
