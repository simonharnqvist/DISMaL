import numpy as np

class StochasticMatrix:
    
    def __init__(self, eigenvect_mat, inv_eigenvect_mat, eigenvals, t):
        """ Fast matrix exponentiation to produce stochastic matrix from eigenvalues and eigenvectors of transition rate matrix. Equivalent to linalg.expm(Q), but faster if eigenvalues+vectors are already available.

        Args:
            matrix (_type_): _description_
            inv_matrix (_type_): _description_
            eigenvalues (_type_): _description_
            t (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.eigenvect_mat = eigenvect_mat
        self.inv_eigenvect_mat = inv_eigenvect_mat
        self.eigenvals = eigenvals
        self.t = t
        self.matrix = self.generate()

    def __repr__(self):
        return str(self.matrix)

    def __getitem__(self, index):
        return self.matrix[index]

    def generate(self):
        return self.inv_eigenvect_mat @ np.diag(np.exp(self.eigenvals * self.t)) @ self.eigenvect_mat