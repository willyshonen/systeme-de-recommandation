"""
src/recommender.py
Classes partagées entre le notebook, train.py et l'API.
Doit être importable depuis n'importe quel contexte pour que pickle fonctionne.
"""

import numpy as np
from tqdm import tqdm


class ALSRecommender:
    """
    Alternating Least Squares pour feedback implicite.
    Implémentation numpy/scipy — compatible Python 3.13+.
    """

    def __init__(
        self,
        n_factors: int = 32,
        n_iterations: int = 15,
        alpha: int = 40,
        regularization: float = 0.1,
        random_state: int = 42,
    ):
        self.n_factors    = n_factors
        self.n_iterations = n_iterations
        self.alpha        = alpha
        self.reg          = regularization
        self.random_state = random_state

    def fit(self, user_item_matrix):
        rng = np.random.default_rng(self.random_state)
        n_users, n_items = user_item_matrix.shape

        C_ui = user_item_matrix.copy().tocsr().astype(np.float32)
        C_ui.data = 1.0 + self.alpha * C_ui.data

        self.user_factors = rng.standard_normal((n_users, self.n_factors)).astype(np.float32) * 0.01
        self.item_factors = rng.standard_normal((n_items, self.n_factors)).astype(np.float32) * 0.01

        reg_I = self.reg * np.eye(self.n_factors, dtype=np.float32)
        C_iu  = C_ui.T.tocsr()

        for _ in tqdm(range(self.n_iterations), desc="ALS"):
            YtY = self.item_factors.T @ self.item_factors
            for u in range(n_users):
                row = C_ui.getrow(u)
                if row.nnz == 0:
                    continue
                conf_u = row.data.astype(np.float32)
                Y_u    = self.item_factors[row.indices]
                A = YtY + Y_u.T @ (np.diag(conf_u - 1.0) @ Y_u) + reg_I
                b = Y_u.T @ conf_u
                self.user_factors[u] = np.linalg.solve(A, b)

            XtX = self.user_factors.T @ self.user_factors
            for i in range(n_items):
                row = C_iu.getrow(i)
                if row.nnz == 0:
                    continue
                conf_i = row.data.astype(np.float32)
                X_i    = self.user_factors[row.indices]
                A = XtX + X_i.T @ (np.diag(conf_i - 1.0) @ X_i) + reg_I
                b = X_i.T @ conf_i
                self.item_factors[i] = np.linalg.solve(A, b)
        return self

    def recommend(self, user_idx: int, n: int = 10, filter_seen=None):
        scores = self.user_factors[user_idx] @ self.item_factors.T
        if filter_seen is not None:
            scores[list(filter_seen)] = -np.inf
        return np.argsort(scores)[::-1][:n]
