import numpy as np


class CEM:
    def __init__(self,
                 dim,
                 x_start,
                 init_sigma,
                 elite_frac=0.2,
                 extra_std=2.0,
                 min_std=0.01,
                 extra_decay_time=10,
                 full_cov=False):
        self.dim = dim
        self.elite_frac = elite_frac
        self.mean = x_start if x_start is not None else np.random.uniform(size=dim)
        self.cov = init_sigma**2 * np.diag(np.ones(dim))
        self.extra_std = extra_std * np.ones(dim)
        self.min_std = min_std
        self.extra_decay_time = extra_decay_time
        self.full_cov = full_cov

    def entropy(self):
        chol = np.linalg.cholesky(self.cov)
        ent = 0.5 * (2 * np.sum(np.log(np.diag(chol))) + self.dim * np.log(2 * np.pi) + self.dim)
        return ent

    def ask(self, batch_size):
        thetas = np.random.multivariate_normal(
            mean=self.mean,
            cov=self.cov,
            size=batch_size
        )

        return thetas

    def tell(self, i, thetas, rewards):

        batch_size = len(rewards)
        num_elite = int(batch_size * self.elite_frac)
        # if num_elite < self.dim:
        #     num_elite = self.dim + 1
        indices = np.argsort(rewards, axis=0)[-num_elite:].flatten()

        elites = thetas[indices]

        means = elites.mean(axis=0)

        if self.full_cov:
            sample_cov = np.cov(elites, rowvar=False)
            extra_cov = np.diag(max(1.0 - i / self.extra_decay_time, self.min_std) * self.extra_std ** 2)
            cov = sample_cov + extra_cov
        else:
            stds = elites.std(axis=0)
            extra_var_mult = max(1.0 - i / self.extra_decay_time, self.min_std)
            sample_std = np.square(stds) + np.square(self.extra_std) * extra_var_mult

            cov = np.diag(sample_std)
        # cov_test = 1 / num_elite * (elites - means).T @ (elites - means)

        self.mean = means
        self.cov = cov
