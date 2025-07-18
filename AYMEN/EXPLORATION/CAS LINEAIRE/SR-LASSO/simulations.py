import numpy as np
from linear_tools import generate_data, pesr, tpr, fdr, f1, plot_scores, ista_backtracking, qut_square_root_lasso

class SimulationSquareRootLassoIstaBacktracking:

    def __init__(self, n, p, list_s, sigma, simu_iter=100, qut_iter=100, max_iter=1000, tol=1e-6, seed=42, verbose=False):

        self.n = n
        self.p = p
        self.list_s = list_s
        self.sigma = sigma

        self.L0 = 10
        self.beta0 = 3 # Valeurs des coefficient non nulles pour les simulations

        self.simu_iter = simu_iter
        self.qut_iter = qut_iter
        self.max_iter = max_iter

        self.tol = tol
        self.seed = seed
        np.random.seed(self.seed)
        np.set_printoptions(precision=3, suppress=True, linewidth=120)

        self.score = self._generate_score_empty()

        self.verbose = verbose

    def _generate_score_empty(self):
        return {
        'pesr': [],
        'f1': [],
        'tpr': [],
        'fdr': []
    }

    def _get_score(self, beta, beta_hat):
        return {
        'pesr': float(pesr(beta, beta_hat)),
        'f1': float(f1(beta, beta_hat)),
        'tpr': float(tpr(beta, beta_hat)),
        'fdr': float(fdr(beta, beta_hat))
    }

    def _generate_data(self, s, seed=False):
        if not seed: seed = self.seed
        return generate_data(n=self.n,
                            p=self.p, 
                            s=s,
                            sigma=self.sigma,
                            beta0=self.beta0,
                            seed=seed)

    def _get_lambda(self, X, alpha=0.05, seed=False):
        if not seed: seed = self.seed
        return qut_square_root_lasso(X=X,
                                M=self.qut_iter,
                                alpha=alpha,
                                seed=seed)
    
    def _f(self, beta, y, X):
        return np.linalg.norm(y - X @ beta, ord=2)

    def _grad_f(self, beta, y, X):
        r = y - X @ beta
        norm_r = np.linalg.norm(r, ord=2)
        if norm_r == 0:
            return np.zeros_like(beta)
        return -X.T @ r / norm_r

    def _prox_g(self, z, L, lmbda):
        return  np.sign(z) * np.maximum(np.abs(z) - lmbda / L, 0)

    def _simulation(self, s):

        score = self._generate_score_empty()

        if self.verbose:
            print(f"|{s}| Simulations pour s = {s} :")

        for i in range(self.simu_iter):

            y, X, beta = self._generate_data(s, seed = self.seed + i)

            lmbda = self._get_lambda(X)

            beta_hat = np.zeros(self.p)

            beta_hat = ista_backtracking(f=self._f,
                            grad_f=self._grad_f,
                            prox_g=self._prox_g,
                            x0=beta_hat,
                            L0=self.L0,
                            f_args=(y, X),
                            grad_f_args=(y, X),
                            prox_g_args=[lmbda],
                            max_iter=self.max_iter,
                            tol=self.tol)

            score_tmp = self._get_score(beta, beta_hat)

            for key in score_tmp:
                score[key].append(score_tmp[key])

            if self.verbose:
                print(f"\t |{i+1}| Simulation {i+1}/{self.simu_iter} :")
                #print(f"\t\t beta : {beta}")
                #print(f"\t\t beta estimé : {beta_hat}")
                print(f"\t\t Lambda : {lmbda}")
                print(f"\t\t Score :")
                for key in score_tmp:
                    print(f"\t\t\t{key} : {score_tmp[key]}")

        score_m = self._generate_score_empty()

        for key in score_m:
            score_m[key].append(float(np.mean(score[key])))

        if self.verbose:
            print(f"\t Score :")
            for key in score_m:
                print(f"\t\t{key} : {score_m[key]}")

        return score_m

    def run(self):

        for s in self.list_s:

            score_tmp = self._simulation(s)

            for key in score_tmp:
                self.score[key].append(score_tmp[key])

    def plot(self):

        plot_scores(self.score, self.list_s, f"SQUARE ROOT LASSO ISTA with backtracking σ={self.sigma}, n={self.n}, p={self.p}")
