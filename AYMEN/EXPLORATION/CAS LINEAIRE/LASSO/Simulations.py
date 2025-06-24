import sys, pathlib
sys.path.append(str(pathlib.Path.cwd())+"\AYMEN\EXPLORATION\CAS LINEAIRE\OUTILS")
import numpy as np
from linear_tools import generate_data, pesr, tpr, fdr, f1, plot_scores, ista, qut_lasso_oracle

class SimulationLassoOracleIsta:

    def __init__(self, n, p, list_s, sigma, simu_iter=100, qut_iter=100, max_iter=1000, tol=1e-6, seed=42, verbose=False):
        
        self.n = n
        self.p = p
        self.list_s = list_s
        self.sigma = sigma

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
        'pesr': pesr(beta, beta_hat),
        'f1': f1(beta, beta_hat),
        'tpr': tpr(beta, beta_hat),
        'fdr': fdr(beta, beta_hat)
    }

    def _generate_data(self, s, seed=False):
        if not seed: seed = self.seed
        return generate_data(self.n, self.p, s, self.sigma, seed)

    def _get_lambda(self, X, alpha=0.05, seed=False):
        if not seed: seed = self.seed
        return qut_lasso_oracle(X, self.sigma, self.qut_iter, alpha, seed)

    def _simulation(self, s):
    
        score = self._generate_score_empty()

        for i in range(self.simu_iter):
            
            y, X, beta = self._generate_data(s, seed=self.seed + i)

            _lambda = self._get_lambda(X)

            L = np.linalg.norm(X.T @ X, ord=2) / self.n
            beta_hat = np.zeros(self.p)

            grad_f = lambda x: -x.T (y - x @ beta_hat)

            prox_g = lambda z: np.sign(z) * np.max(np.abs(z)-_lambda/L, 0)

            beta_hat = ista(grad_f, prox_g, [], beta_hat, L, self.max_iter, self.tol)

            score_tmp = self._get_score(beta, beta_hat)

            for key in score_tmp:
                score[key].append(score_tmp[key])

            if self.verbose:
                print(f"Simulation {i+1}/{self.simu_iter} avec s = {s} :")
                #print(f"\t beta : {beta}")
                #print(f"\t beta estimé : {beta_hat}")
                print(f"\t Score : {score}")

    def run(self):

        for s in self.list_s:

            score_tmp = self._simulation(s)

            for key in score_tmp:
                self.score[key].append(score_tmp[key])

    def plot(self):

        plot_scores(self.score, self.list_s, "LASSO ORACLE ISTA")

Sim = SimulationLassoOracleIsta(
    n = 25,
    p = 50,
    list_s = [1, 2, 3],
    sigma = 0.0,
    simu_iter=10,
    qut_iter=100,
    max_iter=100,
    verbose=True
)

Sim.run()

Sim.plot()
