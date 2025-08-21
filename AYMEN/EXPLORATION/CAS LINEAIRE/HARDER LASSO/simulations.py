import numpy as np
from linear_tools import generate_data, pesr, tpr, fdr, f1, plot_scores, ista_backtracking, qut_square_root_lasso, dichotomie, newton
class SimulationHarderLassoIstaBacktracking:

    def __init__(self, n, p, list_s, sigma, nu = None, simu_iter=100, qut_iter=100, max_iter=1000, tol=1e-6, seed=42, verbose=False):

        self.n = n
        self.p = p
        self.list_s = list_s
        self.sigma = sigma

        self.nu_vals = [nu] if nu is not None else [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]

        self.L0 = 10
        self.beta0 = 3

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
        y, X, beta = generate_data(n=self.n,
                            p=self.p, 
                            s=s,
                            sigma=self.sigma,
                            beta0=self.beta0,
                            seed=seed)
        X = X / X.std(axis=0)
        return y, X, beta

    def _get_lambda(self, X, alpha=0.05, seed=False):
        if not seed: seed = self.seed
        return qut_square_root_lasso(X=X,
                                M=self.qut_iter,
                                alpha=alpha,
                                seed=seed)
    
    def _get_lambda_nu_path(self, lmbda_max):
        path = []
        if len(self.nu_vals) == 1:
            return [(lmbda_max, self.nu_vals[0])]
        for k, nu_k in enumerate(self.nu_vals):
            ratio = np.exp(k) / (1 + np.exp(k))
            lmbda_k = ratio * lmbda_max
            path.append((lmbda_k, nu_k))
        return path
    
    def _f(self, beta, y, X):
        return np.linalg.norm(y - X @ beta, ord=2)
    
    def _g(self, beta, lmbda, nu):
        abs_beta = np.abs(beta)
        return lmbda * np.sum(abs_beta / (1 + abs_beta**(1 - nu)))

    def _grad_f(self, beta, y, X):
        r = y - X @ beta
        norm_r = np.linalg.norm(r, ord=2)
        if norm_r == 0:
            return np.zeros_like(beta)
        return -X.T @ r / norm_r

    def _get_jump(self, lmbda, nu):

        def F(k, lmbda, nu):
            return k**(2 - nu) + 2*k + k**nu + 2 * lmbda * (nu - 1)
        
        a, b = 1e-25, 500 
        return dichotomie(F, [lmbda, nu], a, b, self.tol)

    def _rho_nu_prime(self, beta, nu):
        abs_beta = np.abs(beta)
        denom = (1 + abs_beta**(1 - nu))**2
        return np.sign(beta) * (1 + nu * abs_beta**(1 - nu)) / denom

    def _drho_nu_prime(self, beta, nu):
        abs_beta = np.abs(beta)
        num = nu * (1 - nu) * abs_beta**(-nu)
        denom = (1 + abs_beta**(1 - nu))**3
        return num / denom

    def _F_beta(self, beta, z, lmbda, nu):
        return beta - z + lmbda * self._rho_nu_prime(beta, nu)

    def _F_beta_prime(self, beta, z, lmbda, nu):
        return 1 + lmbda * self._drho_nu_prime(beta, nu)

    def _prox_g(self, z, L, lmbda, nu):
        lmbda_scaled = lmbda / L

        if nu == 1.0:
            return np.sign(z) * np.maximum(np.abs(z) - lmbda_scaled, 0)

        kappa = self._get_jump(lmbda_scaled, nu)
        phi = 0.5 * kappa + lmbda_scaled / (1 + kappa**(1 - nu))

        prox = np.zeros_like(z)
        mask = np.abs(z) > phi
        
        prox[~mask] = 0.0

        for i in np.where(mask)[0]:
            prox[i] = newton(self._F_beta, self._F_beta_prime, F_args=(z[i], lmbda_scaled, nu), x0=z[i], tol=1e-6)
        return prox

    def _simulation(self, s):

        score = self._generate_score_empty()

        if self.verbose:
            print(f"|{s}| Simulations pour s = {s} :")

        for i in range(self.simu_iter):

            y, X, beta = self._generate_data(s, seed = self.seed + i)

            lmbda = 1 * self._get_lambda(X)

            beta_hat = np.zeros(self.p)
            path = self._get_lambda_nu_path(lmbda)

            for lmbda_k, nu_k in path:

                beta_hat = ista_backtracking(
                    f=self._f,
                    g=self._g,
                    grad_f=self._grad_f,
                    prox_g=self._prox_g,
                    x0=beta_hat,
                    L0=self.L0,
                    f_args=(y, X),
                    g_args=[lmbda_k, nu_k],
                    grad_f_args=(y, X),
                    prox_g_args=[lmbda_k, nu_k],
                    max_iter=self.max_iter,
                    tol=self.tol
                )

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

        plot_scores(self.score, self.list_s, f"HARDER LASSO ISTA with backtracking σ={self.sigma}, n={self.n}, p={self.p}, nu={self.nu_vals[-1]}, warm start")