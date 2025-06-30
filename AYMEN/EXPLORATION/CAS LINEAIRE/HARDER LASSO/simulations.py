import numpy as np
from linear_tools import generate_data, pesr, tpr, fdr, f1, plot_scores, ista_backtracking, qut_square_root_lasso, dichotomie, newton
class SimulationHarderLassoIstaBacktracking:

    def __init__(self, n, p, list_s, sigma, nu, simu_iter=100, qut_iter=100, max_iter=1000, tol=1e-6, seed=42, verbose=False):

        self.n = n
        self.p = p
        self.list_s = list_s
        self.sigma = sigma
        self.nu = nu

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

    def _get_jump(self, lmbda, nu):

        def F(k, lmbda, nu):
            return k**(2 - nu) + 2*k + k**nu + 2 * lmbda * (nu - 1)
        
        a = 1e-8
        b = 50

        return dichotomie(F, [lmbda, nu], a, b, self.tol)

    def _prox_g(self, z, L, lmbda, nu):

        lmbda_scaled = lmbda

        def F_jump(k, lmbda, nu):
            return k**(2 - nu) + 2*k + k**nu + 2 * lmbda * (nu - 1)

        kappa = dichotomie(F_jump, (lmbda_scaled, nu), a=1e-8, b=100.0, tol=self.tol)

        phi = 0.5 * kappa + lmbda_scaled / (1 + kappa**(1 - nu))

        def rho_nu_prime(theta, nu):
            abs_theta = abs(theta)
            denom = (1 + abs_theta**(1 - nu))**2
            return np.sign(theta) * (1 + nu * abs_theta**(1 - nu)) / denom

        def drho_nu_prime(theta, nu):
            abs_theta = abs(theta)
            num = nu * (1 - nu) * abs_theta**(-nu)
            denom = (1 + abs_theta**(1 - nu))**3
            return num / denom

        def F_theta(theta, z, lmbda, nu):
            return theta - z + lmbda * rho_nu_prime(theta, nu)

        def F_theta_prime(theta, z, lmbda, nu):
            return 1 + lmbda * drho_nu_prime(theta, nu)

        result = np.zeros_like(z)
        for j, z_j in enumerate(z):
            print(f"⚠️ z_j={z_j:.3f}, phi={phi:.3f}")
            if abs(z_j) <= phi:
                result[j] = 0.0
            else:
                try:
                    theta = newton(F_theta, F_theta_prime, F_args=(z_j, lmbda_scaled, nu), x0=z_j, tol=self.tol)
                except Exception:
                    theta = z_j
                result[j] = theta

        return result

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
                            prox_g_args=[lmbda, self.nu],
                            max_iter=self.max_iter,
                            tol=self.tol)

            score_tmp = self._get_score(beta, beta_hat)

            for key in score_tmp:
                score[key].append(score_tmp[key])

            if self.verbose:
                print(f"\t |{i+1}| Simulation {i+1}/{self.simu_iter} :")
                print(f"\t\t beta : {beta}")
                print(f"\t\t beta estimé : {beta_hat}")
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

        plot_scores(self.score, self.list_s, f"HARDER LASSO ISTA with backtracking σ={self.sigma}, n={self.n}, p={self.p}, nu={self.nu}")
