# linear_tools.py

import numpy as np
import matplotlib.pyplot as plt

def generate_data(
        n: int, 
        p: int, 
        s: int, 
        sigma: float = 0.0, 
        beta0: float = 3.0,
        seed: int = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Génère des données pour un modèle linéaire parcimonieux.

    Args:
        n (int): Nombre d'échantillons.
        p (int): Nombre de variables explicatives.
        s (int): Nombre de variables informatives (non nulles dans beta).
        sigma (float, optional): Écart-type du bruit gaussien ajouté (défaut : 0.0).
        beta0 (float, optional): Valeurs des coefficients non nulle (défaut : 3.0).
        seed (int, optional): Graine aléatoire (défaut : None).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 
            y (observations), 
            X (variables explicatives), 
            beta (coefficients réels).
    """
    np.random.seed(seed)

    X = np.random.randn(n, p)

    beta = np.zeros(p)
    beta[:s] = beta0

    noise = np.random.normal(0, sigma, size=n)

    y = X @ beta + noise

    return y, X, beta

def pesr(
        beta: np.ndarray,
        beta_hat: np.ndarray,
        tol: float = 0.0
    ) -> int:
    """Renvoie si on estime le support exact. Permet de calculer ensuite la probabilité de récuppérer le support exact.

    Args:
        beta (np.ndarray): coefficients réels
        beta_hat (np.ndarray): coefficients estimés
        tol (float, optional): tolérence sur les coefficients estimé ce paramètre doit toujours être nul pour réccupérer exactement le support. Defaults to 0.0.

    Returns:
        int: support estimé == suppport exact
    """
    support = [i for i in range(len(beta)) if beta[i] != 0.]
    support_hat = [i for i in range(len(beta_hat)) if np.abs(beta_hat[i]) > tol]

    return int(support == support_hat)

def tp(
    beta: np.ndarray,
    beta_hat: np.ndarray,
    tol: float = 0.0
    ) -> int:
    """
    Calcule le nombre de vrais positifs (TP) dans la sélection du support.

    Args:
    beta (np.ndarray): Coefficients réels.
    beta_hat (np.ndarray): Coefficients estimés.
    tol (float, optional): Tolérance pour considérer un coefficient comme non nul (défaut : 0.0).

    Returns:
    int: Nombre de vrais positifs.
    """
    support = [i for i in range(len(beta)) if np.abs(beta[i]) > tol]
    support_hat = [i for i in range(len(beta_hat)) if np.abs(beta_hat[i]) > tol]
    return np.sum([1 for i in support if i in support_hat])

def fp(
    beta: np.ndarray,
    beta_hat: np.ndarray,
    tol: float = 0.0
    ) -> int:
    """
    Calcule le nombre de faux positifs (FP) dans la sélection du support.

    Args:
    beta (np.ndarray): Coefficients réels.
    beta_hat (np.ndarray): Coefficients estimés.
    tol (float, optional): Tolérance pour considérer un coefficient comme non nul (défaut : 0.0).

    Returns:
    int: Nombre de faux positifs.
    """
    support = [i for i in range(len(beta)) if np.abs(beta[i]) > tol]
    support_hat = [i for i in range(len(beta_hat)) if np.abs(beta_hat[i]) > tol]
    return np.sum([1 for i in support_hat if i not in support])

def fn(
    beta: np.ndarray,
    beta_hat: np.ndarray,
    tol: float = 0.0
    ) -> int:
    """
    Calcule le nombre de faux négatifs (FN) dans la sélection du support.

    Args:
    beta (np.ndarray): Coefficients réels.
    beta_hat (np.ndarray): Coefficients estimés.
    tol (float, optional): Tolérance pour considérer un coefficient comme non nul (défaut : 0.0).

    Returns:
    int: Nombre de faux négatifs.
    """
    support = [i for i in range(len(beta)) if np.abs(beta[i]) > tol]
    support_hat = [i for i in range(len(beta_hat)) if np.abs(beta_hat[i]) > tol]
    return np.sum([1 for i in support if i not in support_hat])

def tpr(
    beta: np.ndarray,
    beta_hat: np.ndarray,
    tol: float = 0.0
    ) -> float:
    """
    Calcule le taux de vrais positifs (True Positive Rate, TPR).

    Args:
    beta (np.ndarray): Coefficients réels.
    beta_hat (np.ndarray): Coefficients estimés.
    tol (float, optional): Tolérance pour considérer un coefficient comme non nul (défaut : 0.0).

    Returns:
    float: Taux de vrais positifs.
    """
    tp_value = tp(beta, beta_hat, tol)
    fn_value = fn(beta, beta_hat, tol)
    return tp_value / (tp_value + fn_value) if (tp_value + fn_value) > 0 else 0.0

def fdr(
    beta: np.ndarray,
    beta_hat: np.ndarray,
    tol: float = 0.0
    ) -> float:
    """
    Calcule le taux de fausses découvertes (False Discovery Rate, FDR).

    Args:
    beta (np.ndarray): Coefficients réels.
    beta_hat (np.ndarray): Coefficients estimés.
    tol (float, optional): Tolérance pour considérer un coefficient comme non nul (défaut : 0.0).

    Returns:
    float: Taux de fausses découvertes.
    """
    tp_value = tp(beta, beta_hat, tol)
    fp_value = fp(beta, beta_hat, tol)
    return fp_value / (tp_value + fp_value) if (tp_value + fp_value) > 0 else 0.0

def f1(
    beta: np.ndarray,
    beta_hat: np.ndarray,
    tol: float = 0.0
    ) -> float:
    """
    Calcule le score F1 pour la sélection du support.

    Args:
    beta (np.ndarray): Coefficients réels.
    beta_hat (np.ndarray): Coefficients estimés.
    tol (float, optional): Tolérance pour considérer un coefficient comme non nul (défaut : 0.0).

    Returns:
    float: Score F1.
    """
    tp_value = tp(beta, beta_hat, tol)
    fp_value = fp(beta, beta_hat, tol)
    fn_value = fn(beta, beta_hat, tol)
    denom = 2 * tp_value + fp_value + fn_value
    return 2 * tp_value / denom if denom > 0 else 0.0

def plot_scores(
    score: dict,
    x_range: list,
    title: str
    ) -> None:
    """
    Affiche un tableau de graphiques pour présenter les différents scores.
    
    Args:
        score (dict): Dictionnaire contenant différentes métriques de score
        x_range (list): Liste des abscisses sur les différents scores
        title (str): Titre du graphique
    """
    plt.figure(figsize=(12, 5 * (len(score) + 2) // 3))
    plt.suptitle(title, fontsize=16)

    for i, key in enumerate(score, 1):
        plt.subplot((len(score) + 2) // 3, 3, i)
        plt.plot(x_range, score[key], marker='o', linestyle='-', color='tab:blue' if key == 'pesr' else 'tab:orange', label=key.upper())
        plt.xlabel('s')
        plt.ylabel(f'{key.upper()} Score')
        plt.title(f'{key.upper()}')
        plt.legend()
        plt.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_simulations(
    scores: dict,
    x_range: list,
    title: str = ""
    ) -> None:
    """
    Affiche un tableau de graphiques pour présenter les différents scores de simulation différentes.

    Args:
        scores (dict): Dictionnaire contenant les différents scores de simulations
        x_range (list): Liste des abscisses sur les différents scores
        title (str): Titre du graphique
    """

    score_keys = list(next(iter(scores.values())).keys())

    keys_list = [set(score_dict.keys()) for score_dict in scores.values()]
    first_keys = keys_list[0]
    for i, keys in enumerate(keys_list[1:], 2):
        if keys != first_keys:
            raise ValueError(f"Le dictionnaire de scores n°{i} n'a pas les mêmes clés que le premier : {first_keys} vs {keys}")

    n_scores = len(score_keys)
    plt.figure(figsize=(12, 5 * ((n_scores + 2) // 3)))
    plt.suptitle(title, fontsize=16)

    colors = plt.cm.get_cmap('tab10', len(scores))

    for i, key in enumerate(score_keys, 1):
        plt.subplot((n_scores + 2) // 3, 3, i)
        for j, (title_score, score_dict) in enumerate(scores.items()):
            plt.plot(
                x_range,
                score_dict[key],
                marker='o',
                linestyle='-',
                color=colors(j),
                label=title_score
            )
        plt.xlabel('s')
        plt.ylabel(f'{key.upper()} Score')
        plt.title(f'{key.upper()}')
        plt.legend()
        plt.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def ista(
    grad_f: callable,
    prox_g: callable,
    x0: np.ndarray,
    L: float,
    grad_f_args: tuple = (),
    prox_g_args: tuple = (),
    max_iter: int = 1000,
    tol: float = 1e-6
) -> np.ndarray:
    """
    Implémente l'algorithme ISTA (Iterative Shrinkage-Thresholding Algorithm) de façon générale pour minimiser f(x) + g(x).

    Args:
        grad_f (callable): Fonction qui calcule le gradient de f en x.
        prox_g (callable): Opérateur proximal associé à g.
        x0 (np.ndarray): Point de départ.
        L (float): Constante de Lipschitz du gradient de f.
        grad_f_args (tuple): Arguments supplémentaires à passer à grad_f (défaut : ()).
        prox_g_args (tuple): Arguments supplémentaires à passer à prox_g (défaut : ()).
        max_iter (int, optional): Nombre maximal d'itérations (défaut : 1000).
        tol (float, optional): Tolérance pour le critère d'arrêt (défaut : 1e-6).

    Returns:
        np.ndarray: Solution estimée.
    """
    x = x0.copy()
    for _ in range(max_iter):
        x_old = x.copy()
        grad = grad_f(x, *grad_f_args)
        x = prox_g(x - grad / L, *prox_g_args)
        if np.linalg.norm(x - x_old) < tol:
            break
    return x

def ista_backtracking(
    f: callable,
    g: callable,
    grad_f: callable,
    prox_g: callable,
    x0: np.ndarray,
    L0: float,
    f_args: tuple = (),
    g_args: tuple = (),
    grad_f_args: tuple = (),
    prox_g_args: tuple = (),
    eta: float = 2.0,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> np.ndarray:
    """
    Implémente ISTA avec backtracking pour minimiser f(x) + g(x).

    Args:
        f (callable): Fonction f(x) (différentiable).
        g (callable): Fonction g(x) (différentiable).
        grad_f (callable): Fonction qui calcule le gradient de f en x.
        prox_g (callable): Opérateur proximal associé à g.
        x0 (np.ndarray): Point de départ.
        L0 (float): Valeur initiale de la constante de Lipschitz.
        f_args (tuple): Arguments supplémentaires à passer à f (défaut : ()).
        g_args (tuple): Arguments supplémentaires à passer à g (défaut : ()).
        grad_f_args (tuple): Arguments supplémentaires à passer à grad_f (défaut : ()).
        prox_g_args (tuple): Arguments supplémentaires à passer à prox_g (défaut : ()).
        eta (float, optional): Facteur d'augmentation de L (défaut : 2.0).
        max_iter (int, optional): Nombre maximal d'itérations (défaut : 1000).
        tol (float, optional): Tolérance pour le critère d'arrêt (défaut : 1e-6).

    Returns:
        np.ndarray: Solution estimée.
    """
    x = x0.copy()
    for _ in range(max_iter):
        x_old = x
        grad = grad_f(x, *grad_f_args)
        f_x = f(x, *f_args)
        L = L0
        while True:
            z = prox_g(x - grad / L, L, *prox_g_args)
            diff = z - x
            F_z = f(z, *f_args) + g(z, *g_args)
            q = f_x + np.dot(grad, diff) + (L / 2) * np.linalg.norm(diff)**2 + g(z, *g_args)
            if F_z <= q:
                break
            L *= eta
        x = z
        if _ > 1 and np.linalg.norm(x - x_old) < tol:
            break
    return x

def cd(
    grad_f: callable,
    prox_O: callable,
    x0: np.ndarray,
    L: np.ndarray,
    lmbda: float,
    grad_f_args: tuple = [],
    prox_O_args: tuple = [],
    max_iter: int = 1000,
    tol: float = 1e-6
) -> np.ndarray:
    """
    Implémente l'algorithme de descente par coordonnées pour minimiser h(x) := f(x) + λΩ(x).

    Args:
        f (callable): Fonction objectif différentiable f(x).
        grad_f (callable): Fonction qui calcule le gradient de f en x.
        prox_O (callable): Opérateur proximal associé à Ω.
        x0 (np.ndarray): Point initial.
        L (np.ndarray): Constantes de Lipschitz coordinatewise [L₁, ..., Lₙ].
        lmbda (float): Paramètre de régularisation λ.
        prox_O_args (tuple, optional): Arguments supplémentaires à passer à prox_O. (défaut = ())
        grad_f_args (tuple, optional): Arguments supplémentaires à passer à grad_f. (défaut = ())
        max_iter (int, optional): Nombre maximal d'itérations (défaut = 1000).
        tol (float, optional): Tolérance pour le critère d'arrêt (défaut = 1e-6).

    Returns:
        np.ndarray: Solution estimée.
    """
    x = x0.copy()
    p = len(x)
    for k in range(max_iter):
        if k % p == 0:
            x_old = x.copy()
        ik = k % p
        grad = grad_f(x, *grad_f_args)
        x[ik] = prox_O(x[ik] - grad[ik] / L[ik], L[ik], *prox_O_args)
        if ik == p-1 and np.linalg.norm(x - x_old) < tol:
            break
    return x

def qut_lasso_oracle(
    X: np.ndarray,
    sigma: float,
    M: int,
    alpha: float,
    seed: int = None
) -> float:
    """
    Calcule le seuil QUT pour le LASSO par simulation.

    Args:
        X (np.ndarray): Matrice des variables explicatives (n, p).
        sigma (float): Écart-type du bruit.
        M (int): Nombre de simulations.
        alpha (float): Niveau de test (quantile 1 - alpha).
        seed (int, optional): Graine aléatoire (défaut : None).

    Returns:
        float: Seuil QUT estimé.
    """
    n, _ = X.shape
    rng = np.random.default_rng(seed)
    Lambda = []
    for _ in range(M):
        Y_sim = rng.normal(0, sigma, size=n)
        lambda_0 = np.linalg.norm(2 * (X.T @ Y_sim), ord=np.inf)
        Lambda.append(lambda_0)
    lambda_qut = np.quantile(Lambda, 1 - alpha)
    
    return lambda_qut

def qut_square_root_lasso(
    X: np.ndarray,
    M: int,
    alpha: float,
    seed: int = None
) -> float:
    """
    Calcule le seuil QUT pour le square-root LASSO par simulation.

    Args:
        X (np.ndarray): Matrice des variables explicatives (n, p).
        M (int): Nombre de simulations.
        alpha (float): Niveau de test (quantile 1 - alpha).
        seed (int, optional): Graine aléatoire (défaut : None).

    Returns:
        float: Seuil QUT estimé.
    """
    n, _ = X.shape
    rng = np.random.default_rng(seed)
    Lambda = []
    for _ in range(M):
        Y_sim = rng.normal(0, 1.0, size=n)
        denom = np.linalg.norm(Y_sim)
        if denom == 0:
            continue
        lambda_0 = np.linalg.norm(X.T @ Y_sim / denom, ord=np.inf)
        Lambda.append(lambda_0)
    lambda_qut = np.quantile(Lambda, 1 - alpha)
    
    return lambda_qut

def dichotomie(
    F: callable,
    F_args: tuple = [],
    a: float = 1e-8,
    b: float = 50.0,
    tol: float = 1e-3
) -> float:
    """
    Approche la solution de F(x) = 0 sur l'intervalle [a, b] avec une précision epsilon.
    
    Args :
        F (callable): fonction continue pour laquelle on cherche une racine.
        F_args (tuple, optional): Arguments supplémentaires à passer à F. (défaut = ())
        a, b ((float, float), optional) : bornes de l'intervalle, avec F(a) * F(b) < 0. (défaut = (1e-8,50.0))
        tol (float, optional) : précision souhaitée. (défaut = 1e-3)

    Returns:
        float: Une approximation de la racine de F dans [a, b].
    """
    if F(a, *F_args) * F(b, *F_args) > 0:
        print(f"⚠️ Pas de changement de signe : lambda={F_args[0]:.4f}, nu={F_args[1]:.4f}")
        return 0.0
    
    while (b - a) > tol:
        m = (a + b) / 2
        if F(a, *F_args) * F(m, *F_args) <= 0:
            b = m
        else:
            a = m
    return (a + b) / 2

def newton(
    F: callable,
    F_prime: callable,
    F_args: tuple = (),
    x0: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 100
) -> float:
    """
    Approche la solution de F(x) = 0 par la méthode de Newton-Raphson à partir de x0.

    Args :
        F (callable) : fonction pour laquelle on cherche une racine.
        F_prime (callable) : dérivée de F.
        F_args (tuple, optional) : Arguments supplémentaires à passer à F et F'. (défaut = ())
        x0 (float, optional) : point de départ de l'itération. (défaut = 0.0)
        tol (float, optional) : précision souhaitée. (défaut = 1e-6)
        max_iter (int, optional) : nombre maximum d’itérations. (défaut = 100)

    Returns:
        float : Une approximation de la racine de F autour de x0.
    """
    x = x0
    for _ in range(max_iter):
        fx = F(x, *F_args)
        dfx = F_prime(x, *F_args)
        if dfx == 0:
            raise ZeroDivisionError("Dérivée nulle — méthode de Newton échoue.")
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    raise RuntimeError(f"Convergence non atteinte après {max_iter} itérations. Pour la méthode de Newton !")