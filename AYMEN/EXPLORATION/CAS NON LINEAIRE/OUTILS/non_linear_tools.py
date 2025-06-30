# linear_tools.py

import numpy as np
import matplotlib.pyplot as plt

def generate_data(
        n: int, 
        p: int, 
        s: int, 
        sigma: float = 1.0, 
        beta0: float = 10.0,
        seed: int = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Génère des données pour un modèle non linéaire parcimonieux (Figure 3).

    L'association non linéaire est : 
        y = ∑_{i=1}^{s/2} β₀ · |x_{2i} - x_{2i-1}| + bruit

    Args:
        n (int): Nombre d'échantillons.
        p (int): Nombre de variables explicatives.
        s (int): Nombre de variables informatives (doit être pair).
        sigma (float, optional): Écart-type du bruit gaussien ajouté (défaut : 1.0).
        beta0 (float, optional): Valeur multiplicative des contributions non linéaires (défaut : 10.0).
        seed (int, optional): Graine aléatoire (défaut : None).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 
            y (observations), 
            X (variables explicatives), 
            beta (masque des variables informatives).
    """
    assert s % 2 == 0, "s doit être un entier pair pour former des paires (x_{2i-1}, x_{2i})"
    
    np.random.seed(seed)

    X = np.random.randn(n, p)

    beta = np.zeros(p)
    for i in range(0, s, 2):
        beta[i] = beta0
        beta[i+1] = -beta0  # signe inverse mais même poids pour la paire

    noise = np.random.normal(0, sigma, size=n)

    y = np.zeros(n)
    for i in range(0, s, 2):
        y += beta0 * np.abs(X[:, i+1] - X[:, i])  # correspond à |x_{2i} - x_{2i-1}|
    y += noise

    return y, X, beta

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

def pesr(
        beta: np.ndarray,
        W1_hat: np.ndarray,
        tol: float = 0.0
    ) -> int:
    """
    Renvoie si on estime le support exact. Permet de calculer ensuite la probabilité 
    de récuppérer le support exact (PESR).

    Args:
        beta (np.ndarray): Coefficients réels (vecteur de taille p).
        W1_hat (np.ndarray): Poids estimés de la première couche (matrice de taille p2 x p).
        tol (float, optional): Tolérance pour considérer une colonne comme active. Defaults to 0.0.

    Returns:
        int: 1 si support estimé == support réel, 0 sinon.
    """
    support = [i for i in range(len(beta)) if beta[i] != 0.]
    support_hat = [j for j in range(W1_hat.shape[1]) if np.linalg.norm(W1_hat[:, j], ord=2) > tol]
    return int(support == support_hat)

def tp(
    beta: np.ndarray,
    W1_hat: np.ndarray,
    tol: float = 0.0
    ) -> int:
    """
    Calcule le nombre de vrais positifs (TP) dans la sélection du support.

    Args:
        beta (np.ndarray): Coefficients réels (vecteur de taille p).
        W1_hat (np.ndarray): Poids estimés de la première couche (matrice p2 x p).
        tol (float, optional): Tolérance pour considérer une colonne comme active. Defaults to 0.0.

    Returns:
        int: Nombre de vrais positifs.
    """
    support = [i for i in range(len(beta)) if beta[i] != 0.]
    support_hat = [j for j in range(W1_hat.shape[1]) if np.linalg.norm(W1_hat[:, j]) > tol]
    return np.sum([1 for i in support if i in support_hat])

def fp(
    beta: np.ndarray,
    W1_hat: np.ndarray,
    tol: float = 0.0
    ) -> int:
    """
    Calcule le nombre de faux positifs (FP) dans la sélection du support.

    Args:
        beta (np.ndarray): Coefficients réels (vecteur de taille p).
        W1_hat (np.ndarray): Poids estimés de la première couche (matrice p2 x p).
        tol (float, optional): Tolérance pour considérer une colonne comme active. Defaults to 0.0.

    Returns:
        int: Nombre de faux positifs.
    """
    support = [i for i in range(len(beta)) if beta[i] != 0.]
    support_hat = [j for j in range(W1_hat.shape[1]) if np.linalg.norm(W1_hat[:, j]) > tol]
    return np.sum([1 for i in support_hat if i not in support])

def fn(
    beta: np.ndarray,
    W1_hat: np.ndarray,
    tol: float = 0.0
    ) -> int:
    """
    Calcule le nombre de faux négatifs (FN) dans la sélection du support.

    Args:
        beta (np.ndarray): Coefficients réels (vecteur de taille p).
        W1_hat (np.ndarray): Poids estimés de la première couche (matrice p2 x p).
        tol (float, optional): Tolérance pour considérer une colonne comme active. Defaults to 0.0.

    Returns:
        int: Nombre de faux négatifs.
    """
    support = [i for i in range(len(beta)) if beta[i] != 0.]
    support_hat = [j for j in range(W1_hat.shape[1]) if np.linalg.norm(W1_hat[:, j]) > tol]
    return np.sum([1 for i in support if i not in support_hat])

def tpr(
    beta: np.ndarray,
    W1_hat: np.ndarray,
    tol: float = 0.0
    ) -> float:
    """
    Calcule le taux de vrais positifs (True Positive Rate, TPR).

    Args:
        beta (np.ndarray): Coefficients réels (vecteur de taille p).
        W1_hat (np.ndarray): Poids estimés de la première couche (matrice p2 x p).
        tol (float, optional): Tolérance pour considérer une colonne comme active. Defaults to 0.0.

    Returns:
        float: TPR.
    """
    tp_value = tp(beta, W1_hat, tol)
    fn_value = fn(beta, W1_hat, tol)
    return tp_value / (tp_value + fn_value) if (tp_value + fn_value) > 0 else 0.0

def fdr(
    beta: np.ndarray,
    W1_hat: np.ndarray,
    tol: float = 0.0
    ) -> float:
    """
    Calcule le taux de fausses découvertes (False Discovery Rate, FDR).

    Args:
        beta (np.ndarray): Coefficients réels (vecteur de taille p).
        W1_hat (np.ndarray): Poids estimés de la première couche (matrice p2 x p).
        tol (float, optional): Tolérance pour considérer une colonne comme active. Defaults to 0.0.

    Returns:
        float: FDR.
    """
    tp_value = tp(beta, W1_hat, tol)
    fp_value = fp(beta, W1_hat, tol)
    return fp_value / (tp_value + fp_value) if (tp_value + fp_value) > 0 else 0.0

def f1(
    beta: np.ndarray,
    W1_hat: np.ndarray,
    tol: float = 0.0
    ) -> float:
    """
    Calcule le score F1 pour la sélection du support.

    Args:
        beta (np.ndarray): Coefficients réels (vecteur de taille p).
        W1_hat (np.ndarray): Poids estimés de la première couche (matrice p2 x p).
        tol (float, optional): Tolérance pour considérer une colonne comme active. Defaults to 0.0.

    Returns:
        float: Score F1.
    """
    tp_value = tp(beta, W1_hat, tol)
    fp_value = fp(beta, W1_hat, tol)
    fn_value = fn(beta, W1_hat, tol)
    denom = 2 * tp_value + fp_value + fn_value
    return 2 * tp_value / denom if denom > 0 else 0.0