import numpy as np

# -------------------- Classes auxiliaires --------------------
class Activation:
    def __init__(self, f, df):
        self.f = f
        self.df = df

    def __call__(self, u):
        return self.f(u)

    def grad(self, u):
        return self.df(u)


class Loss:
    def __init__(self, f, df):
        self.f = f
        self.df = df

    def __call__(self, y, y_hat):
        return self.f(y, y_hat)

    def grad(self, y, y_hat):
        return self.df(y, y_hat)


# -------------------- Fonction par default --------------------

def relu(u):
    return np.maximum(0, u)

def drelu(u):
    return (u > 0).astype(u.dtype)

def sigmoid(u):
    return 1 / (1 + np.exp(-u))

def dsigmoid(u):
    s = sigmoid(u)
    return s * (1 - s)

def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def dmse(y, y_hat):
    return 2 * (y_hat - y) / y.shape[1]


# -------------------- MLP --------------------
class MLP:
    def __init__(
        self,
        layer_sizes,
        activation=None,
        loss=None,
        optimizer=None,      # 'sgd' (par défaut) ou 'adam'
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        seed=None,
    ):
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.L = len(layer_sizes) - 1
        self.act = activation or Activation(sigmoid, dsigmoid)
        self.loss = loss or Loss(mse, dmse)
        self.optimizer = optimizer or 'sgd'
        self._init_weights()
        if self.optimizer == 'adam':
            self._init_adam()
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps
            self.t = 0

    # -------------------- initialisation des paramètres --------------------
    def _init_weights(self):
        self.W, self.b = [], []
        for k in range(self.L):
            fin, fout = self.layer_sizes[k], self.layer_sizes[k + 1]
            scale = np.sqrt(2 / fin)
            self.W.append(np.random.randn(fout, fin) * scale)
            self.b.append(np.zeros((fout, 1)))

    def _init_adam(self):
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]

    # -------------------- Passage vers l'avant --------------------
    def _forward_layer(self, a, k):
        z = self.W[k] @ a + self.b[k]
        a_next = self.act(z) if k < self.L - 1 else z
        return z, a_next

    def forward(self, x):
        a = x
        zs, activations = [], [x]
        for k in range(self.L):
            z, a = self._forward_layer(a, k)
            zs.append(z)
            activations.append(a)
        return activations[-1], zs, activations

    # -------------------- Rétropropagation : Passage vers l'arrière --------------------
    def backward(self, x, y):

        y_hat, zs, activations = self.forward(x)

        gW = [np.zeros_like(w) for w in self.W]
        gb = [np.zeros_like(b) for b in self.b]

        delta = self.loss.grad(y, y_hat)
        gW[-1] = delta @ activations[-2].T
        gb[-1] = np.sum(delta, axis=1, keepdims=True)

        for k in reversed(range(self.L - 1)):

            delta = (self.W[k + 1].T @ delta) * self.act.grad(zs[k])
            gW[k] = delta @ activations[k].T
            gb[k] = np.sum(delta, axis=1, keepdims=True)

        return gW, gb

    # -------------------- Mise à jours des paramètres --------------------
    def _update_sgd(self, gW, gb, lr):
        for k in range(self.L):
            self.W[k] -= lr * gW[k]
            self.b[k] -= lr * gb[k]

    def _update_adam(self, gW, gb, lr):
        self.t += 1
        for k in range(self.L):
            # --- Moments des poids ---
            self.mW[k] = self.beta1 * self.mW[k] + (1 - self.beta1) * gW[k]
            self.vW[k] = self.beta2 * self.vW[k] + (1 - self.beta2) * (gW[k] ** 2)

            # correction de biais
            m_hat = self.mW[k] / (1 - self.beta1 ** self.t)
            v_hat = self.vW[k] / (1 - self.beta2 ** self.t)

            # mise à jour
            self.W[k] -= lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # --- Moments des biais ---
            self.mb[k] = self.beta1 * self.mb[k] + (1 - self.beta1) * gb[k]
            self.vb[k] = self.beta2 * self.vb[k] + (1 - self.beta2) * (gb[k] ** 2)

            m_hat_b = self.mb[k] / (1 - self.beta1 ** self.t)
            v_hat_b = self.vb[k] / (1 - self.beta2 ** self.t)

            self.b[k] -= lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)

    # -------------------- Entrainement --------------------
    def train_step(self, x, y, lr=1e-3, verbose=False):
        gW, gb = self.backward(x, y)
        
        if self.optimizer == "adam":
            self._update_adam(gW, gb, lr)
        else:  # 'sgd'
            self._update_sgd(gW, gb, lr)

        y_hat, _, _ = self.forward(x)
        loss_val = self.loss(y, y_hat)
        if verbose:
            print(f"loss: {loss_val:.6f}")
        return loss_val

    def train(self,
        X,
        Y,
        epochs=100,
        batch_size=32,
        lr=1e-3,
        shuffle=True,
        verbose=True,
    ):
        n = X.shape[1]

        for epoch in range(1, epochs + 1):

            if shuffle:
                idx = np.random.permutation(n)
                X, Y = X[:, idx], Y[:, idx]
            
            epoch_loss = 0.0

            for start in range(0, n, batch_size):
                end = start + batch_size
                x_batch = X[:, start:end]
                y_batch = Y[:, start:end]
                epoch_loss += self.train_step(x_batch, y_batch, lr)
            
            epoch_loss /= (n // batch_size + (n % batch_size != 0))
            
            if verbose:
                print(f"Epoch {epoch}/{epochs} | loss: {epoch_loss:.6f}")

    def __call__(self, x):
        y_hat, _, _ = self.forward(x)
        return y_hat

