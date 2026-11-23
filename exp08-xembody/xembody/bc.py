"""Behaviour cloning with a small NumPy MLP (2 x 64 tanh, Adam, MSE).

Two observation types:
  agnostic : cube - ee (3), goal - ee (3), ee height (1), holding flag (1)   -> task space only
  proprio  : agnostic + the arm's joint angles, zero-padded to 7              -> adds embodiment state
Output: end-effector move (3, scaled to [-1, 1] by STEP) and gripper command (1).
"""
import numpy as np

from .env import STEP, Arm, run_expert, sample_task


def features(o, kind):
    f = np.r_[o["cube"] - o["ee"], o["goal"] - o["ee"], o["ee"][2], o["held"]]
    if kind == "proprio":
        q = np.zeros(7); q[:len(o["q"])] = o["q"]
        f = np.r_[f, q]
    return f


def collect(name, n_success, seed, noise=0.005, max_tries=None):
    """Successful DART-noised expert demos on one arm: list of (obs, clean action)."""
    rng, arm, data, ok, tries = np.random.default_rng(seed), Arm(name), [], 0, 0
    max_tries = max_tries or 4 * n_success
    while ok < n_success and tries < max_tries:
        cube, goal = sample_task(rng)
        r = run_expert(arm, cube, goal, noise=noise, rng=rng, record=True)
        tries += 1
        if r["success"]:
            data += r["log"]; ok += 1
    arm.close()
    return data, ok, tries


class MLP:
    def __init__(self, n_in, n_out=4, hidden=64, seed=0):
        r = np.random.default_rng(seed)
        self.W = [r.normal(0, 1 / np.sqrt(n_in), (n_in, hidden)), r.normal(0, 1 / np.sqrt(hidden), (hidden, hidden)),
                  r.normal(0, 1 / np.sqrt(hidden), (hidden, n_out))]
        self.b = [np.zeros(hidden), np.zeros(hidden), np.zeros(n_out)]

    def forward(self, x):
        h1 = np.tanh(x @ self.W[0] + self.b[0]); h2 = np.tanh(h1 @ self.W[1] + self.b[1])
        return h1, h2, h2 @ self.W[2] + self.b[2]

    def fit(self, X, Y, epochs=300, lr=3e-3, batch=256, seed=0):
        r = np.random.default_rng(seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xn = (X - self.mu) / self.sd
        params = self.W + self.b
        m = [np.zeros_like(q) for q in params]; v = [np.zeros_like(q) for q in params]
        t, hist = 0, []
        for _ in range(epochs):
            idx = r.permutation(len(X))
            for s in range(0, len(X), batch):
                b = idx[s:s + batch]; x, y = Xn[b], Y[b]
                h1, h2, out = self.forward(x)
                g = 2 * (out - y) / len(b)
                gW2 = h2.T @ g; gb2 = g.sum(0)
                d2 = (g @ self.W[2].T) * (1 - h2 ** 2); gW1 = h1.T @ d2; gb1 = d2.sum(0)
                d1 = (d2 @ self.W[1].T) * (1 - h1 ** 2); gW0 = x.T @ d1; gb0 = d1.sum(0)
                grads = [gW0, gW1, gW2, gb0, gb1, gb2]
                t += 1
                for k, (q, gq) in enumerate(zip(params, grads)):
                    m[k] = 0.9 * m[k] + 0.1 * gq; v[k] = 0.999 * v[k] + 0.001 * gq ** 2
                    q -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
            hist.append(float(np.mean((self.forward(Xn)[2] - Y) ** 2)))
        return hist

    def predict(self, x):
        return self.forward((x - self.mu) / self.sd)[2]


def train(data, kind, seed=0, epochs=300):
    X = np.array([features(o, kind) for o, _ in data])
    Y = np.array([np.r_[a[:3] / STEP, a[3]] for _, a in data])
    net = MLP(X.shape[1], seed=seed)
    hist = net.fit(X, Y, epochs=epochs, seed=seed)

    def policy(o):
        out = net.predict(features(o, kind)[None])[0]
        return np.r_[np.clip(out[:3], -1, 1) * STEP, 1.0 if out[3] > 0 else -1.0]
    return policy, hist
