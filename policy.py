import pickle
import numpy as np


class Policy:
    def __init__(self, resume, i):
        self.model = {}
        if resume:
            self.load_model(i)
        else:
            self.model['W1'] = np.random.randn(256, 128) / np.sqrt(128)
            self.model['W2'] = np.random.randn(256) / np.sqrt(256)
        self.probs = []
        self.hidden_states = []
        self.grad_buffer = {k: np.zeros_like(v) for k, v in self.model.items()}
        self.rmsp_mem = {k: np.zeros_like(v) for k, v in self.model.items()}

    def policy_forward(self, state):
        hs = np.dot(self.model['W1'], state)
        hs[hs < 0] = 0  # ReLU nonlinearity
        logp = np.dot(self.model['W2'], hs)
        p = 1.0 / (1.0 + np.exp(-logp))
        self.hidden_states.append(hs)
        self.probs.append(-p)
        return p

    def policy_backward(self, reward, states, episode):
        ep_probs = np.vstack(self.probs) * reward
        ep_hs = np.vstack(self.hidden_states)
        self.probs = []
        self.hidden_states = []
        dW2 = np.dot(ep_hs.T, ep_probs).ravel()
        dh = np.outer(ep_probs, self.model['W2'])
        dh[ep_hs <= 0] = 0  # backpro prelu
        dW1 = np.dot(dh.T, states)
        self.model['W1'] += 1e-4 * dW1
        self.model['W2'] += 1e-4 * dW2
        grad = {'W1': dW1, 'W2': dW2}
        for k in self.model:
            self.grad_buffer[k] += grad[k]

        # perform rmsprop parameter update every 32 episodes
        if episode % 32 == 0:
            for k, v in self.model.items():
                g = self.grad_buffer[k]
                self.rmsp_mem[k] = 0.99 * self.rmsp_mem[k] + (0.01) * g**2
                self.model[k] += 1e-4 * g / (np.sqrt(self.rmsp_mem[k]) + 1e-5)
                self.grad_buffer[k] = np.zeros_like(v)

    def load_model(self, i):
        self.model = pickle.load(
            open('saved_model/' + str(i) + 'model.p', 'rb')
        )

    def save_model(self, i):
        pickle.dump(
            self.model, open('saved_model/' + str(i) + 'model.p', 'wb')
        )
