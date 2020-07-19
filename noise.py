import numpy as np

class ActorNoise:
    def __init__(self, predict_action, a_dim,  noise_type='OU', action_high=1, action_low=-1):
        self.noise = OrnsteinUhlenbeckActionNoise(predict_action, mu=np.zeros(shape=a_dim), sigma=0.2) \
            if noise_type == 'OU' else EpsilonGreedy(predict_action, a_dim, action_high, action_low)

    def predict_action(self, s):
        return self.noise(s)

    def reset(self):
        self.noise.reset()

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, predict_action, mu, sigma, theta=0.15, dt=1e-2, x0=None, rand_seed=1234):
        np.random.seed(rand_seed)
        self.predict_action = predict_action
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self, s):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return self.predict_action([s])[0] + x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

class EpsilonGreedy:
    def __init__(self, predict_action, a_dim, action_high, action_low, rand_seed=1234):
        assert (action_high != None) and (action_low != None), "action bounds have to be set for epsilon greedy to work"
        np.random.seed(rand_seed)
        self.a_dim = a_dim
        self.predict_action = predict_action
        self.action_high = action_high
        self.action_low = action_low
        self.epsilon = 1.
        self.step = 0

    def __call__(self, s):
        if np.random.uniform() > self.epsilon:
            a = self.predict_action([s])[0]
        else:
            a = np.random.uniform(self.action_low, self.action_high, size=self.a_dim)
        self._update_epsilon()
        return a

    def reset(self):
        pass

    def _update_epsilon(self):
        # define epsilon schedule here
        self.epsilon = 1. if self.step < 10000 else 0.3