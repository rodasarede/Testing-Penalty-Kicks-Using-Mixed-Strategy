import numpy as np

class QLearningAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.actions = actions  # ['Left', 'Center', 'Right']
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.q_table = {}       # (state, action) -> Q-value

    def get_state(self, history):
        # Stateless model — one shared global state
        return 'global'

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            q_values = [self.q_table.get((state, a), 0) for a in self.actions]
            return self.actions[np.argmax(q_values)]

    def learn(self, state, action, reward, next_state):
        old_q = self.q_table.get((state, action), 0)
        max_next_q = max([self.q_table.get((next_state, a), 0) for a in self.actions])
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.q_table[(state, action)] = new_q

   