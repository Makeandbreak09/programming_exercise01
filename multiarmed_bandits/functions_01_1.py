# import numpy
import numpy as np

class MultiArmedBandit:
    """
    Represents a multi-armed bandit environment.

    Each arm has its own reward distribution with:
    - a mean value (stored in mus)
    - a fixed variance (sigma)

    The agent can choose an arm/action and receive
    a randomly sampled reward.
    """

    def __init__(self, mus):
        self.Mus = mus       # means
        self.sigma = 1.0     # uniform variance
        self.K = len(mus)    #     # number of arms
        self.mu_star = max(mus)   # optimal mean
        self.a_star = np.argmax(mus)    # optimal action

    def sample(self, a):
        return np.random.normal(self.Mus[a], self.sigma)

def epsilon_greedy(multiArmedBandit, n=10000, epsilon=0.1, time_varying=False, delta=1):
    """
    Epsilon-greedy algorithm for the multi-armed bandit problem.

    Balances exploration (random actions) and exploitation
    (best estimated action), while tracking rewards,
    optimal-action rate, and cumulative regret.

    Parameters:
    -----------
    multiArmedBandit : MultiArmedBandit
        Bandit environment.

    n : int
        Number of iterations.

    epsilon : float
        Exploration probability.

    time_varying : bool
        If True, uses epsilon = delta / t.

    delta : float
        Decay constant for time-varying epsilon.

    Returns:
    --------
    rewards : list
    optimal_percentage : list
    regret : list
    """

    # Initialize metrics
    avg_rewards = []
    optimal_percentage = []
    regret = []

    # Initialize action value estimates and action counter
    Q = np.zeros(multiArmedBandit.K)  # Initialize action value estimates
    N = np.zeros(multiArmedBandit.K)  # Initialize action counter

    # UCB action selection
    for t in range(1, n + 1):
        # Set up epsilon for time-varying case
        if time_varying:
            eps = delta / t
        else:
            eps = epsilon

        # ε-greedy action
        if np.random.rand() < eps:
            action = np.random.randint(multiArmedBandit.K)      # Explore
        else:
            action = np.argmax(Q)                               # Exploit

        reward = multiArmedBandit.sample(action)    # Sample from bandit

        # Update estimates
        N[action] = N[action] + 1                                   # Update action counter
        Q[action] = Q[action] + (reward - Q[action]) / N[action]    # Update action value

        # Track metrics
        avg_rewards.append(avg_rewards[-1] + (reward - avg_rewards[-1]) / t if avg_rewards else reward)  # Incremental average
        optimal_percentage.append(optimal_percentage[-1] + (1-optimal_percentage[-1])/t if action == multiArmedBandit.a_star else optimal_percentage[-1]  + (0-optimal_percentage[-1])/t if optimal_percentage else 1 if action == multiArmedBandit.a_star else 0)
        regret.append(regret[-1] + (multiArmedBandit.mu_star - multiArmedBandit.Mus[action]) if regret else (multiArmedBandit.mu_star - multiArmedBandit.Mus[action]))

    return avg_rewards, optimal_percentage, regret

def ucb(multiArmedBandit, n=10000, c=2):
    """
    Upper Confidence Bound (UCB) algorithm for the
    multi-armed bandit problem.

    Selects actions using estimated rewards and an
    exploration bonus to balance exploration and exploitation,
    while tracking rewards, optimal-action rate,
    and cumulative regret.

    Parameters:
    -----------
    multiArmedBandit : MultiArmedBandit
        Bandit environment.

    n : int
        Number of iterations.

    c : float
        Exploration coefficient controlling confidence bounds.

    Returns:
    --------
    rewards : list
    optimal_percentage : list
    regret : list
    """

    # Initialize metrics
    avg_rewards = []
    optimal_percentage = []
    regret = []

    # Initialize action value estimates and action counter
    Q = np.zeros(multiArmedBandit.K)  # Initialize action value estimate
    N = np.zeros(multiArmedBandit.K)  # Initialize action counter

    for t in range(1, n + 1):
        # UCB action, c > 0 controls exploration
        if t <= multiArmedBandit.K:
            action = t - 1                                      # Ensure each arm is tried at least once
        else:
            action = np.argmax(Q + c * np.sqrt(np.log(t) / N))  # UCB action selection

        reward = multiArmedBandit.sample(action)                # Observe reward from bandit

        # Update estimates
        N[action] = N[action] + 1                                   # Update action counter   
        Q[action] = Q[action] + (reward - Q[action]) / N[action]    # Update action value

        # Track metrics
        avg_rewards.append(avg_rewards[-1] + (reward - avg_rewards[-1]) / t if avg_rewards else reward)  # Incremental average
        optimal_percentage.append(optimal_percentage[-1] + (1-optimal_percentage[-1])/t if action == multiArmedBandit.a_star else optimal_percentage[-1]  + (0-optimal_percentage[-1])/t if optimal_percentage else 1 if action == multiArmedBandit.a_star else 0)
        regret.append(regret[-1] + (multiArmedBandit.mu_star - multiArmedBandit.Mus[action]) if regret else (multiArmedBandit.mu_star - multiArmedBandit.Mus[action]))

    return avg_rewards, optimal_percentage, regret