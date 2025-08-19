"""
Simulation module.
TODO:
- Simulate counterfactual scenarios (best-response strategies)
- Compare observed vs. alternative strategies
"""
from model import PenaltyKickGame
from Qlearning import QLearningAgent
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis import check_serial_correlation, run_simultaneous_move_test, validate_assumptions
import argparse

def get_payoff_matrix_from_data(df):
    """Compute payoff matrix parameters from the synthetic dataset."""
    P_L = df[(df['Kicker_Side'] == 'Left') & (df['Goalie_Side'] == 'Left')]['Outcome'].mean()
    pi_L = df[(df['Kicker_Side'] == 'Left') & (df['Goalie_Side'] != 'Left')]['Outcome'].mean()
    mu = df[(df['Kicker_Side'] == 'Center') & (df['Goalie_Side'] != 'Center')]['Outcome'].mean()
    P_R = df[(df['Kicker_Side'] == 'Right') & (df['Goalie_Side'] == 'Right')]['Outcome'].mean()
    pi_R = df[(df['Kicker_Side'] == 'Right') & (df['Goalie_Side'] != 'Right')]['Outcome'].mean()
    return P_L, P_R, mu, pi_L, pi_R

def train_agents(episodes=10000, log_every=500):
    """Train kicker and goalie agents using Q-learning."""
    df = pd.read_csv('data/processed/synthetic_penalty_kicks.csv')
    P_L, P_R, mu, pi_L, pi_R = get_payoff_matrix_from_data(df)

    validate_assumptions(P_L, P_R, pi_L, pi_R, mu)

    game = PenaltyKickGame(P_L, P_R, mu, pi_L, pi_R)

    print('Payoff matrix used for training:')
    print(game.payoff_matrix)

    kicker = QLearningAgent(game.action_space)
    goalie = QLearningAgent(game.action_space)

    kicker_history = []
    goalie_history = []

    kicker_freq_log = []
    goalie_freq_log = []

    for episode in range(episodes):
        kicker_state = 'global'
        goalie_state = 'global'

        kicker_action = kicker.choose_action(kicker_state)
        goalie_action = goalie.choose_action(goalie_state)

        scored = game.step(kicker_action, goalie_action)

        kicker_reward = 1 if scored else 0
        goalie_reward = 0 if scored else 1

        next_kicker_state = 'global'
        next_goalie_state = 'global'

        kicker.learn(kicker_state, kicker_action, kicker_reward, next_kicker_state)
        goalie.learn(goalie_state, goalie_action, goalie_reward, next_goalie_state)

        kicker_history.append(kicker_action)
        goalie_history.append(goalie_action)

        if (episode + 1) % log_every == 0:
            k_dist = pd.Series(kicker_history[-log_every:]).value_counts(normalize=True)
            g_dist = pd.Series(goalie_history[-log_every:]).value_counts(normalize=True)
            kicker_freq_log.append([k_dist.get(a, 0.0) for a in game.action_space])
            goalie_freq_log.append([g_dist.get(a, 0.0) for a in game.action_space])

    plot_strategy_evolution_summary(kicker_freq_log, goalie_freq_log, game.action_space, log_every)
    return kicker, goalie, game

def plot_strategy_evolution_summary(kicker_log, goalie_log, actions, step):
    import matplotlib.pyplot as plt
    kicker_log = np.array(kicker_log)
    goalie_log = np.array(goalie_log)
    x = np.arange(1, len(kicker_log)+1) * step

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    for i, action in enumerate(actions):
        plt.plot(x, kicker_log[:, i], label=f'Kicker: {action}')
    plt.title('Kicker Strategy Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(1, 2, 2)
    for i, action in enumerate(actions):
        plt.plot(x, goalie_log[:, i], label=f'Goalie: {action}')
    plt.title('Goalie Strategy Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Frequency')
    plt.legend()

    plt.tight_layout()

    output_path = 'results/strategy_evolution.png'
    plt.savefig(output_path, dpi=300)
    print(f"✅ Strategy evolution plot saved to {output_path}")  

    plt.show()    

def test_agents(kicker, goalie, game, trials=1000):
    wins = 0
    kicker_actions = []
    goalie_actions = []

    for _ in range(trials):
        kicker_action = kicker.choose_action('global')
        goalie_action = goalie.choose_action('global')

        kicker_actions.append(kicker_action)
        goalie_actions.append(goalie_action)

        i = game.action_space.index(kicker_action)
        j = game.action_space.index(goalie_action)
        scored = game.get_payoff(i, j) > np.random.random()
        wins += scored

    print("\n🎯 Final Kicker Q-values:")
    for action in game.action_space:
        q = kicker.q_table.get(('global', action), 0)
        print(f"  {action}: {q:.4f}")

    print("\n🎯 Final Goalie Q-values:")
    for action in game.action_space:
        q = goalie.q_table.get(('global', action), 0)
        print(f"  {action}: {q:.4f}")
    print(f"Kicker success rate: {wins / trials * 100:.2f}% over {trials} trials.")

    # Plot final strategy distributions
    plot_final_strategies(kicker_actions, goalie_actions, game.action_space)

def plot_final_strategies(kicker_actions, goalie_actions, action_space):
    kicker_counts = pd.Series(kicker_actions).value_counts(normalize=True).reindex(action_space, fill_value=0)
    goalie_counts = pd.Series(goalie_actions).value_counts(normalize=True).reindex(action_space, fill_value=0)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    kicker_counts.plot(kind='bar', ax=ax[0], color='skyblue')
    ax[0].set_title("Kicker Final Strategy")
    ax[0].set_ylim(0, 1)
    ax[0].set_ylabel("Frequency")

    goalie_counts.plot(kind='bar', ax=ax[1], color='lightgreen')
    ax[1].set_title("Goalie Final Strategy")
    ax[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('results/final_strategy_distribution.png', dpi=300)
    print("✅ Final strategy distribution saved to results/final_strategy_distribution.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10000, help='Training episodes')
    parser.add_argument('--trials', type=int, default=1000, help='Testing trials')
    args = parser.parse_args()

    kicker, goalie, game = train_agents(episodes=args.episodes)
    test_agents(kicker, goalie, game, trials=args.trials)

    df = pd.read_csv('data/processed/synthetic_penalty_kicks.csv')
    check_serial_correlation(df)
    run_simultaneous_move_test(df)
    