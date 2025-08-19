"""
Game-theoretic model module.
TODO:
- Represent the penalty kick as a 3x3 matrix game
- Compute empirical scoring probabilities
- Model as a zero-sum simultaneous-move game
"""

import numpy as np
import pandas as pd
from pathlib import Path
from data_processing_real_data import data_processing_real_data


class PenaltyKickGame:
    """
    Represents the penalty kick game as a 3x3 payoff matrix.
    Rows: Kicker's action (0=L, 1=C, 2=R)
    Columns: Goalie's action (0=L, 1=C, 2=R)
    Payoff: Probability of scoring (as per Table 2 in the proposal)
    """
    def __init__(self, P_L, P_R, mu, pi_L, pi_R, data):
        """
        Initialize the payoff matrix.
        Args:
            P_L: Probability of scoring when both choose Left
            P_R: Probability of scoring when both choose Right
            mu: Probability of scoring if kicker chooses center and goalie dives (L/R).
            pi_L: Probability of scoring when goalie chooses the wrong side (for kicker L)
            pi_R: Probability of scoring when goalie chooses the wrong side (for kicker R)
        """
        self.payoff_matrix = np.array([
#Goalie:     L     C     R
            [P_L, pi_L, pi_L],   # Kicker L
            [mu, 0, mu],         # Kicker C
            [pi_R, pi_R, P_R]    # Kicker R
        ])

        self.action_space = ['Left', 'Center', 'Right']
        self.state = None
        self.data = data

    def reset(self): 
        """Reset the game state"""
        self.state = None

    def step(self, kicker_action, goalie_action):
        """Take a step in the game"""
        self.state = (kicker_action, goalie_action)

        i = self.action_space.index(kicker_action)
        j = self.action_space.index(goalie_action)
        prob_goal = self.payoff_matrix[i, j]

        goal = np.random.rand() < prob_goal

        return goal
    
    
    def get_payoff(self, kicker_action, goalie_action):
        """
        Returns the payoff (probability of scoring) for a given kicker and goalie action.
        kicker_action: 0 (L), 1 (C), 2 (R)
        goalie_action: 0 (L), 1 (C), 2 (R)
        """
        
        return self.payoff_matrix[kicker_action][goalie_action]

def compute_payoff_params(df):
    P_L = df[(df['Kicker_Side'] == 'Left') & (df['Goalie_Side'] == 'Left')]['Outcome'].mean()
    pi_L = df[(df['Kicker_Side'] == 'Left') & (df['Goalie_Side'] != 'Left')]['Outcome'].mean()
    mu = df[(df['Kicker_Side'] == 'Center') & (df['Goalie_Side'] != 'Center')]['Outcome'].mean()
    P_R = df[(df['Kicker_Side'] == 'Right') & (df['Goalie_Side'] == 'Right')]['Outcome'].mean()
    pi_R = df[(df['Kicker_Side'] == 'Right') & (df['Goalie_Side'] != 'Right')]['Outcome'].mean()
    return P_L, P_R, mu, pi_L, pi_R

def scoring_table(df):
    # Natural side ("Left")
    nat_corr = df[(df['Kicker_Side'] == 'Left') & (df['Goalie_Side'] == 'Left')]['Outcome'].mean()
    nat_wrong = df[(df['Kicker_Side'] == 'Left') & (df['Goalie_Side'] != 'Left')]['Outcome'].mean()
    # Opposite side ("Right")
    opp_corr = df[(df['Kicker_Side'] == 'Right') & (df['Goalie_Side'] == 'Right')]['Outcome'].mean()
    opp_wrong = df[(df['Kicker_Side'] == 'Right') & (df['Goalie_Side'] != 'Right')]['Outcome'].mean()
    # Format as percent
    return {
        "Natural side (\"left\")": [nat_corr * 100, nat_wrong * 100],
        "Opposite side (\"right\")": [opp_corr * 100, opp_wrong * 100]
    }
