
import numpy as np
import pandas as pd

class TestingPredictionsRobustToAggregation:
    def __init__(self, data):
        self.data = data

    def run(self):
        real_df = self.data
        self.testing_predictions_robust_to_aggregation(real_df)

    def testing_predictions_robust_to_aggregation(self, real_df):
        """
        This function tests the predictions that are robust to aggregation across heterogeneous players
        """

        kick_counts = real_df['player_id'].value_counts()

        ## Testing that all kickers and goalies play mixed strategies
        ## Hard to prove because:
        ## 1. small smaple size for both kickers and goalies -> even if they play mixed strategies, we might not see it (same action)
        ## 2. different strategies for different players -> different observations for the same player may suggest mixed strategies, even if they are not


        at_least_4_ids = kick_counts[kick_counts >= 4].index.tolist()
        #print("\nSides of kicks for players with at least 4 kicks:")
        same_side_4 = 0
        for pid in at_least_4_ids:
            sides = real_df[real_df['player_id'] == pid]['Kicker_Side'].tolist()
            #print(f"Player {pid}: {sides}")
            if len(set(sides)) == 1:
                same_side_4 += 1
        print(f"Players with at least 4 kicks who always kick to the same side: {same_side_4}")
        print()
        #print(f"Players with at least 4 kicks: {len(at_least_4_ids)}")

        exactly_3_ids = kick_counts[kick_counts == 3].index.tolist()
        #print("\nSides of kicks for players with exactly 3 kicks:")
        same_side_3 = 0
        for pid in exactly_3_ids:
            sides = real_df[real_df['player_id'] == pid]['Kicker_Side'].tolist()
            if len(set(sides)) == 1:
                same_side_3 += 1
        print(f"Players with exactly 3 kicks who always kick to the same side: {same_side_3}")
        print(f"Players with exactly 3 kicks: {len(exactly_3_ids)}")
        print()

        exactly_2_ids = kick_counts[kick_counts == 2].index.tolist()
        #print("\nSides of kicks for players with exactly 2 kicks:")
        same_side_2 = 0
        for pid in exactly_2_ids:
            sides = real_df[real_df['player_id'] == pid]['Kicker_Side'].tolist()
            if len(set(sides)) == 1:
                same_side_2 += 1
        print(f"Players with exactly 2 kicks who always kick to the same side: {same_side_2}")
        print(f"Players with exactly 2 kicks: {len(exactly_2_ids)}")
        print(f"Percentage of players with exactly 2 kicks who always kick to the same side: {same_side_2 / len(exactly_2_ids) * 100:.2f}%")
        print()
        
        ## Number of players with at least 2 kicks 
        at_least_2_ids = kick_counts[kick_counts >= 2].index.tolist()
        print(f"Number of players with at least 2 kicks: {len(at_least_2_ids)}")
        print()
        

        same_side_2_atLeast = 0
        for pid in at_least_2_ids:
            sides = real_df[real_df['player_id'] == pid]['Kicker_Side'].tolist()
            if len(set(sides)) == 1:
                same_side_2_atLeast += 1
        

        side_probs = real_df['Kicker_Side'].value_counts(normalize=True)
        p_left = side_probs.get('Left', 0)
        p_middle = side_probs.get('Center', 0)
        p_right = side_probs.get('Right', 0)
        print(f"Probability of kicking left: {p_left:.2f}, middle: {p_middle:.2f}, right: {p_right:.2f}")

        expected_count = 0
        variance = 0
        kick_counts = real_df['player_id'].value_counts()

        for n in kick_counts[kick_counts >= 2]:
            p = self.prob_same_side(n, p_left, p_middle, p_right)
            expected_count += p 
            variance += p * (1 - p)
        se = np.sqrt(variance)

        print(f"Expected number of kickers always kicking same side: {expected_count:.1f}")
        print(f"Standard Error(SE) of this estimate: {se:.1f}")
        print(f"Actual number of such kickers: {same_side_2_atLeast}")
        print()

        at_least_2_goalies_ids = real_df['goalkeeper_id'].value_counts()[real_df['goalkeeper_id'].value_counts() >= 2].index.tolist()
        same_side_2_atLeast_Goalies = 0
        for gk_id in at_least_2_goalies_ids:
            sides = real_df[real_df['goalkeeper_id'] == gk_id]['Goalie_Side'].tolist()
            if len(set(sides)) == 1:
                same_side_2_atLeast_Goalies += 1
        print()

        print(f"Number of Goalies with at least 2 jumps: {len(at_least_2_goalies_ids)}")
        print()

        jump_probs = real_df['Goalie_Side'].value_counts(normalize=True)
        p_left_gk = jump_probs.get('Left', 0)
        p_middle_gk = jump_probs.get('Center', 0)
        p_right_gk = jump_probs.get('Right', 0)

        
        print(f"Probability of jumping left: {p_left_gk:.2f}, middle: {p_middle_gk:.2f}, right: {p_right_gk:.2f}")

        jump_counts = real_df['goalkeeper_id'].value_counts()
        jump_counts = jump_counts[jump_counts >= 2]  # Only include goalies with at least 2 kicks faced

        expected_count_gk = 0
        variance_gk = 0

        for n in jump_counts:
            p = p_left_gk**n + p_middle_gk**n + p_right_gk**n
            expected_count_gk += p
            variance_gk += p * (1 - p)

        se_gk = np.sqrt(variance_gk)

        print(f"Expected number of goalies always jumping same side: {expected_count_gk:.1f}")
        print(f"Standard error of this estimate: {se_gk:.1f}")
        print(f"Actual number of such goalies: {same_side_2_atLeast_Goalies}\n")

        #Finally, an additional testable prediction of true randomizing behavior is that there should be no serial correlation in the strategy played
        
        # Table 3: Create observed matrix of shots taken
        shots_matrix = self.create_table3_observed_matrix_of_shots_taken(real_df)
        self.display_table3(shots_matrix)
        print()

        # 5 predictions tested with table 3:
        # 1. kicker will chose center more than the goalie (Proposition 3(i)) -> kickers played center 54 times and goalies 12
        # 2. goalies should play left more than the kickers -> kickers played left 246(51%) times and goalies 265(55%)
        # 3. and 4. Under NS and KS assumptions, the kicker and goalie are both more likely to go left than right -> kicker goes left 246 times and only 181 to right while the goalies jump left 265 times and right only 205
        # 5. left left should have the highest count -> 126(26%); next should be goalie left and kicker left -> in our case is goalie right kicker left with 115 and left right has 109 (smal diff); righ right is the smallest ( with 67) ignoring middle

        # Table 4
        outcome_matrix = self.create_table4_outcome_percentages(real_df)
        self.display_table4(outcome_matrix)


    def create_table3_observed_matrix_of_shots_taken(self, df):
        """Create the observed matrix of shots taken like Table 3"""
        
        shots_matrix = pd.crosstab(
            df['Goalie_Side'], 
            df['Kicker_Side'], 
            margins=True, 
            margins_name='Total'
        )
        if 'Center' in shots_matrix.columns:
            column_order = ['Left', 'Center', 'Right', 'Total']
        else:
            column_order = ['Left', 'Right', 'Total']
        
        available_cols = [col for col in column_order if col in shots_matrix.columns]
        shots_matrix = shots_matrix[available_cols]
        
        if 'Center' in shots_matrix.index:
            row_order = ['Left', 'Center', 'Right', 'Total']
            shots_matrix = shots_matrix.rename(index={'Center': 'Middle'})
            row_order = ['Left', 'Middle', 'Right', 'Total']
        else:
            row_order = ['Left', 'Right', 'Total']
        
        available_rows = [row for row in row_order if row in shots_matrix.index]
        shots_matrix = shots_matrix.loc[available_rows]
        
        shots_matrix.columns.name = 'Kicker'
        shots_matrix.index.name = 'Goalie'
        
        return shots_matrix


    def display_table3(self, matrix):
        
        print("TABLE 3—OBSERVED MATRIX OF SHOTS TAKEN")
        print("=" * 50)
        
        kicker_header = f"{'':<10}"
        for col in matrix.columns:
            if col == 'Center':
                kicker_header += f"{'Kicker':<10}" if col == matrix.columns[0] else f"{'':<10}"
            else:
                kicker_header += f"{'Kicker':<10}" if col == matrix.columns[0] else f"{'':<10}"
        
        print(f"{'':<10}{'Kicker':^40}")
        
        header = f"{'Goalie':<10}"
        for col in matrix.columns:
            if col == 'Center':
                header += f"{'Middle':<10}"
            else:
                header += f"{col:<10}"
        print(header)
        print("-" * len(header))
        
        for idx in matrix.index:
            row = f"{idx:<10}"
            for col in matrix.columns:
                row += f"{matrix.loc[idx, col]:<10}"
            print(row)

    def create_table4_outcome_percentages(self, df):
        
        outcome_counts = pd.crosstab(
            df['Goalie_Side'], 
            df['Kicker_Side'], 
            df['Outcome'],
            aggfunc='sum',
            margins=True, 
            margins_name='Total'
        )
        
        total_shots = pd.crosstab(
            df['Goalie_Side'], 
            df['Kicker_Side'], 
            margins=True, 
            margins_name='Total'
        )
        
        percentage_matrix = (outcome_counts / total_shots * 100).round(1)
        
        if 'Center' in percentage_matrix.columns:
            column_order = ['Left', 'Center', 'Right', 'Total']
        else:
            column_order = ['Left', 'Right', 'Total']
        
        available_cols = [col for col in column_order if col in percentage_matrix.columns]
        percentage_matrix = percentage_matrix[available_cols]
        

        if 'Center' in percentage_matrix.index:
            row_order = ['Left', 'Center', 'Right', 'Total']
            percentage_matrix = percentage_matrix.rename(index={'Center': 'Middle'})
            row_order = ['Left', 'Middle', 'Right', 'Total']
        else:
            row_order = ['Left', 'Right', 'Total']
        
        available_rows = [row for row in row_order if row in percentage_matrix.index]
        percentage_matrix = percentage_matrix.loc[available_rows]
        
        percentage_matrix.columns.name = 'Kicker'
        percentage_matrix.index.name = 'Goalie'
        
        return percentage_matrix

    def display_table4(self, matrix):
        
        print("TABLE 4—OBSERVED MATRIX OF OUTCOMES:")
        print("PERCENTAGE OF SHOTS IN WHICH A GOAL IS SCORED")
        print("=" * 50)
        
        print(f"{'':<10}{'Kicker':^40}")
        
        header = f"{'Goalie':<10}"
        for col in matrix.columns:
            if col == 'Center':
                header += f"{'Middle':<10}"
            else:
                header += f"{col:<10}"
        print(header)
        print("-" * len(header))
        
        for idx in matrix.index:
            row = f"{idx:<10}"
            for col in matrix.columns:
                value = matrix.loc[idx, col]
                row += f"{value:<10.1f}"
            print(row)
        matrix.to_csv('results/tables/table4.csv')
        print(f"Table 4 saved to results/tables/table4.csv")

    def prob_same_side(self, n, p_left, p_middle, p_right):
        return p_left**n + p_middle**n + p_right**n    
        