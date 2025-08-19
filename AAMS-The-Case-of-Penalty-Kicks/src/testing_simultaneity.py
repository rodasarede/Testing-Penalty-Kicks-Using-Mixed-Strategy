import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
from IPython.display import display
import statsmodels.api as sm

def prev_right_prop(series):
    """Calculate historical proportion of right choices excluding current kick"""
    return series.expanding().mean().shift(1)

def calculate_excluding_current(series):
    #Calculate mean of all other rows in the group
    # Calculate total and count for the group
    total = series.sum()
    count = series.count()

    return (total - series) / (count - 1) if count > 1 else np.nan  
    
def leave_one_out_mean(x):
    n = x.count()
    if n <= 1:
        return pd.Series([np.nan] * len(x), index=x.index)
    return (x.sum() - x) / (n - 1)

class TestingSimultaneity:
    def __init__(self, data):
        self.data = data

    def run(self):
        self.reproduce_table_3()
        models = self.run_regression_with_all_history()
        with open('results/models/model_summaries_proportions.txt', 'w') as f:
            for model_name, model in models.items():
                f.write(f"Model: {model_name}\n")
                f.write(str(model.summary()))
                f.write("\n\n")
        results = {
        "(1)": self.extract_results(models['all_no_covars']),
        "(2)": self.extract_results(models['4plus_no_covars']),
        "(3)": self.extract_results(models['all_with_covars']),
        "(4)": self.extract_results(models['4plus_with_covars']),
    }
        table = pd.DataFrame(results).T
        table.index.name = "Variable"
        table.to_csv('results/tables/regression_with_proportions_results_table.csv')
        models = self.run_regression_with_history_before_current()
        with open('results/models/model_summaries_history_before_current.txt', 'w') as f:
            for model_name, model in models.items():
                f.write(f"Model: {model_name}\n")
                f.write(str(model.summary()))
                f.write("\n\n")
        results = {
        "(1)": self.extract_results(models['all_no_covars']),
        "(2)": self.extract_results(models['4plus_no_covars']),
        "(3)": self.extract_results(models['all_with_covars']),
        "(4)": self.extract_results(models['4plus_with_covars']),
    }
        table = pd.DataFrame(results).T
        table.index.name = "Variable"
        table.to_csv('results/tables/regression_with_history_before_current_results_table.csv')

    """
    This function runs the regression with the history of the kickers and goalies before the current kick
    """
    def run_regression_with_history_before_current(self):
        df = self.data.copy()
        kicker_counts = df['player_id'].value_counts()
        goalie_counts = df['goalkeeper_id'].value_counts()
        eligible_kickers = kicker_counts[kicker_counts >= 2].index
        eligible_goalies = goalie_counts[goalie_counts >= 2].index
        
        #only use kickers and goalies with at least 2 kicks
        df = df[df['player_id'].isin(eligible_kickers) & df['goalkeeper_id'].isin(eligible_goalies)].copy()    

        # Prepare base data
        #remove whitespace from country column
        df['Country'] = df['Country'].str.strip()

        df['Kicker_Right'] = (df['Kicker_Side'] == 'Right').astype(int)
        df['Goalie_Right'] = (df['Goalie_Side'] == 'Right').astype(int)

        df['Prev_Kicker_Right'] = df.groupby('player_id')['Kicker_Right'].transform(prev_right_prop)
        df['Prev_Goalie_Right'] = df.groupby('goalkeeper_id')['Goalie_Right'].transform(prev_right_prop)

        # Drop rows with NaN (first kicks for each player)
        df = df.dropna(subset=['Prev_Kicker_Right', 'Prev_Goalie_Right'])

        df['Minute_Category'] = pd.cut(df['Minute'], 
                                    bins=[0, 15, 30, 45, 60, 75, 90],
                                    labels=['0-14', '15-29', '30-44', '45-59', '60-74', '75-90'])

        df['Score_Diff_Category'] = pd.cut(df['Score_Diff_Before'],
                                        bins=[-np.inf, -2, -1, 0, 1, 2, np.inf],
                                        labels=['Trail2+', 'Trail1', 'Tied', 'Lead1', 'Lead2+', 'Other'])
        df['Home_Team'] = (df['Team_Type'] == 'H').astype(int)

        df_4plus = df[df['player_id'].isin(kicker_counts[kicker_counts >= 4].index)].copy()
        
        covariates = '''
        + C(Minute_Category, Treatment('75-90'))
        + Home_Team
        + C(Score_Diff_Category, Treatment('Tied'))
        + C(Country)
        '''
        base_vars = 'Kicker_Right ~ Goalie_Right + Prev_Kicker_Right + Prev_Goalie_Right'

        models = {
            'all_no_covars': smf.ols(base_vars + "+ C(Country)", data=df).fit(cov_type='HC3'),
            'all_with_covars': smf.ols(base_vars + covariates, data=df).fit(cov_type='HC3'),
            '4plus_no_covars': smf.ols(base_vars + "+ C(Country)", data=df_4plus).fit(cov_type='HC3'),
            '4plus_with_covars': smf.ols(base_vars + covariates, data=df_4plus).fit(cov_type='HC3'),
            'reversed': smf.ols('Goalie_Right ~ Kicker_Right + Prev_Goalie_Right + Prev_Kicker_Right' + '+ C(Country)', 
                            data=df).fit(cov_type='HC3')
        }
        return models


    """
    This function runs the regression with the porportion of the kickers and goalies that shot right in all kicks
    """
    def run_regression_with_all_history(self):
        df = self.data.copy()
        kicker_counts = df['player_id'].value_counts()
        goalie_counts = df['goalkeeper_id'].value_counts()
        eligible_kickers = kicker_counts[kicker_counts >= 2].index
        eligible_goalies = goalie_counts[goalie_counts >= 2].index
        #only use kickers and goalies with at least 2 kicks
        df = df[df['player_id'].isin(eligible_kickers) & df['goalkeeper_id'].isin(eligible_goalies)].copy()    

        # Prepare base data
        #remove whitespace from country column
        df['Country'] = df['Country'].str.strip()
        #use italian as reference category
        df['Country'] = pd.Categorical(df['Country'], categories=['Italy', 'England', 'Portugal'])

        df['Kicker_Right'] = (df['Kicker_Side'] == 'Right').astype(int)
        df['Goalie_Right'] = (df['Goalie_Side'] == 'Right').astype(int)

        df['Prev_Kicker_Right'] = df.groupby('player_id')['Kicker_Right'].transform(leave_one_out_mean)
        df['Prev_Goalie_Right'] = df.groupby('goalkeeper_id')['Goalie_Right'].transform(leave_one_out_mean)

        # Create exactly six 15-minute interval indicators (75-90 as reference)
        df['Minute_Category'] = pd.cut(df['Minute'], 
                                    bins=[0, 15, 30, 45, 60, 75, 90],
                                    labels=['0-14', '15-29', '30-44', '45-59', '60-74', '75-90'])
        
        # Create five score differential indicators (Tied as reference)
        df['Score_Diff_Category'] = pd.cut(df['Score_Diff_Before'],
                                        bins=[-np.inf, -2, -1, 0, 1, 2, np.inf],
                                        labels=['Trail2+', 'Trail1', 'Tied', 'Lead1', 'Lead2+', 'Other'])
        df['Home_Team'] = (df['Team_Type'] == 'H').astype(int)
        
        #Filter for 4+ kicks
        df_4plus = df[df['player_id'].isin(kicker_counts[kicker_counts >= 4].index)].copy()

        covariates = '''
        + C(Minute_Category, Treatment('75-90'))
        + Home_Team
        + C(Score_Diff_Category, Treatment('Tied'))
        + C(Country)
        '''
        base_vars = 'Kicker_Right ~ Goalie_Right + Prev_Kicker_Right + Prev_Goalie_Right'
        models = {
            'all_no_covars': smf.ols(base_vars + "+ C(Country)", data=df).fit(cov_type='HC3'),
            'all_with_covars': smf.ols(base_vars + covariates, data=df).fit(cov_type='HC3'),
            '4plus_no_covars': smf.ols(base_vars + "+ C(Country)", data=df_4plus).fit(cov_type='HC3'),
            '4plus_with_covars': smf.ols(base_vars + covariates, data=df_4plus).fit(cov_type='HC3'),
            'reversed': smf.ols('Goalie_Right ~ Kicker_Right + Prev_Goalie_Right + Prev_Kicker_Right' + '+ C(Country)', 
                            data=df).fit(cov_type='HC3')
        }
        probit_model = smf.probit('Kicker_Right ~ Goalie_Right + Prev_Kicker_Right + Prev_Goalie_Right + C(Country)', data=df).fit() #probit model to check the robustness of the results
        models['probit'] = probit_model
        return models

    """
    Table 3: Observed matrix of shots taken (real data) to compare with paper
    """
    def reproduce_table_3(self):
        df = self.data.copy()
        # Standardize relevant categories
        kicker_sides = ['Left', 'Center', 'Right']
        goalie_sides = ['Left', 'Center', 'Right']

        # Ensure columns exist
        if 'Kicker_Side' not in df.columns or 'Goalie_Side' not in df.columns:
            raise ValueError("DataFrame must contain 'Kicker_Side' and 'Goalie_Side' columns.")

        # Create the crosstab matrix
        observed_matrix = pd.crosstab(df['Goalie_Side'], df['Kicker_Side'],
                                    rownames=['Goalie'], colnames=['Kicker'],
                                    dropna=False)

        # Reindex to enforce consistent ordering, filling missing values with 0
        observed_matrix = observed_matrix.reindex(index=goalie_sides, columns=kicker_sides, fill_value=0)

        # Add row and column totals
        observed_matrix['Total'] = observed_matrix.sum(axis=1)
        observed_matrix.loc['Total'] = observed_matrix.sum(axis=0)

        # Save to CSV
        observed_matrix.to_csv('results/tables/observed_matrix.csv')
        print(f'Observed matrix saved to results/tables/observed_matrix.csv')

    def extract_results(self, model):
        coefs = model.params.round(3)
        std_errs = model.bse.round(3)
        # Use .get to avoid KeyError if the variable is not in the model
        goalie_right = f"{coefs.get('Goalie_Right', '')} ({std_errs.get('Goalie_Right', '')})"
        prev_kicker = f"{coefs.get('Prev_Kicker_Right', '')} ({std_errs.get('Prev_Kicker_Right', '')})"
        prev_goalie = f"{coefs.get('Prev_Goalie_Right', '')} ({std_errs.get('Prev_Goalie_Right', '')})"
        return {
            "Keeper jumps right": goalie_right,
            "Kicker's history": prev_kicker,
            "Goalie's history": prev_goalie,
            "R²": round(model.rsquared, 3),
            "Number of observations": int(model.nobs)
        }
