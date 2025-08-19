import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os
from scipy import stats

class TestingIdenticalGoalkeepers:
    def __init__(self, data):
        self.data = data

    def run(self):
        results_table5, f_statistics = self.run_test_identical_goalkeepers_table_5(self.data)
        self.save_table_5_results(results_table5, f_statistics)
        self.print_table_5_results(results_table5, f_statistics)
        self.save_model_summaries(results_table5)
        panel_a_results, panel_b_joint_results, panel_b_rejection_counts = self.run_test_kicker_indifference_table_6(self.data)
        self.print_table_6_results(panel_a_results, panel_b_joint_results, panel_b_rejection_counts)
        self.save_table_6(panel_a_results, panel_b_joint_results, panel_b_rejection_counts)

    def save_table_5_results(self, results, f_statistics):
        path = 'results/tables/table_5_results.csv'
        data = {
        'F_statistic': [f_statistics[var] for var in ['Kick successful', 'Kicker shoots right', 'Kicker shoots middle', 'Goalie jumps right']],
        'p_value': [results[var]['p_value'] for var in ['Kick successful', 'Kicker shoots right', 'Kicker shoots middle', 'Goalie jumps right']],
        'R2': [results[var]['model'].rsquared for var in ['Kick successful', 'Kicker shoots right', 'Kicker shoots middle', 'Goalie jumps right']]
    }
        df = pd.DataFrame(data, index=['Kick successful', 'Kicker shoots right', 'Kicker shoots middle', 'Goalie jumps right'])
        df.to_csv(path)
        print(f"Table 5 results saved to {path}")

    def save_table_6(self, panel_a_results, panel_b_joint_results, panel_b_rejection_counts, path='results/tables/table6.csv'):
        # Save Panel A joint test results
        specs = ['5_kicks_no_covariates', '5_kicks_with_covariates', '8_kicks_no_covariates', '8_kicks_with_covariates']
        data = []
        for spec in specs:
            if spec in panel_a_results:
                row = [
                    panel_a_results[spec]['joint_p_value'],
                    panel_a_results[spec]['joint_f_statistic'],
                    f"({panel_a_results[spec]['df_numerator']};{panel_a_results[spec]['df_denominator']})",
                    panel_a_results[spec]['n_kickers'],
                    sum(1 for test in panel_a_results[spec]['individual_tests'] if not pd.isna(test['p_value']) and test['p_value'] < 0.10)
                ]
            else:
                row = [None, None, None, None, None]
            data.append(row)
        df = pd.DataFrame(data, columns=['joint_p_value', 'joint_f_statistic', 'degrees_of_freedom', 'n_kickers', 'n_rejected'],
                        index=specs)
        df.to_csv(path)
        print(f"Table 6 (Panel A) saved to {path}")

    # Optionally, save Panel B as well

    def save_model_summaries(self, results, prefix='results/tables/table5_model_'):
        os.makedirs('results', exist_ok=True)
        for var, res in results.items():
            with open(f"{prefix}{var.replace(' ', '_')}.txt", 'w') as f:
                f.write(res['model'].summary().as_text())
        print("Model summaries saved.")

    def run_test_identical_goalkeepers_table_5(self, df):
        """Test whether goalkeepers are homogeneous using Table 5 approach"""
        # Filter for goalies with at least 4 kicks to increase power of test
        goalie_counts = df['goalkeeper_id'].value_counts()
        df = df[df['goalkeeper_id'].isin(goalie_counts[goalie_counts >= 4].index)].copy()

        # Create new columns
        new_columns = {
            'Country': df['Country'].str.strip(),
            'Kicker_Right': (df['Kicker_Side'] == 'Right').astype(int),
            'Kicker_Center': (df['Kicker_Side'] == 'Center').astype(int),
            'Goalie_Right': (df['Goalie_Side'] == 'Right').astype(int),
            'Minute_Category': pd.cut(df['Minute'], 
                                    bins=[0, 15, 30, 45, 60, 75, 90],
                                    labels=['0-14', '15-29', '30-44', '45-59', '60-74', '75-90']),
            'Score_Diff_Category': pd.cut(df['Score_Diff_Before'],
                                        bins=[-np.inf, -2, -1, 0, 1, 2, np.inf],
                                        labels=['Trail2+', 'Trail1', 'Tied', 'Lead1', 'Lead2+', 'Other']),
            'Home_Team': (df['Team_Type'] == 'H').astype(int)
        }    
        df = df.assign(**new_columns)
        df['Country'] = pd.Categorical(df['Country'], categories=['Italy', 'England', 'Portugal'])
        
        # Define the base covariates (game characteristics + kicker fixed effects)
        covariates = """
        C(Country) +
        C(Minute_Category, Treatment('75-90')) +
        Home_Team +
        C(Score_Diff_Category, Treatment('Tied')) +
        C(player_id)
        """
        
        # Test four different outcome variables
        dependent_vars = {
            'Kick successful': 'Outcome',
            'Kicker shoots right': 'Kicker_Right', 
            'Kicker shoots middle': 'Kicker_Center',
            'Goalie jumps right': 'Goalie_Right'
        }
        
        results = {}
        f_statistics = {}
        
        for var_name, var_col in dependent_vars.items():
            # Run regression with goalie fixed effects
            formula_with_goalie = f"{var_col} ~ {covariates} + C(goalkeeper_id)"
            model_with_goalie = smf.ols(formula_with_goalie, data=df).fit()
            
            # Run regression without goalie fixed effects (restricted model)
            formula_without_goalie = f"{var_col} ~ {covariates}"
            model_without_goalie = smf.ols(formula_without_goalie, data=df).fit()
            
            # F-test for joint significance of goalie fixed effects
            # F = ((RSS_restricted - RSS_unrestricted) / q) / (RSS_unrestricted / (n-k))
            # where q is number of goalie fixed effects, n is sample size, k is total parameters
            
            rss_restricted = model_without_goalie.ssr
            rss_unrestricted = model_with_goalie.ssr
            n = len(df)
            k_unrestricted = len(model_with_goalie.params)
            k_restricted = len(model_without_goalie.params)
            q = k_unrestricted - k_restricted  # number of goalie fixed effects
            
            f_stat = ((rss_restricted - rss_unrestricted) / q) / (rss_unrestricted / (n - k_unrestricted))
            p_value = 1 - stats.f.cdf(f_stat, q, n - k_unrestricted)
            
            results[var_name] = {
                'model': model_with_goalie,
                'f_statistic': f_stat,
                'p_value': p_value,
                'df_numerator': q,
                'df_denominator': n - k_unrestricted
            }
            
            f_statistics[var_name] = f_stat
        
        return results, f_statistics

    def print_table_5_results(self, results, f_statistics):
        """Print Table 5 results in the format shown in the paper"""
        print("=" * 80)
        print("TABLE 5—TESTING WHETHER GOALIES ARE HOMOGENEOUS")
        print("=" * 80)
        print()
        print(f"{'':35} Dependent variable")
        print(f"{'Independent variable':35} {'Kick':>12} {'Kicker':>12} {'Kicker shoots':>12} {'Goalie':>12}")
        print(f"{'':35} {'successful':>12} {'shoots right':>12} {'middle':>12} {'jumps right':>12}")
        print("-" * 80)
        
        # F statistic row (joint test of goalie fixed effects)
        f_stats_row = "F statistic for joint test of goalie-"
        print(f"{f_stats_row:35}", end="")
        for var_name in ['Kick successful', 'Kicker shoots right', 'Kicker shoots middle', 'Goalie jumps right']:
            f_val = f_statistics[var_name]
            print(f"{f_val:12.3f}", end="")
        print()
        
        # P-values row
        p_vals_row = "fixed effects [p value listed below]"
        print(f"{p_vals_row:35}", end="")
        for var_name in ['Kick successful', 'Kicker shoots right', 'Kicker shoots middle', 'Goalie jumps right']:
            p_val = results[var_name]['p_value']
            print(f"[p = {p_val:.3f}]".center(12), end="")
        print()
        print()
        
        # Coefficients for other covariates
        covariate_labels = {
            'C(Minute_Category, Treatment(\'75-90\'))[T.0-14]': 'Minute 0-14',
            'C(Minute_Category, Treatment(\'75-90\'))[T.15-29]': 'Minute 15-29', 
            'C(Minute_Category, Treatment(\'75-90\'))[T.30-44]': 'Minute 30-44',
            'C(Minute_Category, Treatment(\'75-90\'))[T.45-59]': 'Minute 45-59',
            'C(Minute_Category, Treatment(\'75-90\'))[T.60-74]': 'Minute 60-74'
        }
        
        # Get first model to extract covariate names
        first_model = list(results.values())[0]['model']
        
        for param_name in first_model.params.index:
            if any(covar in param_name for covar in covariate_labels.keys()):
                # Find the appropriate label
                label = None
                for key, val in covariate_labels.items():
                    if key in param_name:
                        label = val
                        break
                
                if label:
                    print(f"{label:35}", end="")
                    for var_name in ['Kick successful', 'Kicker shoots right', 'Kicker shoots middle', 'Goalie jumps right']:
                        model = results[var_name]['model']
                        if param_name in model.params.index:
                            coef = model.params[param_name]
                            se = model.bse[param_name]
                            print(f"{coef:8.3f}".rjust(12), end="")
                        else:
                            print(f"{'':>12}", end="")
                    print()
                    
                    # Standard errors in parentheses
                    print(f"{'':35}", end="")
                    for var_name in ['Kick successful', 'Kicker shoots right', 'Kicker shoots middle', 'Goalie jumps right']:
                        model = results[var_name]['model']
                        if param_name in model.params.index:
                            se = model.bse[param_name]
                            print(f"({se:.3f})".center(12), end="")
                        else:
                            print(f"{'':>12}", end="")
                    print()
        
        print()
        print("(League × year) dummies included?".ljust(35) + "Yes".rjust(12) * 4)
        print("Kicker fixed effects included?".ljust(35) + "Yes".rjust(12) * 4)  
        print("Goalie fixed effects included?".ljust(35) + "Yes".rjust(12) * 4)
        print()
        
        # R-squared values
        print("R²".ljust(35), end="")
        for var_name in ['Kick successful', 'Kicker shoots right', 'Kicker shoots middle', 'Goalie jumps right']:
            r2 = results[var_name]['model'].rsquared
            print(f"{r2:.3f}".rjust(12), end="")
        print()
        print()
        
        # Critical values note
        print("Note: The F statistic cutoff values for rejecting the null hypothesis")
        print("at the 10- and 5-percent level are 1.31 and 1.42, respectively.")



    def run_test_kicker_indifference_table_6(self, df):
        """Replicate Table 6: 
        - Panel A: Tests if kickers are indifferent across Left, Middle, Right.
        - Panel B: Tests if goalies' jump direction matters across all kicks vs a given kicker.
        """
        # Prepare data
        df = df.copy()
        df['Kicker_Right'] = (df['Kicker_Side'] == 'Right').astype(int)
        df['Kicker_Middle'] = (df['Kicker_Side'] == 'Center').astype(int)
        df['Kicker_Left'] = (df['Kicker_Side'] == 'Left').astype(int)
        df['Goalie_Right'] = (df['Goalie_Side'] == 'Right').astype(int)

        df['Country'] = df['Country'].str.strip()
        df['Minute_Category'] = pd.cut(df['Minute'], 
                                    bins=[0, 15, 30, 45, 60, 75, 90],
                                    labels=['0-14', '15-29', '30-44', '45-59', '60-74', '75-90'])
        df['Score_Diff_Category'] = pd.cut(df['Score_Diff_Before'],
                                        bins=[-np.inf, -2, -1, 0, 1, 2, np.inf],
                                        labels=['Trail2+', 'Trail1', 'Tied', 'Lead1', 'Lead2+', 'Other'])
        df['Home_Team'] = (df['Team_Type'] == 'H').astype(int)
        df['Country'] = pd.Categorical(df['Country'], categories=['Italy', 'England', 'Portugal'])

        covariates = """
        C(Country) +
        C(Minute_Category, Treatment('75-90')) +
        Home_Team +
        C(Score_Diff_Category, Treatment('Tied'))
        """

        panel_a_results = {}
        panel_b_rejection_counts = []
        panel_b_joint_results = []

        for min_kicks in [5, 8]:
            kicker_counts = df['player_id'].value_counts()
            eligible_kickers = kicker_counts[kicker_counts >= min_kicks].index
            df_filtered = df[df['player_id'].isin(eligible_kickers)].copy()
            kicker_id_map = {kicker_id: f'k{i}' for i, kicker_id in enumerate(eligible_kickers)}

            for include_covariates in [False, True]:
                cov_str = f"+ {covariates}" if include_covariates else ""
                base_formula = f"Outcome ~ C(player_id) {cov_str}"
                spec_name = f"{min_kicks}_kicks_" + ("with_covariates" if include_covariates else "no_covariates")

                # Panel A: Kicker strategy test
                interaction_terms = []
                for kicker_id in eligible_kickers:
                    kicker_mask = df_filtered['player_id'] == kicker_id
                    used_directions = df_filtered.loc[kicker_mask, ['Kicker_Right', 'Kicker_Middle', 'Kicker_Left']].sum()
                    if (used_directions > 0).sum() <= 1:
                        continue  # skip if kicker used only 1 direction

                    safe_name = kicker_id_map[kicker_id]
                    r_col, m_col = f'{safe_name}_right', f'{safe_name}_middle'
                    df_filtered[r_col] = (kicker_mask & df_filtered['Kicker_Right']).astype(int)
                    df_filtered[m_col] = (kicker_mask & df_filtered['Kicker_Middle']).astype(int)
                    interaction_terms += [r_col, m_col]

                if not interaction_terms:
                    continue

                interaction_formula = base_formula + " + " + " + ".join(interaction_terms)
                model_base = smf.ols(base_formula, data=df_filtered).fit()
                model_full = smf.ols(interaction_formula, data=df_filtered).fit()

                rss0 = model_base.ssr
                rss1 = model_full.ssr
                n = len(df_filtered)
                df0 = len(model_base.params)
                df1 = len(model_full.params)
                df_num = df1 - df0
                df_den = n - df1
                f_stat = ((rss0 - rss1) / df_num) / (rss1 / df_den)
                p_val = 1 - stats.f.cdf(f_stat, df_num, df_den)

                # Individual kicker tests
                individual_tests = []
                for kicker_id in eligible_kickers:
                    safe = kicker_id_map[kicker_id]
                    r_param = f'{safe}_right'
                    m_param = f'{safe}_middle'
                    if r_param in model_full.params and m_param in model_full.params:
                        idx_r = list(model_full.params.index).index(r_param)
                        idx_m = list(model_full.params.index).index(m_param)
                        R = np.zeros((2, len(model_full.params)))
                        R[0, idx_r] = 1
                        R[1, idx_m] = 1
                        try:
                            if np.matrix_rank(R @ model_full.cov_params() @ R.T) == R.shape[0]:
                                f_test_result = model_full.f_test(R)
                                individual_tests.append({
                                    'kicker_id': kicker_id,
                                    'f_statistic': float(f_test_result.fvalue),
                                    'p_value': float(f_test_result.pvalue)
                                })
                            else:
                                # Skip kicker with insufficient rank (warn if needed)
                                continue
                        except:
                            continue

                panel_a_results[spec_name] = {
                    'model': model_full,
                    'joint_f_statistic': f_stat,
                    'joint_p_value': p_val,
                    'df_numerator': df_num,
                    'df_denominator': df_den,
                    'n_kickers': len(eligible_kickers),
                    'individual_tests': individual_tests
                }

                # Panel B: Joint test across kickers using C(player_id):Goalie_Right
                df_b = df[df['player_id'].isin(eligible_kickers)].copy()
                base_b_formula = f"Outcome ~ C(player_id)"
                full_b_formula = base_b_formula + f" + C(player_id):Goalie_Right"
                if include_covariates:
                    base_b_formula += f" + {covariates}"
                    full_b_formula += f" + {covariates}"

                model_b0 = smf.ols(base_b_formula, data=df_b).fit()
                model_b1 = smf.ols(full_b_formula, data=df_b).fit()

                rssb0 = model_b0.ssr
                rssb1 = model_b1.ssr
                n_b = len(df_b)
                k0 = len(model_b0.params)
                k1 = len(model_b1.params)
                df_num_b = k1 - k0
                df_den_b = n_b - k1
                f_stat_b = ((rssb0 - rssb1) / df_num_b) / (rssb1 / df_den_b)
                p_val_b = 1 - stats.f.cdf(f_stat_b, df_num_b, df_den_b)

                panel_b_joint_results.append({
                    'min_kicks': min_kicks,
                    'covariates': include_covariates,
                    'f_stat': f_stat_b,
                    'p_value': p_val_b,
                    'df_numerator': df_num_b,
                    'df_denominator': df_den_b
                })

                # Panel B: Individual goalie jump tests per kicker
                count_rejected = 0
                for kicker_id in eligible_kickers:
                    df_k = df_b[df_b['player_id'] == kicker_id]
                    if df_k['Goalie_Right'].nunique() < 2:
                        continue
                    form = "Outcome ~ Goalie_Right"
                    if include_covariates:
                        form += f" + {covariates}"
                    model = smf.ols(form, data=df_k).fit()
                    if 'Goalie_Right' in model.pvalues and model.pvalues['Goalie_Right'] < 0.10:
                        count_rejected += 1

                panel_b_rejection_counts.append({
                    'min_kicks': min_kicks,
                    'covariates': include_covariates,
                    'n_kickers': len(eligible_kickers),
                    'n_rejected': count_rejected
                })

        return panel_a_results, panel_b_joint_results, panel_b_rejection_counts


    def print_table_6_results(self, panel_a_results, panel_b_joint_results, panel_b_rejection_counts):
        """Print formatted Table 6 output, matching the paper."""
        print("=" * 100)
        print("TABLE 6—TESTING FOR IDENTICAL SCORING PROBABILITIES ACROSS LEFT, MIDDLE, AND RIGHT")
        print("FOR INDIVIDUAL KICKERS AND THE GOALIES THEY FACE")
        print("=" * 100)
        print()
        print(f"{'':40} {'Kickers with five':>20} {'Kickers with eight':>20}")
        print(f"{'':40} {'or more kicks':>20} {'or more kicks':>20}")
        print(f"{'Statistic':40} {'(1)':>10} {'(2)':>10} {'(3)':>10} {'(4)':>10}")
        print("-" * 100)

        print("A. Null hypothesis: For a given kicker, the probability of scoring is the same when kicking right, middle, or left.")
        print()

        specs = ['5_kicks_no_covariates', '5_kicks_with_covariates',
                '8_kicks_no_covariates', '8_kicks_with_covariates']

        # P value of joint test
        print(f"{'P value of joint test':40}", end="")
        for spec in specs:
            if spec in panel_a_results:
                p_val = panel_a_results[spec]['joint_p_value']
                print(f"{p_val:10.2f}", end="")
            else:
                print(f"{'':10}", end="")
        print()

        # F statistic
        print(f"{'F statistic':40}", end="")
        for spec in specs:
            if spec in panel_a_results:
                f_val = panel_a_results[spec]['joint_f_statistic']
                print(f"{f_val:10.2f}", end="")
            else:
                print(f"{'':10}", end="")
        print()

        # Degrees of freedom
        print(f"{'Degrees of freedom (numerator;denominator)':40}", end="")
        for spec in specs:
            if spec in panel_a_results:
                df1 = panel_a_results[spec]['df_numerator']
                df2 = panel_a_results[spec]['df_denominator']
                print(f"({df1};{df2})".center(10), end="")
            else:
                print(f"{'':10}", end="")
        print()

        # Number of kickers
        print(f"{'Number of individual kickers':40}", end="")
        for spec in specs:
            if spec in panel_a_results:
                print(f"{panel_a_results[spec]['n_kickers']:10d}", end="")
            else:
                print(f"{'':10}", end="")
        print()

        # Rejected at 10%
        print(f"{'Number of individual kickers for whom null is':40}")
        print(f"{'rejected at 0.10':40}", end="")
        for spec in specs:
            if spec in panel_a_results:
                rejected = sum(1 for test in panel_a_results[spec]['individual_tests']
                            if not np.isnan(test['p_value']) and test['p_value'] < 0.10)
                print(f"{rejected:10d}", end="")
            else:
                print(f"{'':10}", end="")
        print()

        # Covariates included
        print(f"{'Full set of covariates included in specification?':40}", end="")
        for spec in specs:
            cov = 'yes' if 'with_covariates' in spec else 'no'
            print(f"{cov:>10}", end="")
        print()
        print()

        # Panel B
        print("B. Null hypothesis: For goalies facing a given kicker, the probability of scoring is the same whether the goalie jumps")
        print("right or left")
        print()

        # Match joint results in order: 5_no, 5_yes, 8_no, 8_yes
        for label, key in [
            ("P value of joint test", 'p_value'),
            ("F statistic", 'f_stat'),
            ("Degrees of freedom (numerator;denominator)", 'df')
        ]:
            print(f"{label:40}", end="")
            for entry in panel_b_joint_results:
                if key == 'p_value':
                    print(f"{entry['p_value']:10.2f}", end="")
                elif key == 'f_stat':
                    print(f"{entry['f_stat']:10.2f}", end="")
                elif key == 'df':
                    print(f"({entry['df_numerator']};{entry['df_denominator']})".center(10), end="")
            print()

        # Number of kickers
        print(f"{'Number of individual kickers':40}", end="")
        for entry in panel_b_rejection_counts:
            print(f"{entry['n_kickers']:10d}", end="")
        print()

        # Rejected at 10%
        print(f"{'Number of individual kickers for whom null is':40}")
        print(f"{'rejected at 0.10':40}", end="")
        for entry in panel_b_rejection_counts:
            print(f"{entry['n_rejected']:10d}", end="")
        print()

        # Covariates
        print(f"{'Full set of covariates included in specification?':40}", end="")
        for entry in panel_b_rejection_counts:
            cov = 'yes' if entry['covariates'] else 'no'
            print(f"{cov:>10}", end="")
        print()
        print()

        # Notes
        print("Notes: Statistics in the table are based on linear probability models in which the dependent variable is whether or not a goal")
        print("is scored. The table assumes homogeneity across kickers in success rates; that is, the hypothesis tested is whether, for a given")
        print("kicker, success rates are identical when kicking right, middle, or left. No cross-kicker restrictions are imposed. The results in")
        print("the bottom panel of the table refer to goalies facing a particular kicker, under the assumption that goalies are homogeneous.")
        print("When included, the covariates are the same as those used elsewhere in the paper.")
