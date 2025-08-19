import pandas as pd
import numpy as np

class data_processing_real_data:
    def __init__(self):
        self.data = None

    def load_data(self):
        df = pd.read_excel('data/raw/real_data.xlsx', usecols=range(1, 14))
        df = df.dropna(subset=[df.columns[0]])
        self.data = df

    def standardize_labels(self):
        replace_map = {'L': 'Left', 'R': 'Right', 'C': 'Center'}
        for col in ['Kicker_Foot', 'Kicker_Side', 'Goalie_Side']:
            self.data[col] = self.data[col].replace(replace_map)

    def normalize_sides(self, row):
        if row['Kicker_Foot'] == 'Left':
            side_map = {'Left': 'Right', 'Right': 'Left', 'Center': 'Center'}
            row['Kicker_Side'] = side_map[row['Kicker_Side']]
            row['Goalie_Side'] = side_map[row['Goalie_Side']]
        return row
            
    def add_score_before(self):
        #calculate score difference before the penalty kick, if team is leading +, if trailing -, if equal 0
        self.data['Score_Diff_Before'] = np.where(
        self.data['Team_Type'] == 'H',
        self.data['Home_Goals'] - self.data['Away_Goals'],
        self.data['Away_Goals'] - self.data['Home_Goals']
    ).astype(int)
    
    def add_minute_category(self):
        bins = [0, 15, 30, 45, 60, 75, 91]  # 91 to include 90th minute
        labels = [1, 2, 3, 4, 5, 6] #0-14 ->1, 15-29 ->2, 30-44 ->3, 45-59 ->4, 60-74 ->5, 75-90 ->6
        self.data['Minute_Category'] = pd.cut(
            self.data['Minute'],
            bins=bins,
            labels=labels,
            right=False,
            include_lowest=True
        ).astype(int)

    def process(self):
        self.data = self.data.apply(self.normalize_sides, axis=1)
        self.add_score_before()
        self.add_minute_category()
        self.data['Outcome'] = self.data['Outcome'].round().astype(int)
        self.data['Country'] = self.data['Country'].str.strip()
    def save_data(self):
        output_path = 'data/processed/real_data.csv'
        self.data.to_csv(output_path, index=False)

        self.save_data_statistics()

    def save_data_statistics(self, output_path='data/processed/real_data_statistics.txt'):
        """
        Save summary statistics of the penalty kick data to a text file.
        
        Args:
            data (pd.DataFrame): Processed DataFrame containing penalty kick data
            output_path (str): Path to save the statistics file
        """
        # Calculate statistics
        total_penalties = len(self.data)
        unique_kickers = self.data['player_id'].nunique()
        unique_goalkeepers = self.data['goalkeeper_id'].nunique()
        
        right_footed = len(self.data[self.data['Kicker_Foot'] == 'Right'])
        left_footed = len(self.data[self.data['Kicker_Foot'] == 'Left'])
        
        goals_scored = self.data['Outcome'].sum()  # Assuming 1 = goal, 0 = miss/save
        missed_saved = total_penalties - goals_scored
        
        home_penalties = len(self.data[self.data['Team_Type'] == 'H'])
        away_penalties = len(self.data[self.data['Team_Type'] == 'A'])
        
        # Game state statistics
        leading = len(self.data[self.data['Score_Diff_Before'] > 0])
        drawing = len(self.data[self.data['Score_Diff_Before'] == 0])
        trailing = len(self.data[self.data['Score_Diff_Before'] < 0])
        
        # Minute categories
        minute_cats = {
            '0-14': len(self.data[self.data['Minute_Category'] == 1]),
            '15-29': len(self.data[self.data['Minute_Category'] == 2]),
            '30-44': len(self.data[self.data['Minute_Category'] == 3]),
            '45-59': len(self.data[self.data['Minute_Category'] == 4]),
            '60-74': len(self.data[self.data['Minute_Category'] == 5]),
            '75-90': len(self.data[self.data['Minute_Category'] == 6])
        }
        
            # Add Country statistics (new section)
        if 'Country' in self.data.columns:
            country_counts = self.data['Country'].value_counts().sort_values(ascending=False)
            country_stats = []
            for country, count in country_counts.items():
                country_stats.append(rf"{country} & {count} & {count/total_penalties*100:.1f}\% \\\\")
            country_stats_text = "\n".join(country_stats)
        else:
            country_stats_text = "No country data available \\\\"
    
        # Format the statistics as a text table
        stats_text = rf"""
        \\midrule
        Total penalties & {total_penalties} & 100.0\% \\\\
        Unique kickers & {unique_kickers} & -- \\\\
        Unique goalkeepers & {unique_goalkeepers} & -- \\\\
        Right-footed kickers & {right_footed} & {right_footed/total_penalties*100:.1f}\% \\\\
        Left-footed kickers & {left_footed} & {left_footed/total_penalties*100:.1f}\% \\\\
        Goals scored & {goals_scored} & {goals_scored/total_penalties*100:.1f}\% \\\\
        Missed/saved & {missed_saved} & {missed_saved/total_penalties*100:.1f}\% \\\\
        Home penalties & {home_penalties} & {home_penalties/total_penalties*100:.1f}\% \\\\
        Away penalties & {away_penalties} & {away_penalties/total_penalties*100:.1f}\% \\\\
        \\midrule
        \\textbf{{Game State at Kick}} & & \\\\
        \\quad Leading & {leading} & {leading/total_penalties*100:.1f}\% \\\\
        \\quad Drawing & {drawing} & {drawing/total_penalties*100:.1f}\% \\\\
        \\quad Trailing & {trailing} & {trailing/total_penalties*100:.1f}\% \\\\
        \\midrule
        \\textbf{{Minute of Match}} & & \\\\
        \\quad 0-14 minutes & {minute_cats['0-14']} & {minute_cats['0-14']/total_penalties*100:.1f}\% \\\\
        \\quad 15-29 minutes & {minute_cats['15-29']} & {minute_cats['15-29']/total_penalties*100:.1f}\% \\\\
        \\quad 30-44 minutes & {minute_cats['30-44']} & {minute_cats['30-44']/total_penalties*100:.1f}\% \\\\
        \\quad 45-59 minutes & {minute_cats['45-59']} & {minute_cats['45-59']/total_penalties*100:.1f}\% \\\\
        \\quad 60-74 minutes & {minute_cats['60-74']} & {minute_cats['60-74']/total_penalties*100:.1f}\% \\\\
        \\quad 75-90 minutes & {minute_cats['75-90']} & {minute_cats['75-90']/total_penalties*100:.1f}\% \\\\
        \\midrule
        \\textbf{{By Country}} & & \\\\
        {country_stats_text}
        \\bottomrule
        """
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(stats_text)
        
        print(f"Statistics saved to {output_path}")
            
    def run(self):
        self.load_data()
        self.standardize_labels()
        self.process()
        self.save_data()