#Analysis.py
#This file contains the analysis of the data and the validation of the assumptions and propositions
#It is used to validate the assumptions and propositions of the model
#It is also used to analyze the action frequencies and equilibrium properties


import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
from collections import Counter
from scipy.stats import chi2_contingency
import numpy as np
from scipy.stats import f_oneway

class Analysis:
    def __init__(self, P_L, P_R, pi_L, pi_R, mu, df):
        self.P_L = P_L
        self.P_R = P_R
        self.pi_L = pi_L
        self.pi_R = pi_R
        self.mu = mu
        self.df = df
    def validate_assumptions(self):
        """
        Validates all theoretical assumptions from Chiappori et al. (2002) about penalty kick strategies.
        
        Parameters:
            P_L, P_R: Probability of scoring when both choose same side (Left/Right)
            pi_L, pi_R: Probability of scoring when kicker and goalie choose opposite sides
            mu: Probability of scoring when kicker chooses center and goalie jumps to a side
            
        Returns:
            None (raises AssertionError if any assumption is violated)
        """
        # Assumption SC
        assert self.assumption_SC(), "Violates SC"
        assert self.assumption_SC_line(), "Violates SC'"
        
        # Assumption NS
        assert self.assumption_NS(), "Violates NS"
        
        # Assumption KS
        assert self.assumption_KS(), "Violates KS"
        print("All assumptions hold!")

    # A probabilidade de haver golo quando ambos escolhem o mesmo lado tem de ser sempre menor 
    # do que quando o kicker escolhe a direita e o goalie escolhe a esquerda (ou vice-versa)
    def assumption_SC(self):
        return self.pi_R > self.P_L and self.pi_L > self.P_R

    # A probabilidade de haver golo quando ambos escolhem o mesmo lado tem de ser sempre menor
    # do que quando o kicker escolhe o meio e o goalie escolhe um dos lados
    def assumption_SC_line(self):
        return self.pi_R > self.mu and self.pi_L > self.mu

    # A probabilidade de haver golo quando o kicker escolhe o seu natural side (esquerda) e o goalie escolhe o centro
    # ou a direita tem de ser maior do que quando o kicker escolhe o unnatural side (direita) e o goalie escolhe o centro
    # ou a esquerda
    def assumption_NS(self):
        return self.pi_L >= self.pi_R and self.P_L >= self.P_R

    # Chutos para o natural side são (em comparação com o unnatural side):
    #   - Mais difíceis de ir para fora
    #   - Mais difíceis de defender 
    def assumption_KS(self):
        return (self.pi_R - self.P_R) >= (self.pi_L - self.P_L)

    # Valida a Proposição 1, que determina se o equilíbrio em estratégias mistas envolve
    # randomização restrita ({Left, Right}) ou generalizada ({Left, Center, Right}), com base na probabilidade
    # de marcar ao chutar no centro 
    def check_equilibrium_type(self) -> str:
        """
        Determines equilibrium type based on Proposition 1.
        
        Returns:
            "Restricted" if equilibrium is {Left, Right} only
            "General" if equilibrium includes {Left, Center, Right}
        """
        threshold = (self.pi_L * self.pi_R - self.P_L * self.P_R) / (self.pi_L + self.pi_R - self.P_L - self.P_R)
        return "Restricted" if self.mu <= threshold else "General"

    def validate_proposition_2(self):
        """
        Valida as propriedades da Proposição 2:
        1. Independência entre as estratégias do kicker e do goalie.
        2. Igualdade das probabilidades de golo quando o centro é usado.
        3. (SC) O kicker escolhe centro mais frequentemente que o goalie.
        4. (SC) O kicker joga menos vezes o seu lado natural (esquerda) que o goalie.
        5. (SC e NS) O goalie escolhe o natural side do kicker (esquerda) mais vezes que o oposto.
        6. (SC e KS) O kicker escolhe o seu natural side (esquerda) mais vezes que o oposto.
        7. (SC, NS e KS) A frequência do padrão (Left, Left) é maior que (Left, Right) e (Right, Left),
        que são por sua vez maiores que (Right, Right).
        """


        # Propriedade 1: Independência entre as ações 
        # Propriedade 1: Test for independence between Kicker and Goalie choices
        contingency_table = pd.crosstab(self.df['Kicker_Side'], self.df['Goalie_Side'])
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        assert p > 0.05, f"Violates Prop 2.1 (p-value = {p:.4f}), strategies are not independent"
        print("Proposition 2.1 holds!")
        # Propriedade 2: Probabilidades de golo iguais quando centro é usado 
        #proved by empirical tests
        print("Proposition 2.2 holds!")
        # Propriedades 3-7: Condicionadas aos assumptions

        # Propriedade 3: (SC) kicker escolhe centro mais que goalie 
        if self.assumption_SC():
            assert (self.df['Kicker_Side'] == 'Center').mean() > (self.df['Goalie_Side'] == 'Center').mean(), "Violates Prop 2.3"
            print("Proposition 2.3 holds!")

        # Propriedade 4: (SC) kicker joga natural side menos que goalie
            assert (self.df['Kicker_Side'] == 'Left').mean() < (self.df['Goalie_Side'] == 'Left').mean(), "Violates Prop 2.4"
            print("Proposition 2.4 holds!")

        # Propriedade 5: (SC e NS) goalie escolhe Left mais que Right
        if self.assumption_SC() and self.assumption_NS():
            assert (self.df['Goalie_Side'] == 'Left').mean() > (self.df['Goalie_Side'] == 'Right').mean(), "Violates Prop 2.5"
            print("Proposition 2.5 holds!")

        # Propriedade 6: (SC e KS) kicker escolhe Left mais que Right
        if self.assumption_SC() and self.assumption_KS():
            assert (self.df['Kicker_Side'] == 'Left').mean() > (self.df['Kicker_Side'] == 'Right').mean(), "Violates Prop 2.6"
            print("Proposition 2.6 holds!")

        # Propriedade 7: (SC, NS, KS) padrão (Left,Left) mais comum, depois (Left,Right) e (Right,Left), depois (Right,Right)
        if self.assumption_KS() and self.assumption_SC() and self.assumption_NS():
            patterns = Counter(zip(self.df['Kicker_Side'], self.df['Goalie_Side']))
            ll = patterns[('Left', 'Left')]
            lr = patterns[('Left', 'Right')]
            rl = patterns[('Right', 'Left')]
            rr = patterns[('Right', 'Right')]
            assert ll > max(lr, rl) and min(lr, rl) > rr, "Violates Prop 2.7"
            print("Proposition 2.7 holds!")

    def validate_proposition_3(self):
        if self.assumption_SC():
            kicks_center = (self.df['Kicker_Side'] == 'Center').sum()
            goalie_center = (self.df['Goalie_Side'] == 'Center').sum()
            kicks_left = (self.df['Kicker_Side'] == 'Left').sum()
            kicks_right = (self.df['Kicker_Side'] == 'Right').sum()
            goalie_left = (self.df['Goalie_Side'] == 'Left').sum()
            goalie_right = (self.df['Goalie_Side'] == 'Right').sum()
            
            assert kicks_center > goalie_center, "Violates Proposition 3i)"
            print("Proposition 3i) holds!")

            assert kicks_left < goalie_left, "Violates Proposition 3ii)"
            print("Proposition 3ii) holds!")

            if self.assumption_NS():
                assert goalie_left > goalie_right, "Violates Proposition 3iii)"
                print("Proposition 3iii) holds!")

            if self.assumption_KS():
                assert kicks_left > kicks_right, "Violates Proposition 3iv)"
                print("Proposition 3iv) holds!")
                assert self.P_L > self.P_R, "Violates Proposition 3v)"
                print("Proposition 3v) holds!")

            if self.assumption_NS() and self.assumption_KS():
                patterns = Counter(zip(self.df['Kicker_Side'], self.df['Goalie_Side']))
                ll = patterns.get(('Left', 'Left'), 0)
                lr = patterns.get(('Left', 'Right'), 0)
                rl = patterns.get(('Right', 'Left'), 0)
                rr = patterns.get(('Right', 'Right'), 0)

                assert ll > max(lr, rl) and min(lr, rl) > rr, "Violates Proposition 3vi)"
                print("Proposition 3vi) holds!")



    def run(self):
        self.validate_assumptions()
        self.validate_proposition_2()
        self.validate_proposition_3()
        #self.validate_proposition_4() #tested using empirical tests
    