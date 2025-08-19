from data_processing_real_data import data_processing_real_data
from model import PenaltyKickGame, compute_payoff_params
from analysis import Analysis
from testing_predictions_robust_to_aggregation import TestingPredictionsRobustToAggregation
from testing_simultaneity import TestingSimultaneity
from testing_identical_goalkeepers import TestingIdenticalGoalkeepers

def main():
    data_processor = data_processing_real_data()
    data_processor.run()
    
    P_L, P_R, mu, pi_L, pi_R = compute_payoff_params(data_processor.data)
    model = PenaltyKickGame(P_L, P_R, mu, pi_L, pi_R, data_processor.data)
    print("\n--- Model and Payoff Matrices ---")
    print(model.payoff_matrix)

    analysis = Analysis(P_L, P_R, pi_L, pi_R, mu, data_processor.data)
    analysis.run()

    testing_simultaneity = TestingSimultaneity(data_processor.data)
    testing_simultaneity.run()

    testing_predictions_robust_to_aggregation = TestingPredictionsRobustToAggregation(data_processor.data)
    testing_predictions_robust_to_aggregation.run()

    testing_identical_goalkeepers = TestingIdenticalGoalkeepers(data_processor.data)
    testing_identical_goalkeepers.run()


    

   

if __name__ == "__main__":
    main()