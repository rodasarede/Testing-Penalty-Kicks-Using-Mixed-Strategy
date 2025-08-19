# AAMS - The Case of Penalty Kicks  
**Group 40**  
**Authors**: Rodrigo Arêde, Manuel Navalho, Tomás Alves  

Replication and critical evaluation of the study by Chiappori, Levitt, and Groseclose (2002) on mixed-strategy Nash equilibrium in football penalty kicks.

## Project Structure

- `data/`: Raw and processed datasets  
- `src/`: Source code for data processing, modeling, and analysis  
  - `model.py`: Game theory models  
  - `testing_simultaneity.py`: Runs regressions to check for simultaneity  
  - `testing_predictions_robust_to_aggregation.py`: Tests predictions robust to aggregation across heterogeneous players  
  - `testing_identical_goalkeepers.py`: Tests for homogeneity among goalkeepers  

- `results/`: Output figures, tables, and models. To obtain this you must run the script

## How to Run

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```
### Step 2 : Run the program
```bash
python src/run_program.py
```

### Reference 
Chiappori, P.-A., Levitt, S., & Groseclose, T. (2002). Testing Mixed-Strategy Equilibria When Players Are Heterogeneous: The Case of Penalty Kicks in Soccer. American Economic Review, 92(4), 1138–1151.