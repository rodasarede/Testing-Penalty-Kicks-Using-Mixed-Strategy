### Penalty Kicks – Replication of Mixed-Strategy Equilibria Study

## 📌 Project Overview
This repository replicates the study “Testing Mixed-Strategy Equilibria When Players Are Heterogeneous: The Case of Penalty Kicks in Soccer” by Chiappori, Levitt, and Groseclose (2002).

We investigate whether modern footballers’ penalty kick strategies conform to mixed-strategy Nash equilibrium, using newly collected data from recent European football leagues.

## 📊 Dataset
- **size**: 482 penalties collected manually from official YouTube highlights
- **coverage**:
  - English Premier League (2023–24)
  - Portuguese League (2024–25)
  - Italian League (2020–21, 2022–23)
- **variables**: kicker and goalie identities, shot direction (L/C/R), dominant foot, outcome, game context (minute, scoreline, home/away)
- **summary**:
  - 180 unique kickers, 110 unique goalies
  - 67% right-footed, 33% left-footed
  - 83% conversion rate

## 🧮 Methodology
We reconstruct the penalty kick as a two-player zero-sum game and test:
- **Simultaneity**: Do kickers and goalies decide simultaneously?
- **Robust predictions**: Are mixed-strategy predictions valid at the aggregate level?
- **Identical goalies assumption**: Can goalies be modeled as homogeneous agents?

## ✅ Key Findings
- **Simultaneity rejected**: Unlike the 2002 study, modern kickers (especially experienced ones) appear to react to goalies’ movements.
- **Equilibrium predictions confirmed**: Aggregated outcomes remain consistent with Nash equilibrium theory.
- **Identical goalies assumption holds**: Goalies show no significant heterogeneity in penalty outcomes.

## 📂 Repository Contents
- **AAMS-The-Case-of-Penalty-Kicks/data/**: Raw and processed datasets (see `raw/real_data.xlsx` and `processed/`)
- **AAMS-The-Case-of-Penalty-Kicks/src/**: Python scripts for data processing, modeling, Q-learning, and analysis
- **AAMS-The-Case-of-Penalty-Kicks/results/**: Figures, tables, and model outputs
- **AAMS-The-Case-of-Penalty-Kicks/README.md**: Additional notes specific to the subproject
- **FinalReportAAMS.pdf**: Final report (PDF)

## ▶️ Demo
We also include a short simulation illustrating penalty-kick strategies under equilibrium play.

## 👥 Authors
- Manuel Navalho (IST)
- Rodrigo Arêde (IST)
