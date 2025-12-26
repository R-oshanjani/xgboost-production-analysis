# xgboost-production-analysis
# XGBoost in Production: Experimental Analysis

This repository contains experiments and analysis for the article
**"XGBoost in Production: Why Boosting Works — and When It Fails"**.

## Overview
The goal of this project is to evaluate why XGBoost performs well on tabular
data and demonstrate failure modes such as overfitting caused by excessive
boosting.

## Repository Structure
- `article.md` – Technical article in Markdown
- `experiments/` – Python scripts for experiments
- `plots/` – Generated figures
- `requirements.txt` – Project dependencies

## Reproducing the Results
pip install -r requirements.txt
python experiments/xgboost_overfitting.py
