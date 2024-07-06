# Financial Market Prediction with Reinforcement Learning

This repository contains the implementation and analysis of different trading strategies on financial market data using Reinforcement Learning (RL) and traditional models. The project aims to evaluate the effectiveness of Deep Q-Networks (DQN) in comparison to traditional models under various economic conditions and across different markets.

## Table of Contents

- [Dataset](#dataset)
- [Pre-requisites](#pre-requisites)
- [Data Extraction](#data-extraction)
- [Baseline Models](#baseline-models)
- [Reinforcement Learning Models](#reinforcement-learning-models)
- [Evaluation](#evaluation)
- [Metrics](#metrics)
- [Research Questions](#research-questions)
- [How to Run](#how-to-run)
- [Contributing](#contributing)
- [License](#license)

## Dataset

The dataset spans 20 financial years, from 1st April 2004 to 31st March 2024, and includes daily data for the following indices:
- USA: S&P 500 index (Data A)
- UK: FTSE 100 index (Data B)
- India: Nifty 50 index (Data C)

### Features
- Open
- Close
- High
- Low
- Volume

### Source
The data is scraped from Yahoo Finance using the `yfinance` Python package.

### Split
- Train: First 10 years
- Validation: Next 5 years
- Test: Last 5 years

### Assumptions
- No shorting: Only "Buy" then "Sell".
- Short-term trading strategies within two calendar weeks.

## Pre-requisites

- Python 3.x
- `yfinance` package
- Jupyter Notebook

## Data Extraction

The script for data extraction can be found in the `Data_extraction.py` file. This script scrapes and prepares the financial data from Yahoo Finance.

## Baseline Models

The baseline models are implemented in the `Baseline_models.ipynb` notebook and include:
- Moving Averages
- Highest Profit Scenario
- Highest Loss Scenario

## Reinforcement Learning Models

The RL models, particularly Deep Q-Networks (DQN), are applied to the datasets:
- RL Model A – DQN on S&P500 Data
- RL Model B – DQN on FTSE100 Data
- RL Model C – DQN on Nifty50 Data

The models predict the following output features:
- Buy
- Hold
- Sell

## Evaluation

The performance of each model is evaluated on:
- The respective test data
- Cross-market performance (e.g., Model A on Data B and C)
- Performance during significant economic events (e.g., Covid-19, 2008 Financial Crisis)
- Country-specific economic shifts (e.g., elections, government changes, Brexit)

## Metrics

- Confusion Matrix
- F1 Score
- Sensitivity
- Recall
- Accuracy

## Research Questions

- Is DQN a valid model to use in this scenario? How does the DQN compare against traditional and highest/lowest profit models?
- How does a DQN model trained in one market work in the other markets?
- How does the DQN model fare in different economic conditions?

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/financial-market-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd financial-market-prediction
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the data extraction script:
   ```bash
   python Data_extraction.py
   ```
5. Open and run the Jupyter notebooks for the models and analysis:
   ```bash
   jupyter notebook
   ```

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. Ensure that your code follows the project's coding standards and include relevant tests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
