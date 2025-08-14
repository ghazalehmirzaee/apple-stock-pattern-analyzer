# Apple Stock Pattern Analyzer

A comprehensive machine learning project that analyzes Apple stock patterns and predicts price movements using technical indicators and advanced data science techniques.

## Project Overview

This project demonstrates real-world application of machine learning to financial markets, combining data collection, technical analysis, and predictive modeling to understand stock price patterns.

### What This Project Does

- **Data Collection**: Downloads real Apple stock data from Yahoo Finance API
- **Technical Analysis**: Calculates moving averages, support/resistance levels, and volume patterns
- **Pattern Recognition**: Identifies buy/sell signals based on technical indicators
- **Machine Learning**: Uses Random Forest algorithm to predict price direction
- **Visualization**: Creates comprehensive charts showing analysis results

## Technologies Used

- **Python 3.8+**: Core programming language
- **yfinance**: Real-time stock data collection
- **pandas**: Data manipulation and analysis
- **matplotlib/seaborn**: Data visualization
- **scikit-learn**: Machine learning algorithms
- **numpy**: Numerical computing

## Features

### Technical Analysis
- Moving averages (20, 50, 200 day)
- Support and resistance level detection
- Volume analysis with bullish/bearish signals
- Buy/sell signal generation

### Machine Learning
- Random Forest classification model
- Feature engineering from technical indicators
- Model performance evaluation
- Feature importance analysis

### Visualizations
- Stock price charts with technical indicators
- Trading volume analysis
- Prediction accuracy visualization
- Feature importance rankings

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Step-by-Step Execution

1. **Data Collection**
   ```bash
   python models/data_collector.py
   ```
   Downloads 5 years of Apple stock data and creates initial analysis

2. **Technical Analysis**
   ```bash
   python models/technical_indicators.py
   ```
   Calculates technical indicators and generates trading signals

3. **Machine Learning Prediction**
   ```bash
   python models/ml_predictor.py
   ```
   Trains ML model and evaluates prediction accuracy

### Output Files
- `data/apple_stock_data.csv`: Raw stock data
- `data/apple_stock_with_indicators.csv`: Data with technical indicators
- `results/`: All generated charts and visualizations

## Model Performance

Our Random Forest model achieves approximately 55-65% accuracy in predicting next-day price direction, demonstrating that some predictable patterns exist in stock price movements while highlighting the inherent difficulty of market prediction.

### Key Insights
- Moving averages are among the most important predictive features
- Volume patterns provide significant predictive value
- Short-term price momentum influences next-day direction
- Market prediction remains challenging even with sophisticated models


## Contributing

This project welcomes contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b dev`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin dev`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Yahoo Finance for providing free stock data API
- Scikit-learn community for excellent machine learning tools
- Python data science ecosystem for powerful analysis capabilities

## Contact

If you have questions about this project or want to discuss data science applications, feel free to reach out!


