"""
Apple Stock Data Collector
This script downloads Apple stock data and performs initial exploration
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

def download_apple_data(years_back=5):
    """
    Download Apple stock data from Yahoo Finance
    """
    
    # Calculate start date (years_back years ago from today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years_back * 365)
    
    print(f"Downloading Apple stock data from {start_date.date()} to {end_date.date()}")
    
    # Download data using yfinance
    # AAPL is Apple's stock symbol
    apple_data = yf.download('AAPL', start=start_date, end=end_date)
    
    # Method 1: Flatten the columns if they are MultiIndex
    if isinstance(apple_data.columns, pd.MultiIndex):
        apple_data.columns = apple_data.columns.get_level_values(0)
    
    print(f"Successfully downloaded {len(apple_data)} days of data")
    print(f"Column names: {list(apple_data.columns)}")  # Debug: show actual column names
    
    return apple_data

def explore_basic_data(data):
    """
    Perform basic data exploration to understand our dataset
    
    Parameters:
    data (pandas.DataFrame): Stock data to explore
    """
    
    print("\n" + "="*50)
    print("BASIC DATA EXPLORATION")
    print("="*50)
    
    # Show basic information about our dataset
    print(f"Dataset shape: {data.shape}")
    print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
    
    print("\nColumn names and what they mean:")
    columns_explanation = {
        'Open': 'Opening price each day',
        'High': 'Highest price reached each day', 
        'Low': 'Lowest price reached each day',
        'Close': 'Closing price each day',
        'Adj Close': 'Adjusted closing price (accounts for splits)',
        'Volume': 'Number of shares traded each day'
    }
    
    # Only show explanations for columns that exist in our data
    for col, explanation in columns_explanation.items():
        if col in data.columns:
            print(f"  {col}: {explanation}")
    
    print("\nFirst 5 rows of data:")
    print(data.head())
    
    print("\nBasic statistics:")
    print(data.describe())
    
    # Check for missing data
    missing_data = data.isnull().sum()
    print(f"\nMissing data points: {missing_data.sum()}")
    
    return data

def create_basic_visualizations(data):
    """
    Create fundamental charts to visualize stock patterns
    """
    
    print("\nCreating visualizations...")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Apple Stock Analysis - Basic Patterns', fontsize=16)
    
    # Plot 1: Stock price over time
    # Use 'Adj Close' if available, otherwise use 'Close'
    price_column = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    axes[0, 0].plot(data.index, data[price_column], color='blue', alpha=0.7)
    axes[0, 0].set_title('Apple Stock Price Over Time')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Trading volume over time
    if 'Volume' in data.columns:
        axes[0, 1].plot(data.index, data['Volume'], color='green', alpha=0.7)
        axes[0, 1].set_title('Trading Volume Over Time')
        axes[0, 1].set_ylabel('Volume (shares)')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Volume data not available', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[0, 1].transAxes)
    
    # Plot 3: Daily price range (High - Low)
    if 'High' in data.columns and 'Low' in data.columns:
        daily_range = data['High'] - data['Low']
        axes[1, 0].plot(data.index, daily_range, color='red', alpha=0.7)
        axes[1, 0].set_title('Daily Price Range (Volatility)')
        axes[1, 0].set_ylabel('Price Range ($)')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'High/Low data not available', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[1, 0].transAxes)
    
    # Plot 4: Price distribution histogram
    axes[1, 1].hist(data[price_column], bins=50, color='purple', alpha=0.7)
    axes[1, 1].set_title('Price Distribution')
    axes[1, 1].set_xlabel('Price ($)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('apple_stock_basic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved as 'apple_stock_basic_analysis.png'")

def main():
    """
    Main function that runs our data collection and analysis
    """
    try:
        print("Apple Stock Pattern Analyzer - Data Collection Phase")
        print("="*60)
        
        # Step 1: Download the data
        apple_data = download_apple_data(years_back=5)
        
        # Check if data was downloaded successfully
        if apple_data is None or apple_data.empty:
            print("Error: No data was downloaded. Please check your internet connection or if the ticker symbol is correct.")
            return None
        
        # Step 2: Explore the data
        apple_data = explore_basic_data(apple_data)
        
        # Step 3: Create visualizations
        try:
            create_basic_visualizations(apple_data)
        except Exception as e:
            print(f"Warning: Could not create visualizations: {str(e)}")
        
        # Step 4: Save data for future use
        try:
            apple_data.to_csv('apple_stock_data.csv')
            print(f"\nData saved to 'apple_stock_data.csv' for future analysis")
        except Exception as e:
            print(f"Warning: Could not save data to file: {str(e)}")
        
        return apple_data

    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")
        return None

# Run the script if executed directly
if __name__ == "__main__":
    data = main()
    if data is None:
        exit(1)

        
