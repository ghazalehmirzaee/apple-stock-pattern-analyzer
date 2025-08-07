"""
Technical Indicators Calculator
This module calculates various technical indicators used in stock analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class TechnicalIndicators:
    """
    A class to calculate and visualize technical indicators for stock analysis
    """
    
    def __init__(self, data):
        """Initialize with stock data"""
        self.data = data.copy()
        # Print available columns for debugging
        print("Available columns:", self.data.columns.tolist())
        
    def calculate_moving_averages(self):
        """Calculate different moving averages to identify trends"""
        print("Calculating moving averages...")
        
        # Use 'Close' instead of 'Adj Close'
        self.data['MA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['MA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['MA_200'] = self.data['Close'].rolling(window=200).mean()
        
        print("Moving averages calculated successfully!")
        return self.data
    
    def identify_trend_signals(self):
        """Generate buy/sell signals based on moving average crossovers"""
        print("Identifying trend signals...")
        
        self.data['Signal'] = np.where(
            self.data['MA_20'] > self.data['MA_50'], 1, 0
        )
        self.data['Position'] = self.data['Signal'].diff()
        
        buy_signals = len(self.data[self.data['Position'] == 1])
        sell_signals = len(self.data[self.data['Position'] == -1])
        print(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
        
        return self.data

    def calculate_volume_analysis(self):
        """Analyze trading volume patterns"""
        print("Analyzing volume patterns...")
        
        self.data['Volume_MA'] = self.data['Volume'].rolling(window=20).mean()
        volume_threshold = self.data['Volume_MA'] * 1.5
        self.data['High_Volume'] = self.data['Volume'] > volume_threshold
        
        # Calculate price change using 'Close' instead of 'Adj Close'
        self.data['Price_Change'] = self.data['Close'].pct_change()
        
        self.data['Volume_Signal'] = 'Normal'
        bullish_condition = (self.data['High_Volume']) & (self.data['Price_Change'] > 0.02)
        bearish_condition = (self.data['High_Volume']) & (self.data['Price_Change'] < -0.02)
        
        self.data.loc[bullish_condition, 'Volume_Signal'] = 'Bullish'
        self.data.loc[bearish_condition, 'Volume_Signal'] = 'Bearish'
        
        volume_signals = self.data['Volume_Signal'].value_counts()
        print("Volume signal distribution:")
        for signal, count in volume_signals.items():
            print(f"  {signal}: {count} days")
        
        return self.data
    
    def calculate_support_resistance(self, window=20):
        """Identify support and resistance levels"""
        print(f"Calculating support and resistance levels (window={window})...")
        
        local_minima = []
        local_maxima = []
        
        for i in range(window, len(self.data) - window):
            current_low = self.data['Low'].iloc[i]
            current_high = self.data['High'].iloc[i]
            
            surrounding_lows = self.data['Low'].iloc[i-window:i+window+1]
            surrounding_highs = self.data['High'].iloc[i-window:i+window+1]
            
            if current_low == surrounding_lows.min():
                local_minima.append((self.data.index[i], current_low))
            
            if current_high == surrounding_highs.max():
                local_maxima.append((self.data.index[i], current_high))
        
        print(f"Found {len(local_minima)} potential support levels")
        print(f"Found {len(local_maxima)} potential resistance levels")
        
        return local_minima, local_maxima
    
    def create_technical_analysis_chart(self):
        """Create comprehensive chart showing all technical indicators"""
        print("Creating technical analysis visualization...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
        
        # Main price chart with moving averages
        ax1.plot(self.data.index, self.data['Close'], 
                label='Apple Stock Price', color='black', linewidth=1)
        ax1.plot(self.data.index, self.data['MA_20'], 
                label='20-day MA', color='blue', alpha=0.7)
        ax1.plot(self.data.index, self.data['MA_50'], 
                label='50-day MA', color='orange', alpha=0.7)
        ax1.plot(self.data.index, self.data['MA_200'], 
                label='200-day MA', color='red', alpha=0.7)
        
        # Add buy and sell signals
        buy_signals = self.data[self.data['Position'] == 1]
        sell_signals = self.data[self.data['Position'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['Close'], 
                   marker='^', color='green', s=100, label='Buy Signal', zorder=5)
        ax1.scatter(sell_signals.index, sell_signals['Close'], 
                   marker='v', color='red', s=100, label='Sell Signal', zorder=5)
        
        ax1.set_title('Apple Stock Technical Analysis', fontsize=16)
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volume chart
        colors = ['green' if x == 'Bullish' else 'red' if x == 'Bearish' else 'gray' 
                 for x in self.data['Volume_Signal']]
        
        ax2.bar(self.data.index, self.data['Volume'], 
               color=colors, alpha=0.6, width=1)
        ax2.plot(self.data.index, self.data['Volume_MA'], 
                color='purple', label='Volume MA', linewidth=2)
        
        ax2.set_title('Trading Volume Analysis')
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('technical_analysis_chart.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Technical analysis chart saved as 'technical_analysis_chart.png'")
    
    def generate_analysis_report(self):
        """Generate a comprehensive analysis report"""
        print("\n" + "="*60)
        print("TECHNICAL ANALYSIS REPORT")
        print("="*60)
        
        latest_data = self.data.iloc[-1]
        current_price = latest_data['Close']  # Use Close instead of Adj Close
        ma_20 = latest_data['MA_20']
        ma_50 = latest_data['MA_50']
        ma_200 = latest_data['MA_200']
        
        print(f"Current Apple Stock Price: ${current_price:.2f}")
        print(f"20-day Moving Average: ${ma_20:.2f}")
        print(f"50-day Moving Average: ${ma_50:.2f}")
        print(f"200-day Moving Average: ${ma_200:.2f}")
        
        # Trend determination
        print("\nTrend Analysis:")
        if current_price > ma_20 > ma_50 > ma_200:
            trend = "Strong Uptrend"
        elif current_price > ma_20 > ma_50:
            trend = "Moderate Uptrend"
        elif current_price < ma_20 < ma_50 < ma_200:
            trend = "Strong Downtrend"
        elif current_price < ma_20 < ma_50:
            trend = "Moderate Downtrend"
        else:
            trend = "Sideways/Mixed"
        
        print(f"Overall Trend: {trend}")
        
        # Signal summary
        total_signals = len(self.data[self.data['Position'] != 0])
        if total_signals > 0:
            recent_signal = self.data[self.data['Position'] != 0].iloc[-1]
            signal_type = "Buy" if recent_signal['Position'] == 1 else "Sell"
            signal_date = recent_signal.name.date()
            print(f"Most Recent Signal: {signal_type} on {signal_date}")
        
        return self.data

def main():
    """Main function to run technical analysis"""
    print("Apple Stock Technical Analysis")
    print("="*40)
    
    try:
        # Load data and verify columns
        data = pd.read_csv('apple_stock_data.csv', index_col=0, parse_dates=True)
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        print(f"Loaded {len(data)} days of Apple stock data")
        print(f"Available columns: {data.columns.tolist()}")
        
        # Initialize and run analysis
        analyzer = TechnicalIndicators(data)
        analyzer.calculate_moving_averages()
        analyzer.identify_trend_signals()
        analyzer.calculate_volume_analysis()
        
        support_levels, resistance_levels = analyzer.calculate_support_resistance()
        analyzer.create_technical_analysis_chart()
        final_data = analyzer.generate_analysis_report()
        
        # Save enhanced data
        final_data.to_csv('apple_stock_with_indicators.csv')
        print(f"\nEnhanced data saved to 'apple_stock_with_indicators.csv'")
        
        return final_data
        
    except FileNotFoundError:
        print("Error: apple_stock_data.csv not found!")
        print("Please run data_collector.py first")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    data = main()

    