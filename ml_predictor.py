"""
Machine Learning Stock Predictor
This module creates and evaluates ML models for stock price prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class StockMLPredictor:
    """
    Machine Learning predictor for stock price movements
    """
    
    def __init__(self, data):
        """
        Initialize with stock data that includes technical indicators
        
        Parameters:
        data (pandas.DataFrame): Stock data with technical indicators
        """
        self.data = data.copy()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def prepare_ml_features(self):
        """
        Create features and target variable for machine learning
        
        We'll predict whether tomorrow's price will be higher than today's
        """
        
        print("Preparing machine learning features...")
        
        # Create target variable: 1 if tomorrow's price is higher, 0 if lower
        self.data['Tomorrow_Higher'] = (
            self.data['Adj Close'].shift(-1) > self.data['Adj Close']
        ).astype(int)
        
        # Create additional features from our existing data
        self.data['Price_Change_Pct'] = self.data['Adj Close'].pct_change()
        self.data['Volume_Change_Pct'] = self.data['Volume'].pct_change()
        
        # Calculate relative position of current price vs moving averages
        self.data['Price_vs_MA20'] = self.data['Adj Close'] / self.data['MA_20'] - 1
        self.data['Price_vs_MA50'] = self.data['Adj Close'] / self.data['MA_50'] - 1
        self.data['Price_vs_MA200'] = self.data['Adj Close'] / self.data['MA_200'] - 1
        
        # Calculate momentum indicators
        self.data['MA20_vs_MA50'] = self.data['MA_20'] / self.data['MA_50'] - 1
        self.data['MA50_vs_MA200'] = self.data['MA_50'] / self.data['MA_200'] - 1
        
        # Calculate volatility (rolling standard deviation of returns)
        self.data['Volatility'] = self.data['Price_Change_Pct'].rolling(window=20).std()
        
        # Calculate volume relative to average
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA']
        
        # Define our feature set
        self.feature_names = [
            'Price_Change_Pct',      # Today's price change
            'Volume_Change_Pct',     # Today's volume change
            'Price_vs_MA20',         # Price position vs short-term trend
            'Price_vs_MA50',         # Price position vs medium-term trend  
            'Price_vs_MA200',        # Price position vs long-term trend
            'MA20_vs_MA50',          # Short vs medium term momentum
            'MA50_vs_MA200',         # Medium vs long term momentum
            'Volatility',            # Recent price volatility
            'Volume_Ratio',          # Volume vs average
            'Signal'                 # Our technical analysis signal
        ]
        
        print(f"Created {len(self.feature_names)} features for ML model")
        
        # Remove rows with missing data
        initial_rows = len(self.data)
        self.data = self.data.dropna()
        final_rows = len(self.data)
        
        print(f"Data cleaning: {initial_rows} -> {final_rows} rows ({initial_rows-final_rows} removed)")
        
        return self.data
    
    def split_and_scale_data(self):
        """
        Split data into training and testing sets, and scale features
        
        We use older data for training and recent data for testing
        This simulates real-world scenario where we predict future from past
        """
        
        print("Splitting and scaling data...")
        
        # Prepare feature matrix X and target vector y
        X = self.data[self.feature_names].values
        y = self.data['Tomorrow_Higher'].values
        
        # Remove the last row (no tomorrow data available)
        X = X[:-1]
        y = y[:-1]
        
        # Split data: 80% for training, 20% for testing
        # We use the most recent 20% for testing to simulate predicting the future
        split_index = int(0.8 * len(X))
        
        X_train = X[:split_index]
        X_test = X[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]
        
        # Scale features to have mean=0 and std=1
        # This helps the ML algorithm work better
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training data: {len(X_train)} samples")
        print(f"Testing data: {len(X_test)} samples")
        print(f"Target distribution in training: {np.mean(y_train)*100:.1f}% up days")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """
        Train Random Forest model
        
        Random Forest combines many decision trees to make better predictions
        It's good for this problem because it handles complex patterns well
        """
        
        print("Training Random Forest model...")
        
        # Create Random Forest classifier
        # n_estimators = number of trees in the forest
        # random_state = ensures reproducible results
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate training performance using cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance on test data
        
        This tells us how well our model might perform on future, unseen data
        """
        
        print("Evaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]  # Probability of price going up
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Detailed classification report
        print("\nDetailed Performance Report:")
        print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted Down', 'Predicted Up'],
                   yticklabels=['Actually Down', 'Actually Up'])
        plt.title('Confusion Matrix - Model Predictions vs Reality')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return y_pred, y_pred_proba, accuracy
    
    def analyze_feature_importance(self):
        """
        Analyze which features are most important for predictions
        
        This helps us understand what the model considers most valuable
        """
        
        print("Analyzing feature importance...")
        
        # Get feature importance from Random Forest
        importance = self.model.feature_importances_
        
        # Create DataFrame for better visualization
        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        print("\nFeature Importance Ranking:")
        for i, row in feature_importance_df.iterrows():
            print(f"{row['Feature']}: {row['Importance']:.3f}")
        
        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis')
        plt.title('Feature Importance in Stock Prediction Model')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance_df
    
    def create_prediction_visualization(self, X_test, y_test, y_pred, y_pred_proba):
        """
        Create visualization showing model predictions vs actual results
        """
        
        print("Creating prediction visualization...")
        
        # Get test data dates (last portion of our dataset)
        test_dates = self.data.index[-len(X_test):]
        test_prices = self.data['Adj Close'].iloc[-len(X_test):].values
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: Stock price with prediction accuracy
        colors = ['red' if pred != actual else 'green' 
                 for pred, actual in zip(y_pred, y_test)]
        
        axes[0].plot(test_dates, test_prices, 'k-', label='Stock Price', alpha=0.7)
        axes[0].scatter(test_dates, test_prices, c=colors, s=30, alpha=0.6)
        axes[0].set_title('Stock Price with Prediction Accuracy (Green=Correct, Red=Wrong)')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Prediction probabilities
        axes[1].plot(test_dates, y_pred_proba, 'b-', label='Probability Price Goes Up')
        axes[1].axhline(y=0.5, color='r', linestyle='--', label='Decision Threshold')
        axes[1].fill_between(test_dates, y_pred_proba, 0.5, alpha=0.3)
        axes[1].set_title('Model Confidence in Predictions')
        axes[1].set_ylabel('Probability')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Actual vs predicted outcomes
        axes[2].plot(test_dates, y_test, 'g-', label='Actual: Up=1, Down=0', linewidth=2)
        axes[2].plot(test_dates, y_pred, 'r--', label='Predicted: Up=1, Down=0', linewidth=2)
        axes[2].set_title('Actual vs Predicted Price Direction')
        axes[2].set_ylabel('Direction')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Prediction visualization saved as 'prediction_analysis.png'")
    
    def generate_prediction_report(self, accuracy, feature_importance_df):
        """
        Generate comprehensive report of model performance
        """
        
        print("\n" + "="*60)
        print("MACHINE LEARNING PREDICTION REPORT")
        print("="*60)
        
        print(f"Model Type: Random Forest Classifier")
        print(f"Prediction Task: Tomorrow's price direction (up/down)")
        print(f"Overall Accuracy: {accuracy:.1%}")
        
        # Interpret accuracy
        if accuracy > 0.6:
            interpretation = "Good - Better than random guessing"
        elif accuracy > 0.55:
            interpretation = "Moderate - Slightly better than random"
        elif accuracy > 0.45:
            interpretation = "Random - No clear predictive power"
        else:
            interpretation = "Poor - Worse than random guessing"
        
        print(f"Performance Interpretation: {interpretation}")
        
        print(f"\nTop 3 Most Important Features:")
        for i in range(min(3, len(feature_importance_df))):
            feature = feature_importance_df.iloc[i]
            print(f"  {i+1}. {feature['Feature']}: {feature['Importance']:.3f}")
        
        print(f"\nModel Insights:")
        print(f"- The model uses {len(self.feature_names)} different indicators")
        print(f"- Random Forest combines 100 decision trees for predictions")
        print(f"- Higher accuracy suggests some predictable patterns exist")
        print(f"- Lower accuracy suggests markets are largely unpredictable")
        
        print(f"\nEducational Takeaways:")
        print(f"- This demonstrates machine learning techniques, not investment advice")
        print(f"- Real markets are influenced by factors our model cannot capture")
        print(f"- Professional trading requires risk management, not just predictions")
        print(f"- Pattern recognition skills transfer to many other domains")

def main():
    """
    Main function to run machine learning analysis
    """
    
    print("Apple Stock Machine Learning Predictor")
    print("="*45)
    
    # Load data with technical indicators
    try:
        data = pd.read_csv('apple_stock_with_indicators.csv', index_col=0, parse_dates=True)
        print(f"Loaded {len(data)} days of Apple stock data with indicators")
    except FileNotFoundError:
        print("Error: apple_stock_with_indicators.csv not found!")
        print("Please run technical_indicators.py first")
        return
    
    # Initialize ML predictor
    predictor = StockMLPredictor(data)
    
    # Prepare features for machine learning
    predictor.prepare_ml_features()
    
    # Split and scale data
    X_train, X_test, y_train, y_test = predictor.split_and_scale_data()
    
    # Train model
    predictor.train_model(X_train, y_train)
    
    # Evaluate model
    y_pred, y_pred_proba, accuracy = predictor.evaluate_model(X_test, y_test)
    
    # Analyze feature importance
    feature_importance = predictor.analyze_feature_importance()
    
    # Create visualizations
    predictor.create_prediction_visualization(X_test, y_test, y_pred, y_pred_proba)
    
    # Generate final report
    predictor.generate_prediction_report(accuracy, feature_importance)
    
    print(f"\nAll analysis complete! Check the generated charts and reports.")
    
    return predictor

if __name__ == "__main__":
    predictor = main()