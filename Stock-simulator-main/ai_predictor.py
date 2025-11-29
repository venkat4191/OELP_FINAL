import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
from calculations import StockCalculations

class StockAI:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = MinMaxScaler()
        self.calculations = StockCalculations()
        plt.style.use('dark_background')
    
    def predict_future_prices(self, df, days=150, features=None, anchor_interval=10):
        """Predict future prices using historical patterns and technical analysis."""
        try:
            if df is None or df.empty:
                print("No data available for prediction")
                return None, None
            
            # Calculate technical indicators
            df = self.calculations.calculate_technical_indicators(df)
            if df is None or df.empty:
                print("Failed to calculate technical indicators")
                return None, None
            
            # Prepare features for anchor predictions if provided
            if features is None:
                features = self.calculations.prepare_features(df)
            if features is None or features.empty:
                anchor_targets = {}
            else:
                anchor_targets = {}
                anchor_offsets = list(range(1, days + 1, anchor_interval))
                if anchor_offsets[-1] != days:
                    anchor_offsets.append(days)
                for offset in anchor_offsets:
                    anchor_prediction = self.predict_price(features.copy(), horizon=offset)
                    if anchor_prediction > 0:
                        anchor_targets[offset] = anchor_prediction
            
            # Get current price and historical data
            current_price = float(df['Close'].iloc[-1])
            
            # Calculate historical patterns
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            
            # Calculate trend using multiple timeframes
            short_trend = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100
            medium_trend = (df['Close'].iloc[-1] / df['Close'].iloc[-50] - 1) * 100
            long_trend = (df['Close'].iloc[-1] / df['Close'].iloc[-200] - 1) * 100
            
            # Generate future dates
            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date, periods=days+1, freq='B')[1:]
            
            # Initialize future prices
            future_prices = [current_price]
            
            # Set a fixed random seed for deterministic predictions
            np.random.seed(42)
            
            # Calculate base components
            base_volatility = volatility * 0.01  # Convert to daily volatility
            trend_strength = (short_trend + medium_trend + long_trend) / 300  # Normalized trend
            
            # Anchor helpers
            if anchor_targets:
                anchor_offsets_sorted = sorted(anchor_targets.keys())
                anchor_index = 0
                next_anchor_offset = anchor_offsets_sorted[anchor_index]
                next_anchor_price = anchor_targets[next_anchor_offset]
                prev_anchor_offset = 0
                prev_anchor_price = current_price
            else:
                next_anchor_offset = None
                next_anchor_price = None
                prev_anchor_offset = 0
                prev_anchor_price = current_price
            
            # Generate predictions
            for i in range(days):
                # Create realistic price movement
                daily_volatility = base_volatility * (1 + 0.5 * np.sin(i * 0.1))  # Varying volatility
                trend_effect = trend_strength * (1 + 0.2 * np.cos(i * 0.05))  # Varying trend strength
                
                # Add market cycles
                cycle_effect = 0.5 * np.sin(i * 0.02) + 0.3 * np.cos(i * 0.01)
                
                # Combine all effects
                price_movement = (
                    trend_effect * 0.001 +  # Trend component
                    daily_volatility * np.random.normal(0, 1) +  # Random component
                    cycle_effect * 0.001  # Market cycle component
                )
                
                # Calculate next price
                deterministic_price = future_prices[-1] * (1 + price_movement)
                
                # Blend towards anchor targets when available
                day_offset = i + 1
                if next_anchor_offset is not None:
                    if day_offset == next_anchor_offset:
                        next_price = next_anchor_price
                        prev_anchor_offset = next_anchor_offset
                        prev_anchor_price = next_anchor_price
                        anchor_index += 1
                        if anchor_index < len(anchor_offsets_sorted):
                            next_anchor_offset = anchor_offsets_sorted[anchor_index]
                            next_anchor_price = anchor_targets[next_anchor_offset]
                        else:
                            next_anchor_offset = None
                            next_anchor_price = None
                    else:
                        segment_length = next_anchor_offset - prev_anchor_offset
                        if segment_length <= 0:
                            next_price = deterministic_price
                        else:
                            progress = (day_offset - prev_anchor_offset) / segment_length
                            desired_price = prev_anchor_price + (next_anchor_price - prev_anchor_price) * progress
                            next_price = (deterministic_price + desired_price) / 2
                else:
                    next_price = deterministic_price
                
                # Add realistic price bounds
                max_change = 0.03  # Maximum 3% daily change
                price_change = (next_price - future_prices[-1]) / future_prices[-1]
                if abs(price_change) > max_change:
                    next_price = future_prices[-1] * (1 + np.sign(price_change) * max_change)
                
                future_prices.append(next_price)
            
            return future_dates, future_prices[1:]
            
        except Exception as e:
            print(f"Error predicting future prices: {str(e)}")
            return None, None

    def generate_prediction_graph(self, df, future_dates, future_prices, symbol):
        """Generate comprehensive prediction graph"""
        try:
            if df is None or df.empty or future_dates is None or future_prices is None:
                print("Missing data for prediction graph")
                return None
            
            # Create figure with subplots
            plt.style.use('dark_background')
            fig = plt.figure(figsize=(15, 10))
            
            # Price and Moving Averages
            ax1 = plt.subplot(2, 2, 1)
            ax1.plot(df.index, df['Close'], label='Historical Price', color='#3b82f6', linewidth=2)
            if 'SMA_20' in df.columns and not pd.isna(df['SMA_20'].iloc[-1]):
                ax1.plot(df.index, df['SMA_20'], label='20-day MA', color='#f59e0b', alpha=0.7)
            if 'SMA_50' in df.columns and not pd.isna(df['SMA_50'].iloc[-1]):
                ax1.plot(df.index, df['SMA_50'], label='50-day MA', color='#10b981', alpha=0.7)
            ax1.plot(future_dates, future_prices, label='Predicted Price', color='#22c55e', linestyle='--', linewidth=2)
            
            # Add confidence interval
            std_dev = float(df['Close'].pct_change().dropna().std())
            upper_bound = [price * (1 + 2*std_dev) for price in future_prices]
            lower_bound = [price * (1 - 2*std_dev) for price in future_prices]
            ax1.fill_between(future_dates, lower_bound, upper_bound, color='#22c55e', alpha=0.1)
            
            ax1.set_title(f'{symbol} Price Prediction', fontsize=12, pad=20)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # RSI
            ax2 = plt.subplot(2, 2, 2)
            if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]):
                ax2.plot(df.index, df['RSI'], label='RSI', color='#8b5cf6')
                ax2.axhline(y=70, color='#ef4444', linestyle='--', alpha=0.5)
                ax2.axhline(y=30, color='#22c55e', linestyle='--', alpha=0.5)
                ax2.set_title('Relative Strength Index (RSI)', fontsize=12)
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            # MACD
            ax3 = plt.subplot(2, 2, 3)
            if 'MACD' in df.columns and not pd.isna(df['MACD'].iloc[-1]):
                ax3.plot(df.index, df['MACD'], label='MACD', color='#3b82f6')
                if 'MACD_Signal' in df.columns and not pd.isna(df['MACD_Signal'].iloc[-1]):
                    ax3.plot(df.index, df['MACD_Signal'], label='Signal', color='#f59e0b')
                if 'MACD_Hist' in df.columns and not pd.isna(df['MACD_Hist'].iloc[-1]):
                    ax3.bar(df.index, df['MACD_Hist'], label='Histogram', color='#10b981', alpha=0.5)
                ax3.set_title('Moving Average Convergence Divergence (MACD)', fontsize=12)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            # Bollinger Bands
            ax4 = plt.subplot(2, 2, 4)
            ax4.plot(df.index, df['Close'], label='Price', color='#3b82f6')
            if 'BB_Upper' in df.columns and not pd.isna(df['BB_Upper'].iloc[-1]):
                ax4.plot(df.index, df['BB_Upper'], label='Upper Band', color='#ef4444', alpha=0.7)
            if 'BB_Middle' in df.columns and not pd.isna(df['BB_Middle'].iloc[-1]):
                ax4.plot(df.index, df['BB_Middle'], label='Middle Band', color='#f59e0b', alpha=0.7)
            if 'BB_Lower' in df.columns and not pd.isna(df['BB_Lower'].iloc[-1]):
                ax4.plot(df.index, df['BB_Lower'], label='Lower Band', color='#22c55e', alpha=0.7)
            ax4.set_title('Bollinger Bands', fontsize=12)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert plot to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#1e293b')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_str
            
        except Exception as e:
            print(f"Error generating prediction graph: {str(e)}")
            return None

    def generate_future_prediction_graph(self, df, future_dates, future_prices, symbol):
        """Generate graph showing historical and predicted future prices"""
        try:
            if df.empty or future_dates is None or future_prices is None:
                print("Missing data for future prediction graph")
                return None

            # Create figure with a dark background
            plt.style.use('dark_background')
            fig = plt.figure(figsize=(15, 8))
            
            # Plot historical prices
            plt.plot(df.index, df['Close'], label='Historical Prices', color='#3b82f6', linewidth=2)
            
            # Plot predicted prices
            plt.plot(future_dates, future_prices, label='Predicted Prices', color='#22c55e', linestyle='--', linewidth=2)
            
            # Add confidence interval
            std_dev = np.std(df['Close'].pct_change().dropna())
            upper_bound = [price * (1 + 2*std_dev) for price in future_prices]
            lower_bound = [price * (1 - 2*std_dev) for price in future_prices]
            plt.fill_between(future_dates, lower_bound, upper_bound, color='#22c55e', alpha=0.1)
            
            # Add trend line
            z = np.polyfit(range(len(future_prices)), future_prices, 1)
            p = np.poly1d(z)
            plt.plot(future_dates, p(range(len(future_prices))), '--', color='#f59e0b', alpha=0.5, label='Trend Line')
            
            # Customize the plot
            plt.title(f'{symbol} Price Prediction (Next 5 Months)', fontsize=14, pad=20, color='white')
            plt.xlabel('Date', fontsize=12, color='white')
            plt.ylabel('Price (₹)', fontsize=12, color='white')
            plt.grid(True, alpha=0.3)
            plt.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, color='white')
            plt.yticks(color='white')
            
            # Set background color
            fig.patch.set_facecolor('#1e293b')
            plt.gca().set_facecolor('#1e293b')
            
            # Adjust layout
            plt.tight_layout()
            
            # Convert plot to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#1e293b')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_str
            
        except Exception as e:
            print(f"Error generating future prediction graph: {e}")
            return None

    def predict_price(self, features, horizon=1):
        """Predict stock price using Random Forest"""
        try:
            if features.empty:
                return 0
                
            # Prepare target variable (next day's price)
            if horizon < 1:
                horizon = 1
            target = features['Close'].shift(-horizon).dropna()
            features = features[:-horizon]  # Remove rows without targets
            
            if features.empty or target.empty:
                return 0
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train model
            self.model.fit(features_scaled, target)
            
            # Predict next price
            last_features = features_scaled[-1].reshape(1, -1)
            predicted_price = self.model.predict(last_features)[0]
            
            return predicted_price
            
        except Exception as e:
            print(f"Error in price prediction: {str(e)}")
            return 0

    def predict_stock(self, symbol):
        """Predict stock price using all available data"""
        try:
            # Add .NS suffix for Indian stocks if not present
            if not symbol.endswith('.NS'):
                symbol = f"{symbol}.NS"
            
            # Get stock data
            stock_data = self.calculations.get_stock_data(symbol)
            if stock_data.empty:
                print(f"No data found for {symbol}")
                return {
                    'current_price': 'N/A',
                    'predicted_price': 'N/A',
                    'confidence': 0,
                    'market_regime': 'Unknown',
                    'sentiment_score': 0,
                    'technical_indicators': {
                        'rsi': 0,
                        'macd': 0,
                        'bollinger_bands': {'upper': 0, 'lower': 0}
                    },
                    'analysis_graph': None,
                    'future_prediction_graph': None,
                    'market_analysis': None,
                    'future_prices': None
                }
            
            # Get current price
            current_price = stock_data['Close'].iloc[-1]
            
            # Calculate technical indicators
            stock_data = self.calculations.calculate_technical_indicators(stock_data)
            if stock_data is None:
                print("Failed to calculate technical indicators")
                return {
                    'current_price': current_price,
                    'predicted_price': 'N/A',
                    'confidence': 0,
                    'market_regime': 'Unknown',
                    'sentiment_score': 0,
                    'technical_indicators': {
                        'rsi': 0,
                        'macd': 0,
                        'bollinger_bands': {'upper': 0, 'lower': 0}
                    },
                    'analysis_graph': None,
                    'future_prediction_graph': None,
                    'market_analysis': None,
                    'future_prices': None
                }
            
            # Prepare features for prediction
            features = self.calculations.prepare_features(stock_data)
            if features.empty:
                print("No features available for prediction")
                return {
                    'current_price': current_price,
                    'predicted_price': 'N/A',
                    'confidence': 0,
                    'market_regime': 'Unknown',
                    'sentiment_score': 0,
                    'technical_indicators': {
                        'rsi': 0,
                        'macd': 0,
                        'bollinger_bands': {'upper': 0, 'lower': 0}
                    },
                    'analysis_graph': None,
                    'future_prediction_graph': None,
                    'market_analysis': None,
                    'future_prices': None
                }
            
            # Generate analysis graphs
            analysis_graph = self.calculations.generate_analysis_graphs(stock_data, symbol)
            
            # Generate future price predictions
            print("Generating future price predictions...")
            future_dates, future_prices = self.predict_future_prices(stock_data, features=features)
            if future_dates is not None and future_prices is not None:
                print("Generating future prediction graph...")
                future_prediction_graph = self.generate_future_prediction_graph(stock_data, future_dates, future_prices, symbol)
            else:
                print("Failed to generate future price predictions")
                future_prediction_graph = None
                future_prices = None
            
            # Generate market analysis
            market_analysis = self.calculations.generate_market_analysis(stock_data)
            
            # Train model and predict
            predicted_price = self.predict_price(features)
            
            # Get sentiment score
            sentiment_score = self.calculations.analyze_sentiment(symbol)
            
            # Determine market regime
            market_regime = self.calculations.analyze_market_regime(stock_data)
            
            # Calculate confidence
            confidence = self.calculations.calculate_confidence(stock_data)
            
            return {
                'current_price': round(current_price, 2),
                'predicted_price': round(predicted_price, 2),
                'confidence': round(confidence, 2),
                'market_regime': market_regime,
                'sentiment_score': round(sentiment_score, 2),
                'technical_indicators': {
                    'rsi': float(stock_data['RSI'].iloc[-1]) if 'RSI' in stock_data.columns and not pd.isna(stock_data['RSI'].iloc[-1]) else 0,
                    'macd': float(stock_data['MACD'].iloc[-1]) if 'MACD' in stock_data.columns and not pd.isna(stock_data['MACD'].iloc[-1]) else 0,
                    'bollinger_bands': {
                        'upper': float(stock_data['BB_Upper'].iloc[-1]) if 'BB_Upper' in stock_data.columns and not pd.isna(stock_data['BB_Upper'].iloc[-1]) else 0,
                        'lower': float(stock_data['BB_Lower'].iloc[-1]) if 'BB_Lower' in stock_data.columns and not pd.isna(stock_data['BB_Lower'].iloc[-1]) else 0
                    }
                },
                'analysis_graph': analysis_graph,
                'future_prediction_graph': future_prediction_graph,
                'market_analysis': market_analysis,
                'future_prices': [round(price, 2) for price in future_prices] if future_prices else None
            }
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {
                'current_price': 'N/A',
                'predicted_price': 'N/A',
                'confidence': 0,
                'market_regime': 'Error',
                'sentiment_score': 0,
                'technical_indicators': {
                    'rsi': 0,
                    'macd': 0,
                    'bollinger_bands': {'upper': 0, 'lower': 0}
                },
                'analysis_graph': None,
                'future_prediction_graph': None,
                'market_analysis': None,
                'future_prices': None
            } 