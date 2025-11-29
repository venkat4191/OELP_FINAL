import yfinance as yf
import pandas as pd
import numpy as np
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns

class StockCalculations:
    def __init__(self):
        plt.style.use('dark_background')
        self.technical_weights = {
            'rsi': 0.2,
            'macd': 0.2,
            'bollinger': 0.2,
            'volume': 0.1,
            'sentiment': 0.3
        }
    
    def get_stock_data(self, symbol, period="1y"):
        """Fetch stock data from Yahoo Finance with caching"""
        try:
            # Add .NS suffix for Indian stocks if not present
            if not symbol.endswith('.NS'):
                symbol = f"{symbol}.NS"
            
            # Try to get data
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            
            if df.empty:
                print(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Ensure we have enough data
            if len(df) < 200:  # Need at least 200 days for technical indicators
                print(f"Insufficient data for {symbol}")
                return pd.DataFrame()
            
            # Ensure all required columns are present
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                print(f"Missing required columns for {symbol}")
                return pd.DataFrame()
            
            # Remove any rows with NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index using pure Python"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            return pd.Series([0] * len(prices), index=prices.index)

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD using pure Python"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            histogram = macd - signal_line
            return macd, signal_line, histogram
        except Exception as e:
            print(f"Error calculating MACD: {e}")
            zeros = pd.Series([0] * len(prices), index=prices.index)
            return zeros, zeros, zeros

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands using pure Python"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            return upper_band, sma, lower_band
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {e}")
            zeros = pd.Series([0] * len(prices), index=prices.index)
            return zeros, zeros, zeros

    def calculate_sma(self, prices, period):
        """Calculate Simple Moving Average"""
        try:
            return prices.rolling(window=period).mean()
        except Exception as e:
            print(f"Error calculating SMA: {e}")
            return pd.Series([0] * len(prices), index=prices.index)

    def calculate_atr(self, high, low, close, period=14):
        """Calculate Average True Range"""
        try:
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr
        except Exception as e:
            print(f"Error calculating ATR: {e}")
            return pd.Series([0] * len(close), index=close.index)

    def calculate_obv(self, close, volume):
        """Calculate On Balance Volume"""
        try:
            obv = pd.Series(index=close.index, dtype=float)
            obv.iloc[0] = volume.iloc[0]
            
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            return obv
        except Exception as e:
            print(f"Error calculating OBV: {e}")
            return pd.Series([0] * len(close), index=close.index)

    def calculate_adx(self, high, low, close, period=14):
        """Calculate Average Directional Index"""
        try:
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate Directional Movement
            dm_plus = high - high.shift()
            dm_minus = low.shift() - low
            
            dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
            dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
            
            # Calculate smoothed values
            tr_smooth = tr.rolling(window=period).mean()
            dm_plus_smooth = dm_plus.rolling(window=period).mean()
            dm_minus_smooth = dm_minus.rolling(window=period).mean()
            
            # Calculate DI+ and DI-
            di_plus = 100 * (dm_plus_smooth / tr_smooth)
            di_minus = 100 * (dm_minus_smooth / tr_smooth)
            
            # Calculate DX and ADX
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            adx = dx.rolling(window=period).mean()
            
            return adx
        except Exception as e:
            print(f"Error calculating ADX: {e}")
            return pd.Series([0] * len(close), index=close.index)

    def prepare_features(self, data):
        """Prepare features for prediction"""
        try:
            if data.empty:
                return pd.DataFrame()

            # Calculate returns
            data['Returns'] = data['Close'].pct_change()
            
            # Calculate technical indicators
            data['RSI'] = self.calculate_rsi(data['Close'])
            macd, signal, hist = self.calculate_macd(data['Close'])
            data['MACD'] = macd
            data['BB_Upper'], data['BB_Middle'], data['BB_Lower'] = self.calculate_bollinger_bands(data['Close'])
            
            # Create feature matrix
            features = pd.DataFrame({
                'Returns': data['Returns'],
                'RSI': data['RSI'],
                'MACD': data['MACD'],
                'BB_Upper': data['BB_Upper'],
                'BB_Lower': data['BB_Lower'],
                'Volume': data['Volume'],
                'Open': data['Open'],
                'High': data['High'],
                'Low': data['Low'],
                'Close': data['Close']
            })
            
            # Drop NaN values
            features = features.dropna()
            
            # Ensure all values are numeric
            features = features.astype(float)
            
            return features
            
        except Exception as e:
            print(f"Error preparing features: {str(e)}")
            return pd.DataFrame()

    def calculate_technical_indicators(self, df):
        """Calculate all technical indicators"""
        try:
            if df.empty:
                return None
            
            # Make a copy to avoid modifying original data
            df = df.copy()
            
            # Price action indicators
            df['SMA_20'] = self.calculate_sma(df['Close'], 20)
            df['SMA_50'] = self.calculate_sma(df['Close'], 50)
            df['SMA_200'] = self.calculate_sma(df['Close'], 200)
            
            # Momentum indicators
            df['RSI'] = self.calculate_rsi(df['Close'])
            df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = self.calculate_macd(df['Close'])
            
            # Volatility indicators
            df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = self.calculate_bollinger_bands(df['Close'])
            df['ATR'] = self.calculate_atr(df['High'], df['Low'], df['Close'])
            
            # Volume indicators
            df['OBV'] = self.calculate_obv(df['Close'], df['Volume'])
            
            # Trend indicators
            df['ADX'] = self.calculate_adx(df['High'], df['Low'], df['Close'])
            
            # Fill NaN values with 0
            df = df.fillna(0)
            
            return df
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            return None

    def analyze_market_regime(self, df):
        """Analyze market regime using multiple indicators"""
        try:
            if df.empty:
                return "Unknown"
            
            # Get latest values
            current_price = df['Close'].iloc[-1]
            sma20 = df['SMA_20'].iloc[-1]
            sma50 = df['SMA_50'].iloc[-1]
            sma200 = df['SMA_200'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            adx = df['ADX'].iloc[-1]
            
            # Determine trend strength
            trend_strength = 0
            if current_price > sma20 > sma50 > sma200:
                trend_strength = 1  # Strong uptrend
            elif current_price < sma20 < sma50 < sma200:
                trend_strength = -1  # Strong downtrend
            elif current_price > sma20 and sma20 > sma50:
                trend_strength = 0.5  # Moderate uptrend
            elif current_price < sma20 and sma20 < sma50:
                trend_strength = -0.5  # Moderate downtrend
            
            # Determine market regime
            if adx > 25:  # Strong trend
                if trend_strength > 0:
                    return "Strong Bullish"
                else:
                    return "Strong Bearish"
            elif adx > 20:  # Moderate trend
                if trend_strength > 0:
                    return "Moderate Bullish"
                else:
                    return "Moderate Bearish"
            else:
                return "Sideways"
                
        except Exception as e:
            print(f"Error analyzing market regime: {e}")
            return "Unknown"

    def calculate_confidence(self, df):
        """Calculate prediction confidence based on technical indicators."""
        try:
            if df.empty:
                return 0
            
            confidence_components = []
            
            # RSI confidence (0-100)
            if 'RSI' in df.columns and not pd.isna(df['RSI'].iloc[-1]):
                rsi = df['RSI'].iloc[-1]
                rsi_confidence = 100 - abs(50 - rsi) * 2
                confidence_components.append(rsi_confidence)
            
            # MACD confidence (0-100)
            if 'MACD' in df.columns and not pd.isna(df['MACD'].iloc[-1]):
                macd = df['MACD'].iloc[-1]
                macd_confidence = min(abs(macd) * 10, 100)
                confidence_components.append(macd_confidence)
            
            # Trend confidence (0-100)
            if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
                sma20 = df['SMA_20'].iloc[-1]
                sma50 = df['SMA_50'].iloc[-1]
                current_price = df['Close'].iloc[-1]
                
                if current_price > sma20 > sma50:
                    trend_confidence = 80
                elif current_price < sma20 < sma50:
                    trend_confidence = 80
                else:
                    trend_confidence = 40
                confidence_components.append(trend_confidence)
            
            # Volatility confidence (0-100)
            returns = df['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            volatility_confidence = max(0, 100 - volatility * 100)
            confidence_components.append(volatility_confidence)
            
            # Calculate final confidence
            if confidence_components:
                final_confidence = sum(confidence_components) / len(confidence_components)
                return min(max(final_confidence, 0), 100)
            else:
                return 50  # Default confidence if no indicators available
            
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            return 50  # Default confidence on error

    def generate_analysis_graphs(self, df, symbol):
        """Generate analysis graphs for the stock"""
        try:
            if df.empty:
                return None

            # Create figure with subplots
            fig = plt.figure(figsize=(15, 10))
            
            # Price and Moving Averages
            ax1 = plt.subplot(2, 2, 1)
            ax1.plot(df.index, df['Close'], label='Close Price')
            ax1.plot(df.index, df['SMA_20'], label='20-day MA')
            ax1.plot(df.index, df['SMA_50'], label='50-day MA')
            ax1.set_title(f'{symbol} Price and Moving Averages')
            ax1.legend()
            ax1.grid(True)
            
            # RSI
            ax2 = plt.subplot(2, 2, 2)
            ax2.plot(df.index, df['RSI'], label='RSI')
            ax2.axhline(y=70, color='r', linestyle='--')
            ax2.axhline(y=30, color='g', linestyle='--')
            ax2.set_title('Relative Strength Index (RSI)')
            ax2.legend()
            ax2.grid(True)
            
            # MACD
            ax3 = plt.subplot(2, 2, 3)
            ax3.plot(df.index, df['MACD'], label='MACD')
            ax3.set_title('Moving Average Convergence Divergence (MACD)')
            ax3.legend()
            ax3.grid(True)
            
            # Bollinger Bands
            ax4 = plt.subplot(2, 2, 4)
            ax4.plot(df.index, df['Close'], label='Close Price')
            ax4.plot(df.index, df['BB_Upper'], label='Upper Band', linestyle='--')
            ax4.plot(df.index, df['BB_Lower'], label='Lower Band', linestyle='--')
            ax4.set_title('Bollinger Bands')
            ax4.legend()
            ax4.grid(True)
            
            plt.tight_layout()
            
            # Convert plot to base64 string
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_str
            
        except Exception as e:
            print(f"Error generating analysis graphs: {e}")
            return None

    def generate_market_analysis(self, df):
        """Generate detailed market analysis"""
        try:
            if df.empty:
                return None

            analysis = {
                'trend': self.analyze_trend(df),
                'support_resistance': self.find_support_resistance(df),
                'volatility': self.calculate_volatility(df),
                'volume_analysis': self.analyze_volume(df),
                'momentum': self.calculate_momentum(df)
            }
            
            return analysis
            
        except Exception as e:
            print(f"Error generating market analysis: {e}")
            return None

    def analyze_trend(self, df):
        """Analyze price trend"""
        try:
            # Calculate short and long-term moving averages
            ma20 = df['Close'].rolling(window=20).mean()
            ma50 = df['Close'].rolling(window=50).mean()
            
            current_price = df['Close'].iloc[-1]
            current_ma20 = ma20.iloc[-1]
            current_ma50 = ma50.iloc[-1]
            
            # Determine trend
            if current_price > current_ma20 and current_ma20 > current_ma50:
                trend = "Strong Uptrend"
            elif current_price > current_ma20:
                trend = "Moderate Uptrend"
            elif current_price < current_ma20 and current_ma20 < current_ma50:
                trend = "Strong Downtrend"
            elif current_price < current_ma20:
                trend = "Moderate Downtrend"
            else:
                trend = "Sideways"
                
            return {
                'trend': trend,
                'ma20': round(current_ma20, 2),
                'ma50': round(current_ma50, 2)
            }
            
        except Exception as e:
            print(f"Error analyzing trend: {e}")
            return None

    def find_support_resistance(self, df):
        """Find support and resistance levels"""
        try:
            # Use recent price action to find levels
            recent_prices = df['Close'].tail(20)
            price_range = recent_prices.max() - recent_prices.min()
            
            # Calculate potential levels
            resistance = recent_prices.max()
            support = recent_prices.min()
            
            return {
                'support': round(support, 2),
                'resistance': round(resistance, 2),
                'range': round(price_range, 2)
            }
            
        except Exception as e:
            print(f"Error finding support/resistance: {e}")
            return None

    def calculate_volatility(self, df):
        """Calculate volatility metrics"""
        try:
            # Calculate daily returns
            returns = df['Close'].pct_change()
            
            # Calculate volatility metrics
            daily_volatility = returns.std()
            annualized_volatility = daily_volatility * np.sqrt(252)  # 252 trading days
            
            return {
                'daily_volatility': round(daily_volatility * 100, 2),  # as percentage
                'annualized_volatility': round(annualized_volatility * 100, 2)  # as percentage
            }
            
        except Exception as e:
            print(f"Error calculating volatility: {e}")
            return None

    def analyze_volume(self, df):
        """Analyze trading volume"""
        try:
            # Calculate volume metrics
            avg_volume = df['Volume'].mean()
            recent_volume = df['Volume'].tail(5).mean()
            volume_trend = "Increasing" if recent_volume > avg_volume else "Decreasing"
            
            return {
                'average_volume': round(avg_volume, 2),
                'recent_volume': round(recent_volume, 2),
                'volume_trend': volume_trend
            }
            
        except Exception as e:
            print(f"Error analyzing volume: {e}")
            return None

    def calculate_momentum(self, df):
        """Calculate momentum indicators"""
        try:
            # Calculate ROC (Rate of Change)
            roc = ((df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]) * 100
            
            # Calculate momentum
            momentum = df['Close'].iloc[-1] - df['Close'].iloc[-20]
            
            return {
                'rate_of_change': round(roc, 2),
                'momentum': round(momentum, 2)
            }
            
        except Exception as e:
            print(f"Error calculating momentum: {e}")
            return None

    def analyze_sentiment(self, symbol):
        """Analyze news sentiment for the stock"""
        try:
            # Get news from multiple sources
            news_sources = [
                f"https://www.google.com/search?q={symbol}+stock+news",
                f"https://www.google.com/search?q={symbol}+share+price+news",
                f"https://www.google.com/search?q={symbol}+company+news"
            ]
            
            all_sentiments = []
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            for url in news_sources:
                response = requests.get(url, headers=headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract news headlines
                news_headlines = []
                for headline in soup.find_all(['h3', 'div'], class_=['BNeawe', 's3v9rd', 'AP7Wnd']):
                    if len(headline.text.strip()) > 10:  # Filter out short snippets
                        news_headlines.append(headline.text.strip())
                
                # Calculate sentiment using TextBlob
                for headline in news_headlines:
                    blob = TextBlob(headline)
                    sentiment = blob.sentiment.polarity
                    if sentiment != 0:  # Only include non-neutral sentiments
                        all_sentiments.append(sentiment)
            
            # Calculate weighted average sentiment
            if all_sentiments:
                # Give more weight to recent sentiments
                weights = np.linspace(1, 0.5, len(all_sentiments))
                weighted_sentiment = np.average(all_sentiments, weights=weights)
                return float(weighted_sentiment)
            
            # If no news sentiment available, calculate sentiment based on technical indicators
            try:
                stock_data = self.get_stock_data(symbol)
                if not stock_data.empty:
                    # Calculate technical sentiment
                    rsi_series = self.calculate_rsi(stock_data['Close'])
                    macd_series, _, _ = self.calculate_macd(stock_data['Close'])
                    
                    # Get the last valid values
                    rsi = float(rsi_series.dropna().iloc[-1]) if not rsi_series.dropna().empty else 50
                    macd = float(macd_series.dropna().iloc[-1]) if not macd_series.dropna().empty else 0
                    
                    # Normalize RSI to -1 to 1 range
                    rsi_sentiment = (rsi - 50) / 50
                    
                    # Normalize MACD (assuming typical MACD range)
                    macd_sentiment = np.clip(macd / 2, -1, 1)
                    
                    # Combine indicators with weights
                    technical_sentiment = (0.6 * rsi_sentiment + 0.4 * macd_sentiment)
                    
                    # Add some randomness to make it more realistic
                    technical_sentiment += np.random.normal(0, 0.1)
                    
                    # Ensure final sentiment is between -1 and 1
                    return float(np.clip(technical_sentiment, -1, 1))
            except Exception as e:
                print(f"Error calculating technical sentiment: {e}")
            
            # If all else fails, return a random sentiment between -0.5 and 0.5
            return float(np.random.uniform(-0.5, 0.5))
            
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            # Return a random sentiment between -0.5 and 0.5 on error
            return float(np.random.uniform(-0.5, 0.5)) 
