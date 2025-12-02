import os
import secrets
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from dotenv import load_dotenv
import json
import re
from ai_predictor import StockAI
from werkzeug.security import check_password_hash, generate_password_hash
import google.generativeai as genai
from functools import lru_cache
import time

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyDuQ9w3pthLreRzDGNvHbqqaLv483RgcrM')

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# In-memory cache for stock suggestions (TTL: 1 hour)
_stock_search_cache = {}
_cache_ttl = 3600  # 1 hour in seconds

# Common stock mappings for instant results (no API call needed)
COMMON_STOCKS = {
    'reliance': [{'symbol': 'RELIANCE', 'company_name': 'Reliance Industries Limited'}],
    'tata motors': [{'symbol': 'TATAMOTORS', 'company_name': 'Tata Motors Limited'}],
    'tcs': [{'symbol': 'TCS', 'company_name': 'Tata Consultancy Services Limited'}],
    'infosys': [{'symbol': 'INFY', 'company_name': 'Infosys Limited'}],
    'hdfc bank': [{'symbol': 'HDFCBANK', 'company_name': 'HDFC Bank Limited'}],
    'icici bank': [{'symbol': 'ICICIBANK', 'company_name': 'ICICI Bank Limited'}],
    'bharat electronics': [{'symbol': 'BEL', 'company_name': 'Bharat Electronics Limited'}],
    'bharath electronics': [{'symbol': 'BEL', 'company_name': 'Bharat Electronics Limited'}],
    'wipro': [{'symbol': 'WIPRO', 'company_name': 'Wipro Limited'}],
    'hcl': [{'symbol': 'HCLTECH', 'company_name': 'HCL Technologies Limited'}],
    'bharti airtel': [{'symbol': 'BHARTIARTL', 'company_name': 'Bharti Airtel Limited'}],
    'airtel': [{'symbol': 'BHARTIARTL', 'company_name': 'Bharti Airtel Limited'}],
}

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'postgresql://localhost/stock_simulator')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    balance = db.Column(db.Float, default=100000.0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    portfolios = db.relationship('Portfolio', backref='user', lazy=True)

class Portfolio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    shares = db.Column(db.Integer, nullable=False)
    avg_price = db.Column(db.Float, nullable=False)
    purchased_at = db.Column(db.DateTime, default=datetime.utcnow)
    stop_loss = db.Column(db.Float, nullable=True)  # Stop loss price (optional)

class IntradayPosition(db.Model):
    """Stores intraday (margin) positions that must be squared off same day."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    shares = db.Column(db.Integer, nullable=False)
    entry_price = db.Column(db.Float, nullable=False)
    leverage = db.Column(db.Float, default=5.0)  # e.g. 5x
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_closed = db.Column(db.Boolean, default=False)
    exit_price = db.Column(db.Float, nullable=True)
    realised_pl = db.Column(db.Float, nullable=True)
    stop_loss = db.Column(db.Float, nullable=True)  # Stop loss price (optional)

class TransactionHistory(db.Model):
    """Stores all buy/sell transactions for performance tracking."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    transaction_type = db.Column(db.String(10), nullable=False)  # 'BUY' or 'SELL'
    trade_type = db.Column(db.String(20), nullable=False)  # 'DELIVERY' or 'INTRADAY'
    symbol = db.Column(db.String(20), nullable=False)
    shares = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Float, nullable=False)
    total_value = db.Column(db.Float, nullable=False)
    realised_pl = db.Column(db.Float, default=0.0)  # P/L for sell transactions
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create database tables
with app.app_context():
    db.create_all()

# The get_news function fetches the latest business news from News API
# The display_portfolio function generates a summary of the user's portfolio including current values and profit/loss
# The generate_portfolio_pie_chart function creates a pie chart to visualize portfolio performance
# The buy_stock and sell_stock functions handle buying and selling of stocks respectively

def scrape_stock_news(limit=12):
    """Scrape latest stock market news from trusted Indian financial portals."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        )
    }
    
    def fetch_soup(url):
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')
    
    def parse_moneycontrol():
        url = "https://www.moneycontrol.com/news/business/markets/"
        soup = fetch_soup(url)
        articles = []
        for card in soup.select('li.clearfix')[:6]:
            title_el = card.select_one('h2 a')
            summary_el = card.select_one('p')
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            summary = summary_el.get_text(strip=True) if summary_el else "Click to read the full story on Moneycontrol."
            link = title_el.get('href', url)
            articles.append({
                "title": title,
                "description": summary,
                "url": link,
                "source": "Moneycontrol"
            })
        return articles
    
    def parse_economic_times():
        url = "https://economictimes.indiatimes.com/markets/stocks/news"
        soup = fetch_soup(url)
        articles = []
        for block in soup.select('div.eachStory, div.story-box')[:6]:
            title_el = block.select_one('a')
            summary_el = block.select_one('p')
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            summary = summary_el.get_text(strip=True) if summary_el else "Developing story from Economic Times."
            link = title_el.get('href', url)
            link = urljoin("https://economictimes.indiatimes.com", link)
            articles.append({
                "title": title,
                "description": summary,
                "url": link,
                "source": "Economic Times"
            })
        return articles
    
    def parse_livemint():
        url = "https://www.livemint.com/market/stock-market-news"
        soup = fetch_soup(url)
        articles = []
        for block in soup.select('div.listingNew, li.listingNew')[:6]:
            title_el = block.select_one('a')
            summary_el = block.select_one('p')
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            summary = summary_el.get_text(strip=True) if summary_el else "Latest update from LiveMint."
            link = title_el.get('href', url)
            link = urljoin("https://www.livemint.com", link)
            articles.append({
                "title": title,
                "description": summary,
                "url": link,
                "source": "LiveMint"
            })
        return articles
    
    def parse_ndtv():
        """Scrape latest markets news from NDTV Profit."""
        url = "https://www.ndtv.com/business/latest"
        soup = fetch_soup(url)
        articles = []
        # NDTV uses story cards with <a> and <p>
        for card in soup.select('div.new_storylister, div.storylist, div.lisingNews')[:6]:
            title_el = card.select_one('a')
            summary_el = card.select_one('p')
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            if not title:
                continue
            summary = summary_el.get_text(strip=True) if summary_el else "Click to read the full story on NDTV Profit."
            link = title_el.get('href', url)
            articles.append({
                "title": title,
                "description": summary,
                "url": link,
                "source": "NDTV Profit"
            })
        return articles
    
    def parse_business_standard():
        """Scrape latest markets news from Business Standard."""
        url = "https://www.business-standard.com/markets/news"
        soup = fetch_soup(url)
        articles = []
        for card in soup.select('div.listing-txt, div.row_listing, li.clearfix')[:6]:
            title_el = card.select_one('h2 a, h3 a, a')
            summary_el = card.select_one('p')
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            if not title:
                continue
            summary = summary_el.get_text(strip=True) if summary_el else "Developing story from Business Standard."
            link = title_el.get('href', url)
            link = urljoin("https://www.business-standard.com", link)
            articles.append({
                "title": title,
                "description": summary,
                "url": link,
                "source": "Business Standard"
            })
        return articles
    
    def parse_financial_express():
        """Scrape latest markets news from Financial Express."""
        url = "https://www.financialexpress.com/market/"
        soup = fetch_soup(url)
        articles = []
        for card in soup.select('div.listitembx, div.listing, li')[:6]:
            title_el = card.select_one('h3 a, h2 a, a')
            summary_el = card.select_one('p')
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            if not title:
                continue
            summary = summary_el.get_text(strip=True) if summary_el else "Latest update from Financial Express."
            link = title_el.get('href', url)
            link = urljoin("https://www.financialexpress.com", link)
            articles.append({
                "title": title,
                "description": summary,
                "url": link,
                "source": "Financial Express"
            })
        return articles
    
    # Add more sources here – they will be mixed later so you see a variety
    aggregators = [
        parse_moneycontrol,
        parse_economic_times,
        parse_ndtv,
        parse_business_standard,
        parse_financial_express,
    ]
    collected = []
    
    # Collect articles from all sources
    for scraper in aggregators:
        try:
            collected.extend(scraper())
        except Exception as e:
            print(f"Error scraping news: {e}")
            continue
    
    # Remove duplicates by URL
    unique = []
    seen = set()
    for article in collected:
        url = article.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(article)
    
    if not unique:
        print("News scraping failed, using fallback stories.")
        return [
            {
                "title": "Unable to fetch live stock news",
                "description": "We could not reach our news sources right now. Please try again in a few minutes.",
                "url": "https://www.moneycontrol.com/news/markets/",
                "source": "System"
            }
        ]
    
    # Mix sources so the list is interleaved (Moneycontrol, ET, NDTV, ...)
    source_order = ["Moneycontrol", "Economic Times", "NDTV Profit"]
    buckets = {src: [] for src in source_order}
    buckets["Other"] = []
    
    for article in unique:
        src = article.get("source", "Other")
        if src not in buckets:
            src = "Other"
        buckets[src].append(article)
    
    mixed = []
    # Round-robin pick from each source to ensure variety
    while len(mixed) < limit and any(buckets[src] for src in buckets):
        for src in source_order + ["Other"]:
            if buckets[src]:
                mixed.append(buckets[src].pop(0))
                if len(mixed) >= limit:
                    break
    
    return mixed

def display_portfolio(portfolio_dict, user_balance):
    portfolio_info = {
        'total_value': 0,
        'total_investment': 0,
        'return_percentage': 0,
        'holdings': [],
        'best_performer': {'symbol': 'N/A', 'return_percentage': 0},
        'worst_performer': {'symbol': 'N/A', 'return_percentage': 0},
        'beta': 0,
        'sharpe_ratio': 0
    }
    
    total_investment = 0
    current_portfolio_value = 0
    best_return = float('-inf')
    worst_return = float('inf')
    
    for symbol, shares in portfolio_dict.items():
        try:
            stock = yf.Ticker(symbol + ".NS")
            current_price = stock.history(period="1d")["Close"].iloc[-1]
            # Convert NumPy types to Python types
            current_price = float(current_price)
            current_value = current_price * shares
            
            # Get average purchase price from database
            user_portfolio_items = Portfolio.query.filter_by(user_id=current_user.id, symbol=symbol).all()
            total_purchased_value = sum(item.avg_price * item.shares for item in user_portfolio_items)
            total_purchased_shares = sum(item.shares for item in user_portfolio_items)
            avg_purchase_price = total_purchased_value / total_purchased_shares if total_purchased_shares > 0 else 0
            
            purchased_value = avg_purchase_price * shares
            total_investment += purchased_value
            current_portfolio_value += current_value
            
            return_percentage = ((current_value - purchased_value) / purchased_value) * 100 if purchased_value > 0 else 0
            
            # Update best and worst performers
            if return_percentage > best_return:
                best_return = return_percentage
                portfolio_info['best_performer'] = {'symbol': symbol, 'return_percentage': float(return_percentage)}
            if return_percentage < worst_return:
                worst_return = return_percentage
                portfolio_info['worst_performer'] = {'symbol': symbol, 'return_percentage': float(return_percentage)}
            
            # Add holding information
            portfolio_info['holdings'].append({
                'symbol': symbol,
                'quantity': shares,
                'avg_price': round(float(avg_purchase_price), 2),
                'current_price': round(float(current_price), 2),
                'value': round(float(current_value), 2),
                'return_percentage': round(float(return_percentage), 2),
                'percentage': round(float((current_value / current_portfolio_value * 100)), 2) if current_portfolio_value > 0 else 0
            })
            
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
    
    # Calculate portfolio metrics
    portfolio_info['total_value'] = round(float(current_portfolio_value), 2)
    portfolio_info['total_investment'] = round(float(total_investment), 2)
    portfolio_info['return_percentage'] = round(float(((current_portfolio_value - total_investment) / total_investment * 100)), 2) if total_investment > 0 else 0
    
    # Generate pie chart for portfolio performance
    generate_portfolio_pie_chart(portfolio_dict)
    
    return portfolio_info

def get_user_holdings_summary(user_id):
    """Aggregate user's holdings with average cost and latest price."""
    user_positions = Portfolio.query.filter_by(user_id=user_id).all()
    holdings_map = {}
    
    for position in user_positions:
        symbol = position.symbol.upper()
        if symbol not in holdings_map:
            holdings_map[symbol] = {'shares': 0, 'total_cost': 0.0}
        holdings_map[symbol]['shares'] += position.shares
        holdings_map[symbol]['total_cost'] += position.shares * position.avg_price
    
    holdings = []
    for symbol, data in holdings_map.items():
        total_shares = data['shares']
        if total_shares <= 0:
            continue
        
        avg_price = data['total_cost'] / total_shares if total_shares else 0
        current_price = None
        current_value = None
        
        try:
            stock = yf.Ticker(symbol + ".NS")
            hist = stock.history(period="1d")
            if not hist.empty and 'Close' in hist.columns:
                current_price = float(hist['Close'].iloc[-1])
                current_value = current_price * total_shares
        except Exception as e:
            print(f"Error fetching current price for {symbol}: {e}")
        
        # Calculate unrealized P/L
        unrealised_pl = None
        unrealised_pl_percent = None
        if current_price is not None and avg_price > 0:
            unrealised_pl = (current_price - avg_price) * total_shares
            unrealised_pl_percent = ((current_price - avg_price) / avg_price) * 100
        
        holdings.append({
            'symbol': symbol,
            'shares': int(total_shares),
            'avg_price': round(float(avg_price), 2),
            'current_price': round(float(current_price), 2) if current_price is not None else None,
            'current_value': round(float(current_value), 2) if current_value is not None else None,
            'unrealised_pl': round(float(unrealised_pl), 2) if unrealised_pl is not None else None,
            'unrealised_pl_percent': round(float(unrealised_pl_percent), 2) if unrealised_pl_percent is not None else None
        })
    
    holdings.sort(key=lambda h: h['symbol'])
    return holdings

def get_user_intraday_positions_summary(user_id):
    """Aggregate user's open intraday positions with latest price and P/L."""
    positions = IntradayPosition.query.filter_by(user_id=user_id, is_closed=False).all()
    intraday_list = []

    for pos in positions:
        symbol = pos.symbol.upper()
        shares = pos.shares
        entry_price = float(pos.entry_price)
        leverage = float(pos.leverage or 5.0)

        current_price = None
        position_value = None
        unrealised_pl = None
        margin_used = (entry_price * shares) / leverage if leverage > 0 else entry_price * shares

        try:
            stock = yf.Ticker(symbol + ".NS")
            hist = stock.history(period="1d")
            if not hist.empty and 'Close' in hist.columns:
                current_price = float(hist['Close'].iloc[-1])
                position_value = current_price * shares
                unrealised_pl = (current_price - entry_price) * shares
        except Exception as e:
            print(f"Error fetching intraday price for {symbol}: {e}")

        intraday_list.append({
            'id': pos.id,
            'symbol': symbol,
            'shares': shares,
            'entry_price': round(entry_price, 2),
            'leverage': leverage,
            'margin_used': round(margin_used, 2),
            'current_price': round(float(current_price), 2) if current_price is not None else None,
            'position_value': round(float(position_value), 2) if position_value is not None else None,
            'unrealised_pl': round(float(unrealised_pl), 2) if unrealised_pl is not None else None,
        })

    intraday_list.sort(key=lambda p: p['symbol'])
    return intraday_list

def generate_portfolio_pie_chart(portfolio):
    try:
        labels = list(portfolio.keys())
        sizes = list(portfolio.values())

        plt.figure(figsize=(6, 6))  # It indicates the figure size
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')  # Equal means it ensures the pie chart is circle 
        plt.title('Portfolio Performance')
        plt.savefig('static/portfolio_pie_chart.png')  # Save the pie chart as a PNG file
        plt.close()
    except Exception as e:
        print(f"Error generating pie chart: {e}")

def buy_stock(portfolio, symbol, shares):
    global balance
    symbol = symbol.upper()  # Always use uppercase, no .NS in portfolio
    print("[BUY] Before:", portfolio)
    try:
        stock = yf.Ticker(symbol + ".NS")
        price = stock.history(period="1d")["Close"].iloc[-1]
        if symbol in venkat:
            venkat[symbol] = float(f"{price:.2f}")
        else:
            venkat[symbol] = float(f"{price:.2f}")

        total_cost = price * shares
        if total_cost > balance:
            return "Insufficient balance to buy!"
        balance = balance - total_cost
        if symbol in portfolio:
            portfolio[symbol] += shares
        else:
            portfolio[symbol] = shares
        print("[BUY] After:", portfolio)
        return f"Bought {shares} shares of {symbol} at ₹{price:.2f} each.\nRemaining balance: ₹{balance:.2f}"
    except Exception as e:
        print("[BUY] Error:", e)
        return "Error: " + str(e)

def sell_stock(portfolio, symbol, shares):
    global balance
    symbol = symbol.upper()  # Always use uppercase, no .NS in portfolio
    print("[SELL] Before:", portfolio)
    if symbol not in portfolio:
        return "You don't own any shares of " + symbol
    if portfolio[symbol] < shares:
        return "You don't have enough shares to sell!"
    stock = yf.Ticker(symbol + ".NS")
    price = stock.history(period="1d")["Close"].iloc[-1]
    balance = balance + (price * shares)
    portfolio[symbol] -= shares
    print("[SELL] After:", portfolio)
    return f"Sold {shares} shares of {symbol} at ₹{price:.3f} each.\nRemaining balance: ₹{balance:.3f}"

def get_stock_prices(tickers):
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            
            # Get history data to get current price and yesterday's close
            hist = stock.history(period="2d")
            
            if not hist.empty and 'Close' in hist.columns:
                # Get current price (most recent close price)
                current_price = float(hist['Close'].iloc[-1])
                
                # Get yesterday's closing price
                percent_change = 0.0
                if len(hist) > 1:
                    # We have at least 2 days of data, so get yesterday's close
                    yesterday_close = float(hist['Close'].iloc[-2])
                    if yesterday_close > 0:
                        percent_change = ((current_price - yesterday_close) / yesterday_close) * 100
                else:
                    # Only one day of data, try to get previous close from info
                    try:
                        info = stock.info
                        if 'previousClose' in info and info['previousClose'] is not None:
                            yesterday_close = float(info['previousClose'])
                            if yesterday_close > 0:
                                percent_change = ((current_price - yesterday_close) / yesterday_close) * 100
                    except:
                        pass
                
                data[ticker] = {
                    'price': f"{current_price:.2f}",
                    'percent_change': round(percent_change, 2)
                }
            else:
                # Fallback: Try using info() method
                try:
                    info = stock.info
                    if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                        current_price = float(info['regularMarketPrice'])
                        yesterday_close = None
                        if 'previousClose' in info and info['previousClose'] is not None:
                            yesterday_close = float(info['previousClose'])
                        
                        percent_change = 0.0
                        if yesterday_close and yesterday_close > 0:
                            percent_change = ((current_price - yesterday_close) / yesterday_close) * 100
                        
                        data[ticker] = {
                            'price': f"{current_price:.2f}",
                            'percent_change': round(percent_change, 2)
                        }
                    else:
                        data[ticker] = {"price": "N/A", "percent_change": 0.0}
                except:
                    data[ticker] = {"price": "N/A", "percent_change": 0.0}
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            data[ticker] = {"price": "N/A", "percent_change": 0.0}
    return data

def google_search(query):
    try:
        url = f"https://www.google.com/search?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(url, headers=headers) #Sends the request to get the information from the google
        response.raise_for_status()  
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('div', class_='BNeawe s3v9rd AP7Wnd') #We got the class code by inspecting the google search page
        return results[0].get_text() if results else "No results found"
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"

def get_stock_symbol_from_name(company_name):
    """
    Use Gemini Flash API to convert company name to NSE stock symbol.
    Returns the stock symbol if found, None otherwise.
    """
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY not found")
        return None
    
    try:
        # Use Gemini Flash model for faster responses
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""You are a stock market expert. Convert the following company name to its NSE (National Stock Exchange of India) stock symbol.

Company name: {company_name}

Rules:
1. Return ONLY the NSE stock symbol in uppercase (e.g., RELIANCE, TCS, INFY, BEL)
2. Do NOT include .NS suffix or any other text
3. Handle common typos (e.g., "electornics" should be treated as "electronics")
4. If the company name is ambiguous, return the most common/largest company
5. If you cannot find the symbol, return "NOT_FOUND"
6. For Indian companies, use NSE symbols. For international companies, use their primary exchange symbol.

Examples:
- "Reliance Industries" -> RELIANCE
- "Tata Consultancy Services" -> TCS
- "Bharat Electronics" or "Bharath Electronics" or "Bharat electornics" -> BEL
- "HDFC Bank" -> HDFCBANK
- "Infosys" -> INFY

Now convert: {company_name}
Return ONLY the symbol in uppercase, nothing else:"""

        response = model.generate_content(prompt)
        
        # Get the text from response - handle different response formats
        if hasattr(response, 'text'):
            symbol = response.text.strip().upper()
        elif hasattr(response, 'candidates') and len(response.candidates) > 0:
            if hasattr(response.candidates[0], 'content'):
                symbol = response.candidates[0].content.parts[0].text.strip().upper()
            else:
                symbol = str(response.candidates[0]).strip().upper()
        else:
            symbol = str(response).strip().upper()
        
        print(f"Gemini raw response for '{company_name}': {symbol}")
        
        # Clean up the response - remove any extra text, newlines, punctuation
        symbol = symbol.replace('\n', ' ').replace('.', '').strip()
        # Extract just the symbol part (first word that looks like a stock symbol)
        words = symbol.split()
        symbol = None
        for word in words:
            word = word.strip().upper()
            # Check if it looks like a stock symbol (alphanumeric, reasonable length)
            if word and word != "NOT_FOUND" and len(word) <= 20 and word.replace('_', '').isalnum():
                symbol = word
                break
        
        if not symbol or symbol == "NOT_FOUND":
            print(f"Gemini returned NOT_FOUND or invalid for '{company_name}'")
            return None
        
        print(f"Extracted symbol: {symbol}")
        
        # Verify the symbol exists by trying to fetch data
        # Try multiple periods in case 1d doesn't work
        periods_to_try = ["5d", "1mo", "1d"]
        for period in periods_to_try:
            try:
                stock = yf.Ticker(symbol + ".NS")
                hist = stock.history(period=period)
                if not hist.empty and 'Close' in hist.columns:
                    print(f"Verified symbol {symbol} exists (using {period} period)")
                    return symbol
            except Exception as e:
                # Continue to next period if this one fails
                print(f"Tried {period} period for {symbol}: {str(e)[:100]}")
                continue
        
        # If all periods fail, still return the symbol if it looks valid
        # (Gemini is usually correct, and yfinance might have temporary issues)
        print(f"Could not verify {symbol} with yfinance, but returning it anyway (Gemini suggested it)")
        return symbol
    except Exception as e:
        print(f"Error using Gemini API for '{company_name}': {e}")
        import traceback
        traceback.print_exc()
        return None

def get_multiple_stock_suggestions(query, max_results=2):
    """
    Use Gemini Flash API to get multiple stock suggestions with company names.
    Returns a list of dictionaries with 'symbol' and 'company_name' keys.
    Optimized for speed - caching, common stocks lookup, no verification.
    """
    # Normalize query for lookup
    query_lower = query.lower().strip()
    
    # Check cache first
    cache_key = query_lower
    if cache_key in _stock_search_cache:
        cached_data, cached_time = _stock_search_cache[cache_key]
        if time.time() - cached_time < _cache_ttl:
            print(f"Cache hit for: {query}")
            return cached_data
    
    # Check common stocks first (instant, no API call)
    if query_lower in COMMON_STOCKS:
        print(f"Common stock match for: {query}")
        result = COMMON_STOCKS[query_lower][:max_results]
        _stock_search_cache[cache_key] = (result, time.time())
        return result
    
    # Check partial matches in common stocks
    for key, value in COMMON_STOCKS.items():
        if query_lower in key or key in query_lower:
            print(f"Partial common stock match for: {query}")
            result = value[:max_results]
            _stock_search_cache[cache_key] = (result, time.time())
            return result
    
    if not GEMINI_API_KEY:
        return []
    
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Ultra-minimal prompt for fastest response
        prompt = f'Find 2 NSE stocks for "{query}". Format: SYMBOL|NAME (one per line).'
        
        response = model.generate_content(prompt)
        
        # Get the text from response - optimized parsing
        if hasattr(response, 'text'):
            text = response.text.strip()
        elif hasattr(response, 'candidates') and len(response.candidates) > 0:
            if hasattr(response.candidates[0], 'content'):
                text = response.candidates[0].content.parts[0].text.strip()
            else:
                text = str(response.candidates[0]).strip()
        else:
            text = str(response).strip()
        
        # Parse the response - optimized, no verification (trust Gemini)
        suggestions = []
        for line in text.split('\n'):
            line = line.strip()
            if '|' in line:
                parts = line.split('|', 1)
                if len(parts) == 2:
                    symbol = parts[0].strip().upper()
                    company_name = parts[1].strip()
                    
                    # Quick validation only
                    if symbol and len(symbol) <= 20 and symbol.replace('_', '').isalnum():
                        suggestions.append({
                            'symbol': symbol,
                            'company_name': company_name
                        })
                        if len(suggestions) >= max_results:
                            break
        
        # Cache the result
        if suggestions:
            _stock_search_cache[cache_key] = (suggestions, time.time())
        
        return suggestions
        
    except Exception as e:
        print(f"Error getting suggestions: {e}")
        return []

# Authentication Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose a different username.', 'error')
            return render_template('register.html')
        
        # Create new user
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('landing'))

# Routes for different pages: index, portfolio, buy, sell, latest_news, analyze
# The index route displays the homepage with scrolling live stock prices
# The portfolio route shows the user's portfolio and its performance
# The buy route allows users to buy stocks
# The sell route allows users to sell stocks
# The latest_news route displays the latest business news
# The analyze route analyzes a stock's performance and displays relevant charts

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/learn')
@login_required
def learn():
    return render_template('learn.html', user=current_user)

@app.route('/')
def index():
    if current_user.is_authenticated:
        # Get prices for some popular Indian stocks
        tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS']
        stock_prices = get_stock_prices(tickers)
        
        # Check if user is authenticated to show user info
        user = current_user if current_user.is_authenticated else None
        return render_template('index.html', stock_prices=stock_prices, user=user)
    else:
        return redirect(url_for('landing'))

@app.route('/portfolio')
@login_required
def view_portfolio():
    # Get user's portfolio from database
    user_portfolio = Portfolio.query.filter_by(user_id=current_user.id).all()
    
    # Convert to dictionary format for existing display function
    portfolio_dict = {}
    for item in user_portfolio:
        if item.symbol in portfolio_dict:
            portfolio_dict[item.symbol] += item.shares
        else:
            portfolio_dict[item.symbol] = item.shares
    
    portfolio_info = display_portfolio(portfolio_dict, current_user.balance)
    return render_template('portfolio.html', portfolio_info=portfolio_info, user=current_user)

@app.route('/buy', methods=['GET', 'POST'])
@login_required
def buy():
    if request.method == 'POST':
        symbol = request.form['symbol'].upper()
        shares = int(request.form['shares'])
        print(f"[BUY] Attempting to buy {shares} shares of {symbol}")
        
        # Get current stock price
        try:
            stock = yf.Ticker(symbol + ".NS")
            price = stock.history(period="1d")["Close"].iloc[-1]
            # Convert NumPy types to Python types
            price = float(price)
            total_cost = price * shares
            
            if total_cost > current_user.balance:
                flash("Insufficient balance to buy!", 'error')
                return render_template('buy.html', message="Insufficient balance", symbol=symbol, shares=shares)
            
            # Update user balance
            current_user.balance = float(current_user.balance - total_cost)
            
            # Add to portfolio
            new_portfolio_item = Portfolio(
                user_id=current_user.id,
                symbol=symbol,
                shares=shares,
                avg_price=price
            )
            db.session.add(new_portfolio_item)
            
            # Record transaction
            transaction = TransactionHistory(
                user_id=current_user.id,
                transaction_type='BUY',
                trade_type='DELIVERY',
                symbol=symbol,
                shares=shares,
                price=price,
                total_value=total_cost,
                realised_pl=0.0
            )
            db.session.add(transaction)
            db.session.commit()
            
            message = f"Bought {shares} shares of {symbol} at ₹{price:.2f} each.\nRemaining balance: ₹{current_user.balance:.2f}"
            flash(message, 'success')
            return render_template('buy.html', message=message, symbol=symbol, shares=shares)
            
        except Exception as e:
            flash(f"Error: {str(e)}", 'error')
            return render_template('buy.html')
    
    return render_template('buy.html')

@app.route('/confirm_buy', methods=['POST'])
@login_required
def confirm_buy():
    symbol = request.form['symbol'].upper()
    shares = int(request.form['shares'])
    trade_type = request.form.get('trade_type', 'delivery')
    stop_loss_str = request.form.get('stop_loss', '').strip()
    
    # Safely convert stop loss to float
    stop_loss_price = None
    if stop_loss_str:
        try:
            stop_loss_price = float(stop_loss_str)
        except ValueError:
            stop_loss_price = None
    
    print(f"[CONFIRM BUY] Attempting to buy {shares} shares of {symbol}, stop_loss: {stop_loss_price}")
    
    try:
        stock = yf.Ticker(symbol + ".NS")
        hist = stock.history(period="1d")
        
        # Check if history is empty
        if hist.empty or 'Close' not in hist.columns:
            flash(f"No price data available for {symbol}. Please check the symbol and try again.", 'error')
            return redirect(url_for('buy'))
        
        price = hist["Close"].iloc[-1]
        # Convert NumPy types to Python types
        price = float(price)
        total_cost = price * shares

        # Validate stop loss if provided
        if stop_loss_price is not None:
            if stop_loss_price >= price:
                flash(f"Stop loss price (₹{stop_loss_price:.2f}) must be less than current price (₹{price:.2f})", 'error')
                return redirect(url_for('buy'))
            if stop_loss_price <= 0:
                flash("Stop loss price must be greater than 0", 'error')
                return redirect(url_for('buy'))

        if trade_type == 'intraday':
            # Intraday margin trade with 5x leverage
            leverage = 5.0
            required_margin = total_cost / leverage

            if required_margin > current_user.balance:
                flash(
                    f"Insufficient margin for intraday order. "
                    f"Required: ₹{required_margin:.2f}, Available: ₹{current_user.balance:.2f}",
                    'error'
                )
                return render_template('message.html', message="Insufficient margin for intraday order")

            # Block only margin from balance
            current_user.balance = float(current_user.balance - required_margin)

            # Create intraday position (not added to portfolio)
            new_position = IntradayPosition(
                user_id=current_user.id,
                symbol=symbol,
                shares=shares,
                entry_price=price,
                leverage=leverage,
                stop_loss=stop_loss_price
            )
            db.session.add(new_position)
            
            # Record transaction
            transaction = TransactionHistory(
                user_id=current_user.id,
                transaction_type='BUY',
                trade_type='INTRADAY',
                symbol=symbol,
                shares=shares,
                price=price,
                total_value=total_cost,
                realised_pl=0.0
            )
            db.session.add(transaction)
            db.session.commit()

            message = (
                f"Intraday (5x) buy: {shares} shares of {symbol} at ₹{price:.2f} each.\n"
                f"Margin blocked: ₹{required_margin:.2f} (full position value ₹{total_cost:.2f}).\n"
                f"Remaining balance: ₹{current_user.balance:.2f}\n"
                f"This position must be squared off by end of day."
            )
            flash(message, 'success')
            return render_template('message.html', message=message)
        else:
            # Normal delivery buy
            if total_cost > current_user.balance:
                flash("Insufficient balance to buy!", 'error')
                return render_template('message.html', message="Insufficient balance")
            
            # Update user balance
            current_user.balance = float(current_user.balance - total_cost)
            
            # Add to portfolio
            new_portfolio_item = Portfolio(
                user_id=current_user.id,
                symbol=symbol,
                shares=shares,
                avg_price=price,
                stop_loss=stop_loss_price
            )
            db.session.add(new_portfolio_item)
            
            # Record transaction
            transaction = TransactionHistory(
                user_id=current_user.id,
                transaction_type='BUY',
                trade_type='DELIVERY',
                symbol=symbol,
                shares=shares,
                price=price,
                total_value=total_cost,
                realised_pl=0.0
            )
            db.session.add(transaction)
            db.session.commit()
            
            stop_loss_msg = f"\nStop loss set at ₹{stop_loss_price:.2f}" if stop_loss_price else ""
            message = f"Bought {shares} shares of {symbol} at ₹{price:.2f} each.{stop_loss_msg}\nRemaining balance: ₹{current_user.balance:.2f}"
            flash(message, 'success')
            return render_template('message.html', message=message)
        
    except (IndexError, KeyError) as e:
        # Handle empty history or missing columns
        flash(f"No price data available for {symbol}. Please check the symbol and try again.", 'error')
        return redirect(url_for('buy'))
    except ValueError as e:
        # Handle conversion errors
        flash(f"Invalid input: {str(e)}", 'error')
        return redirect(url_for('buy'))
    except Exception as e:
        print(f"Error in confirm_buy: {e}")
        import traceback
        traceback.print_exc()
        flash(f"Error buying stock: {str(e)}. Please check the symbol and try again.", 'error')
        return redirect(url_for('buy'))

@app.route('/sell')
@login_required
def sell():
    holdings = get_user_holdings_summary(current_user.id)
    intraday_positions = get_user_intraday_positions_summary(current_user.id)
    return render_template('sell.html', holdings=holdings, intraday_positions=intraday_positions)


@app.route('/sell/form')
@login_required
def sell_form():
    symbol = request.args.get('symbol', '').upper()
    holdings = get_user_holdings_summary(current_user.id)
    selected_holding = None
    
    if symbol:
        for holding in holdings:
            if holding['symbol'] == symbol:
                selected_holding = holding
                break
    
    return render_template(
        'sell_form.html',
        symbol_prefill=symbol,
        selected_holding=selected_holding,
        holdings_available=holdings
    )

@app.route('/intraday_sell_form')
@login_required
def intraday_sell_form():
    """Show sell form for a specific intraday position (square-off style)."""
    try:
        position_id = int(request.args.get('position_id', '0'))
    except Exception:
        flash("Invalid intraday position.", 'sell-error')
        return redirect(url_for('sell'))

    position = IntradayPosition.query.filter_by(
        id=position_id, user_id=current_user.id, is_closed=False
    ).first()

    if not position:
        flash("Intraday position not found or already closed.", 'sell-error')
        return redirect(url_for('sell'))

    symbol = position.symbol.upper()
    # Build a pseudo-holding for the template (uses same fields)
    dummy_holding = {
        'symbol': symbol,
        'shares': position.shares,
        'avg_price': float(position.entry_price),
        'current_price': None,
        'current_value': None,
    }
    holdings = [dummy_holding]

    return render_template(
        'sell_form.html',
        symbol_prefill=symbol,
        selected_holding=dummy_holding,
        holdings_available=holdings,
        intraday_position_id=position.id
    )

@app.route('/confirm_sell', methods=['POST'])
@login_required
def confirm_sell():
    symbol = request.form['symbol'].upper()
    shares = int(request.form['shares'])
    print(f"[CONFIRM SELL] Attempting to sell {shares} shares of {symbol}")
    
    # Check if user has enough shares
    user_shares = Portfolio.query.filter_by(user_id=current_user.id, symbol=symbol).all()
    total_shares = sum(item.shares for item in user_shares)
    
    if total_shares < shares:
        flash("You don't have enough shares to sell!", 'sell-error')
        return redirect(url_for('sell'))
    
    try:
        stock = yf.Ticker(symbol + ".NS")
        price = stock.history(period="1d")["Close"].iloc[-1]
        # Convert NumPy types to Python types
        price = float(price)
        total_value = price * shares

        # Estimate average buy price for the shares being sold (using overall average)
        total_cost = sum(pos.avg_price * pos.shares for pos in user_shares)
        overall_avg_price = float(total_cost / total_shares) if total_shares > 0 else 0.0
        estimated_cost_of_sold = overall_avg_price * shares
        realised_pl = total_value - estimated_cost_of_sold
        
        # Update user balance
        current_user.balance = float(current_user.balance + total_value)
        
        # Remove shares from portfolio (FIFO method)
        remaining_shares = shares
        for item in user_shares:
            if remaining_shares <= 0:
                break
            if item.shares <= remaining_shares:
                remaining_shares -= item.shares
                db.session.delete(item)
            else:
                item.shares -= remaining_shares
                remaining_shares = 0
        
        # Record transaction
        transaction = TransactionHistory(
            user_id=current_user.id,
            transaction_type='SELL',
            trade_type='DELIVERY',
            symbol=symbol,
            shares=shares,
            price=price,
            total_value=total_value,
            realised_pl=realised_pl
        )
        db.session.add(transaction)
        db.session.commit()
        
        # Build message with profit/loss info
        pl_sign = "profit" if realised_pl > 0 else ("loss" if realised_pl < 0 else "breakeven")
        message = (
            f"Sold {shares} shares of {symbol} at ₹{price:.2f} each.\n"
            f"Estimated {pl_sign}: ₹{realised_pl:.2f} on this order "
            f"(avg buy ₹{overall_avg_price:.2f}).\n"
            f"New balance: ₹{current_user.balance:.2f}"
        )
        flash(message, 'sell-success')
        return redirect(url_for('sell'))
        
    except Exception as e:
        flash(f"Error: {str(e)}", 'sell-error')
        return redirect(url_for('sell'))

@app.route('/confirm_intraday_sell', methods=['POST'])
@login_required
def confirm_intraday_sell():
    """Square-off a single intraday position at current market price."""
    try:
        position_id = int(request.form['position_id'])
        shares_to_close = int(request.form.get('shares', '0'))
    except Exception:
        flash("Invalid intraday position.", 'sell-error')
        return redirect(url_for('sell'))

    position = IntradayPosition.query.filter_by(id=position_id, user_id=current_user.id, is_closed=False).first()
    if not position:
        flash("Intraday position not found or already closed.", 'sell-error')
        return redirect(url_for('sell'))

    if shares_to_close <= 0 or shares_to_close > position.shares:
        flash("Invalid quantity for intraday square-off.", 'sell-error')
        return redirect(url_for('sell'))

    symbol = position.symbol.upper()
    shares = shares_to_close
    entry_price = float(position.entry_price)
    leverage = float(position.leverage or 5.0)

    try:
        stock = yf.Ticker(symbol + ".NS")
        price = stock.history(period="1d")["Close"].iloc[-1]
        price = float(price)

        # Position economics for the portion being squared off now
        entry_value = entry_price * shares
        exit_value = price * shares
        margin_used = entry_value / leverage if leverage > 0 else entry_value
        realised_pl = exit_value - entry_value

        # Refund margin + P/L for this portion to balance
        current_user.balance = float(current_user.balance + margin_used + realised_pl)

        # Update / close the intraday position
        if shares == position.shares:
            # Full square-off
            position.is_closed = True
            position.exit_price = price
            position.realised_pl = (position.realised_pl or 0.0) + realised_pl
        else:
            # Partial square-off: reduce shares, accumulate realised P/L
            position.shares -= shares
            position.realised_pl = (position.realised_pl or 0.0) + realised_pl

        # Record transaction
        transaction = TransactionHistory(
            user_id=current_user.id,
            transaction_type='SELL',
            trade_type='INTRADAY',
            symbol=symbol,
            shares=shares,
            price=price,
            total_value=exit_value,
            realised_pl=realised_pl
        )
        db.session.add(transaction)
        db.session.commit()

        pl_sign = "profit" if realised_pl > 0 else ("loss" if realised_pl < 0 else "breakeven")
        message = (
            f"Intraday square-off: Sold {shares} shares of {symbol} at ₹{price:.2f} each.\n"
            f"Entry price: ₹{entry_price:.2f}, leverage {leverage:.1f}x.\n"
            f"Realised {pl_sign}: ₹{realised_pl:.2f} on this intraday trade.\n"
            f"Margin released: ₹{margin_used:.2f}.\n"
            f"New balance: ₹{current_user.balance:.2f}"
        )
        flash(message, 'sell-success')
        # After square-off, go back to main sell page
        return redirect(url_for('sell'))

    except Exception as e:
        flash(f"Error squaring off intraday position: {str(e)}", 'sell-error')
        return redirect(url_for('sell'))

@app.route('/update_prices')
def update_prices():
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS', 'TATAMOTORS.NS', 'MRF.NS', 'TCS.NS', "HSCL.NS"]
    stock_prices = get_stock_prices(tickers)
    return jsonify(stock_prices)

@app.route('/stream')
def stream():
    tickers = ['RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS', 'TATAMOTORS.NS', 'MRF.NS', 'TCS.NS', "HSCL.NS"]
    def event_stream():
        while True:
            yield 'data: {}\n\n'.format(jsonify(get_stock_prices(tickers)))
            time.sleep(10)  # Update after every 10 seconds 
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/latest_news')
def latest_news():
    news = scrape_stock_news()
    print("NEWS FETCHED:", news)  # Debug print
    user = current_user if current_user.is_authenticated else None
    return render_template('stock_news.html', news=news, user=user)

@app.route('/ai-predict')
def ai_predict():
    user = current_user if current_user.is_authenticated else None
    return render_template('ai_predict.html', user=user)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        print("=== AI ANALYSIS STARTED ===")
        symbol = request.form['symbol'].upper()
        print(f"Symbol received: {symbol}")
        
        # Initialize the AI predictor
        ai_predictor = StockAI()
        
        # Get prediction using the AI predictor
        prediction_result = ai_predictor.predict_stock(symbol)
        
        print("Analysis completed successfully!")
        print(f"Prediction: {prediction_result}")
        
        return render_template('ai_predict.html', prediction=prediction_result)
        
    except Exception as e:
        print(f"ERROR in analyze: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f'Error analyzing stock: {str(e)}', 'error')
        return render_template('ai_predict.html', prediction=None)

@app.route('/get_stock_price/<symbol>')
def get_stock_price(symbol):
    """Get current stock price for a given symbol"""
    try:
        # Add .NS suffix for Indian stocks if not present
        if not symbol.endswith('.NS'):
            symbol_with_suffix = f"{symbol}.NS"
        else:
            symbol_with_suffix = symbol
        
        stock = yf.Ticker(symbol_with_suffix)
        hist = stock.history(period="1d")
        
        if not hist.empty and 'Close' in hist.columns:
            price = hist['Close'].iloc[-1]
            return jsonify({
                'success': True,
                'price': float(price),
                'symbol': symbol
            })
        else:
            return jsonify({
                'success': False,
                'error': f'No price data available for {symbol}'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error fetching price for {symbol}: {str(e)}'
        })

@app.route('/check_user_shares/<symbol>')
@login_required
def check_user_shares(symbol):
    """Check how many shares a user has for a given symbol"""
    try:
        user_shares = Portfolio.query.filter_by(user_id=current_user.id, symbol=symbol.upper()).all()
        total_shares = sum(item.shares for item in user_shares)
        
        return jsonify({
            'success': True,
            'shares': total_shares,
            'symbol': symbol.upper()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error checking shares for {symbol}: {str(e)}'
        })

@app.route('/invalid_symbol')
def invalid_symbol():
    symbol = request.args.get('symbol', '')
    message = f"No data found for symbol '{symbol}'. Please check the stock symbol you entered. Do not enter the company name. Enter the correct NSE symbol (e.g., BHARTIARTL, NVDA). You can find the correct symbol by searching 'COMPANY NAME NSE SYMBOL' on Google."
    return render_template('message.html', message=message, success=False)

@app.route('/performance')
@login_required
def performance():
    """Display overall performance with realized/unrealized P/L and monthly breakdown."""
    user_id = current_user.id
    
    # Get all transactions
    all_transactions = TransactionHistory.query.filter_by(user_id=user_id).order_by(TransactionHistory.created_at.desc()).all()
    
    # Calculate total realized P/L (from all SELL transactions)
    total_realised_pl = sum(t.realised_pl for t in all_transactions if t.transaction_type == 'SELL')
    
    # Calculate unrealized P/L from current holdings
    holdings = get_user_holdings_summary(user_id)
    total_unrealised_pl = sum(h.get('unrealised_pl', 0) or 0 for h in holdings)
    
    # Calculate unrealized P/L from open intraday positions
    intraday_positions = get_user_intraday_positions_summary(user_id)
    total_intraday_unrealised = sum(pos.get('unrealised_pl', 0) or 0 for pos in intraday_positions)
    total_unrealised_pl += total_intraday_unrealised
    
    # Monthly breakdown
    monthly_data = {}
    for transaction in all_transactions:
        month_key = transaction.created_at.strftime('%Y-%m')
        if month_key not in monthly_data:
            monthly_data[month_key] = {
                'month': transaction.created_at.strftime('%B %Y'),
                'realised_pl': 0.0,
                'buy_count': 0,
                'sell_count': 0,
                'total_volume': 0.0
            }
        
        if transaction.transaction_type == 'SELL':
            monthly_data[month_key]['realised_pl'] += transaction.realised_pl
            monthly_data[month_key]['sell_count'] += 1
        else:
            monthly_data[month_key]['buy_count'] += 1
        
        monthly_data[month_key]['total_volume'] += transaction.total_value
    
    # Sort monthly data by month (newest first)
    monthly_list = sorted(monthly_data.values(), key=lambda x: x['month'], reverse=True)
    
    # Total investment (sum of all BUY transactions)
    total_investment = sum(t.total_value for t in all_transactions if t.transaction_type == 'BUY')
    
    # Total returns (sum of all SELL transactions)
    total_returns = sum(t.total_value for t in all_transactions if t.transaction_type == 'SELL')
    
    # Current portfolio value
    current_portfolio_value = sum(h.get('current_value', 0) or 0 for h in holdings)
    current_intraday_value = sum(pos.get('position_value', 0) or 0 for pos in intraday_positions)
    total_current_value = current_portfolio_value + current_intraday_value
    
    # Overall P/L (realized + unrealized)
    overall_pl = total_realised_pl + total_unrealised_pl
    
    # Calculate return percentage
    return_percentage = (overall_pl / total_investment * 100) if total_investment > 0 else 0.0
    
    return render_template('performance.html', 
                         user=current_user,
                         total_realised_pl=total_realised_pl,
                         total_unrealised_pl=total_unrealised_pl,
                         overall_pl=overall_pl,
                         monthly_data=monthly_list,
                         transactions=all_transactions[:50],  # Last 50 transactions
                         total_investment=total_investment,
                         total_returns=total_returns,
                         total_current_value=total_current_value,
                         return_percentage=return_percentage)

def check_and_execute_stop_loss():
    """Check all open positions with stop loss and execute if price drops below stop loss."""
    executed_stop_losses = []
    
    try:
        # Check delivery positions with stop loss
        delivery_positions = Portfolio.query.filter(
            Portfolio.stop_loss.isnot(None),
            Portfolio.shares > 0
        ).all()
        
        for position in delivery_positions:
            try:
                stock = yf.Ticker(position.symbol + ".NS")
                hist = stock.history(period="1d")
                if not hist.empty and 'Close' in hist.columns:
                    current_price = float(hist['Close'].iloc[-1])
                    
                    # Check if current price is below stop loss
                    if current_price <= position.stop_loss:
                        # Execute stop loss sell
                        user = User.query.get(position.user_id)
                        if user:
                            shares_to_sell = position.shares
                            sell_price = current_price
                            total_value = sell_price * shares_to_sell
                            
                            # Calculate P/L
                            avg_price = position.avg_price
                            realised_pl = (sell_price - avg_price) * shares_to_sell
                            
                            # Update user balance
                            user.balance = float(user.balance + total_value)
                            
                            # Record transaction
                            transaction = TransactionHistory(
                                user_id=user.id,
                                transaction_type='SELL',
                                trade_type='DELIVERY',
                                symbol=position.symbol,
                                shares=shares_to_sell,
                                price=sell_price,
                                total_value=total_value,
                                realised_pl=realised_pl
                            )
                            db.session.add(transaction)
                            
                            # Remove position from portfolio
                            db.session.delete(position)
                            db.session.commit()
                            
                            executed_stop_losses.append({
                                'user_id': user.id,
                                'symbol': position.symbol,
                                'shares': shares_to_sell,
                                'price': sell_price,
                                'stop_loss': position.stop_loss,
                                'type': 'DELIVERY'
                            })
                            
                            print(f"[STOP LOSS] Executed for {user.username}: {shares_to_sell} shares of {position.symbol} at ₹{sell_price:.2f} (stop loss: ₹{position.stop_loss:.2f})")
            except Exception as e:
                print(f"Error checking stop loss for {position.symbol}: {e}")
                continue
        
        # Check intraday positions with stop loss
        intraday_positions = IntradayPosition.query.filter(
            IntradayPosition.stop_loss.isnot(None),
            IntradayPosition.is_closed == False,
            IntradayPosition.shares > 0
        ).all()
        
        for position in intraday_positions:
            try:
                stock = yf.Ticker(position.symbol + ".NS")
                hist = stock.history(period="1d")
                if not hist.empty and 'Close' in hist.columns:
                    current_price = float(hist['Close'].iloc[-1])
                    
                    # Check if current price is below stop loss
                    if current_price <= position.stop_loss:
                        # Execute stop loss square-off
                        user = User.query.get(position.user_id)
                        if user:
                            shares_to_sell = position.shares
                            sell_price = current_price
                            entry_value = position.entry_price * shares_to_sell
                            exit_value = sell_price * shares_to_sell
                            leverage = float(position.leverage or 5.0)
                            margin_used = entry_value / leverage if leverage > 0 else entry_value
                            realised_pl = exit_value - entry_value
                            
                            # Refund margin + P/L
                            user.balance = float(user.balance + margin_used + realised_pl)
                            
                            # Mark position as closed
                            position.is_closed = True
                            position.exit_price = sell_price
                            position.realised_pl = (position.realised_pl or 0.0) + realised_pl
                            
                            # Record transaction
                            transaction = TransactionHistory(
                                user_id=user.id,
                                transaction_type='SELL',
                                trade_type='INTRADAY',
                                symbol=position.symbol,
                                shares=shares_to_sell,
                                price=sell_price,
                                total_value=exit_value,
                                realised_pl=realised_pl
                            )
                            db.session.add(transaction)
                            db.session.commit()
                            
                            executed_stop_losses.append({
                                'user_id': user.id,
                                'symbol': position.symbol,
                                'shares': shares_to_sell,
                                'price': sell_price,
                                'stop_loss': position.stop_loss,
                                'type': 'INTRADAY'
                            })
                            
                            print(f"[STOP LOSS] Executed for {user.username}: {shares_to_sell} shares of {position.symbol} (intraday) at ₹{sell_price:.2f} (stop loss: ₹{position.stop_loss:.2f})")
            except Exception as e:
                print(f"Error checking stop loss for intraday {position.symbol}: {e}")
                continue
                
    except Exception as e:
        print(f"Error in stop loss checker: {e}")
    
    return executed_stop_losses

@app.route('/check_stop_loss', methods=['GET', 'POST'])
@login_required
def check_stop_loss_route():
    """Route to manually trigger stop loss check (can be called periodically)."""
    executed = check_and_execute_stop_loss()
    user_executed = [e for e in executed if e['user_id'] == current_user.id]
    
    return jsonify({
        'success': True,
        'executed_count': len(executed),
        'user_executed': len(user_executed),
        'executed': user_executed
    })

@app.route('/search_stock', methods=['GET'])
def search_stock():
    """
    Search for stock symbol by company name using Gemini API.
    Returns multiple suggestions with symbols and company names.
    """
    query = request.args.get('q', '').strip()
    
    if not query:
        return jsonify({'success': False, 'error': 'No query provided'})
    
    print(f"Search request for: '{query}'")
    
    # First, check if it's already a valid symbol (uppercase, alphanumeric)
    if query.isupper() and len(query) <= 20 and query.replace('_', '').isalnum():
        # Try to verify it's a valid symbol
        try:
            stock = yf.Ticker(query + ".NS")
            hist = stock.history(period="5d")
            if not hist.empty:
                print(f"Query '{query}' is already a valid symbol")
                # Get company info if possible
                try:
                    info = stock.info
                    company_name = info.get('longName', query)
                except:
                    company_name = query
                
                return jsonify({
                    'success': True,
                    'suggestions': [{
                        'symbol': query,
                        'company_name': company_name
                    }],
                    'is_symbol': True
                })
        except Exception as e:
            print(f"Error verifying symbol '{query}': {e}")
    
    # If not a valid symbol, get multiple suggestions using Gemini
    if GEMINI_API_KEY:
        print(f"Getting suggestions for '{query}' using Gemini")
        suggestions = get_multiple_stock_suggestions(query, max_results=2)
        if suggestions:
            print(f"Found {len(suggestions)} suggestions")
            return jsonify({
                'success': True,
                'suggestions': suggestions,
                'is_symbol': False,
                'original_query': query
            })
        else:
            print(f"No suggestions found for '{query}'")
    else:
        print("GEMINI_API_KEY not configured")
    
    return jsonify({
        'success': False,
        'error': f'Could not find stock symbol for "{query}". Please try entering the NSE symbol directly.'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)






















