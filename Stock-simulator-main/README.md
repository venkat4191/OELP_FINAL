# StockSage AI: Advanced Stock Market Simulator

An intelligent stock market simulator that combines real-time market data, predictive analytics, and AI-powered portfolio management to help users master the art of trading in a risk-free environment.

## 🌟 Features

- **Real-time Market Data**: Access live stock prices and market data
- **AI-Powered Predictions**: Get market trend predictions and trading insights
- **Smart Stock Search**: Type company names (e.g., "Bharat Electronics") and automatically get the stock symbol (BEL)
- **Virtual Portfolio Management**: Start with ₹100,000 virtual currency
- **Risk-Free Trading**: Practice trading strategies without financial risk
- **Comprehensive Analytics**: Track performance with detailed metrics
- **Market Regime Detection**: Understand market conditions with AI analysis
- **Portfolio Optimization**: Get AI-driven portfolio recommendations

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/StockSage-AI.git
cd StockSage-AI
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add your Google Gemini API key (optional but recommended for company name search):
   ```
   GEMINI_API_KEY=your-gemini-api-key-here
   SECRET_KEY=your-secret-key-here
   DATABASE_URL=postgresql://localhost/stock_simulator
   ```
   - Get your free Gemini API key from: https://makersuite.google.com/app/apikey

5. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:5000
```

## 💡 Usage

1. **Start Trading**:
   - Begin with ₹100,000 virtual currency
   - Buy and sell stocks in real-time
   - Track your portfolio performance

2. **AI Features**:
   - View market predictions
   - Get portfolio recommendations
   - Analyze market regimes

3. **Portfolio Management**:
   - Monitor your holdings
   - Track performance metrics
   - Get risk analysis

## 🛠️ Technical Stack

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **Data Analysis**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Market Data**: Yahoo Finance API
- **AI Search**: Google Gemini API (for company name to symbol conversion)
- **Visualization**: Matplotlib, Seaborn

## 📊 Project Structure

```
StockSage-AI/
├── app.py              # Main application file
├── ml_models.py        # Machine learning models
├── static/            # Static files (CSS, JS)
├── templates/         # HTML templates
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Yahoo Finance for market data
- All contributors who have helped shape this project

## 📞 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)
Project Link: [https://github.com/yourusername/StockSage-AI](https://github.com/yourusername/StockSage-AI)
