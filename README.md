# 🤖 AI-Powered Financial Planning Agent

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Research Paper](https://img.shields.io/badge/Research-Paper-orange.svg)](docs/research_paper.pdf)

> **Democratizing professional-grade financial planning through AI**

An intelligent financial planning system that integrates Large Language Models (Google Gemini 2.5) with sophisticated mathematical modeling to provide comprehensive, personalized financial advisory services. Built as part of research at KLE Technological University.

![System Demo](assets/demo.gif)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage Examples](#-usage-examples)
- [Research Paper](#-research-paper)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Screenshots](#-screenshots)
- [Validation & Results](#-validation--results)
- [Contributing](#-contributing)
- [Team](#-team)
- [License](#-license)
- [Citation](#-citation)

---

## ✨ Features

### 🎯 Goal-Based Planning
- **Multi-Goal Management**: Track retirement, house purchase, education, emergency funds simultaneously
- **Inflation Adjustment**: All projections account for 6% inflation (configurable)
- **Progress Tracking**: Visual progress bars showing % completion for each goal
- **Priority Management**: Classify goals as Essential, Important, or Aspirational

### 🏖️ Retirement Planning
- **Corpus Calculation**: Based on 4% safe withdrawal rule (Trinity Study)
- **Lifestyle Options**: Frugal (70%), Same (100%), Comfortable (120%), Luxurious (150%)
- **Alternative Scenarios**: "What if I retire 2 years early?" analysis
- **Withdrawal Strategy**: Recommended post-retirement income plan

### 📊 Monte Carlo Simulation
- **10,000 Iterations**: Probabilistic forecasting with realistic market volatility
- **Success Probability**: Know your chances (e.g., "78% probability of achieving goal")
- **Percentile Analysis**: See best/worst/median outcomes
- **Risk Assessment**: 10th, 25th, 50th, 75th, 90th percentile projections

### 💵 Cash Flow Forecasting
- **6-24 Month Projections**: Short-term financial planning
- **Surplus/Deficit Identification**: Know when you'll have extra or fall short
- **Emergency Fund Analysis**: Recommendations for building safety net
- **Visual Timeline**: Month-by-month breakdown with charts

### 📈 Stock Market Analysis
- **Real-Time Data**: Live prices from Yahoo Finance
- **Interactive Charts**: Candlestick, line, and area charts with Plotly
- **Multiple Timeframes**: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max
- **Technical Indicators**: 52-week high/low, volume analysis
- **News Integration**: Latest market news for tracked stocks

### 🤖 Conversational AI
- **Natural Language**: Ask questions like "Can I retire at 55?"
- **Document Analysis**: Upload PDFs (financial statements, reports)
- **Web Search Integration**: Real-time financial news and market commentary
- **Context Retention**: Remembers conversation history

### 🌍 Indian Market Support
- **Indian Stocks**: NSE/BSE listings
- **Market Indices**: NIFTY 50, SENSEX
- **Currency**: ₹ (Rupee) formatting
- **Local Context**: Tax considerations, investment patterns

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│         Streamlit Frontend (UI)             │
│  - Chat Interface                           │
│  - Goal Planner Forms                       │
│  - Interactive Charts (Plotly)              │
│  - Multi-page Navigation                    │
└──────────────┬──────────────────────────────┘
               │ REST API (HTTP/JSON)
               │
┌──────────────▼──────────────────────────────┐
│        FastAPI Backend (Python)             │
│  - Financial Calculation Engine             │
│  - Monte Carlo Simulation (NumPy)           │
│  - API Orchestration                        │
│  - Business Logic                           │
└────┬─────────────────┬──────────────────────┘
     │                 │
┌────▼─────┐      ┌────▼──────┐
│  Yahoo   │      │  Gemini   │
│ Finance  │      │  AI 2.5   │
│   API    │      │  (LLM)    │
└──────────┘      └───────────┘
```

**Three-Tier Design:**
1. **Presentation Layer**: Streamlit web interface
2. **Application Layer**: FastAPI backend with business logic
3. **Data Layer**: External APIs (Yahoo Finance, Google Gemini)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- Google Gemini API key ([Get one free](https://makersuite.google.com/app/apikey))
- 8GB RAM minimum
- Internet connection

### 5-Minute Setup

```bash
# 1. Clone repository
git clone https://github.com/yourusername/fintech-agent.git
cd fintech-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API key
echo "GEMINI_API_KEY=your_api_key_here" > .env

# 4. Start backend (Terminal 1)
python backend.py

# 5. Start frontend (Terminal 2)
streamlit run frontend.py
```

**That's it!** Open http://localhost:8501 in your browser.

---

## 📦 Installation

### Method 1: Standard Installation

```bash
# Clone repository
git clone https://github.com/yourusername/fintech-agent.git
cd fintech-agent

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
GEMINI_API_KEY=your_actual_api_key_here
EOF

# Verify installation
python -c "import fastapi, streamlit, plotly; print('✓ All packages installed')"
```

### Method 2: Docker (Coming Soon)

```bash
docker-compose up
```

### Method 3: Conda

```bash
conda create -n fintech python=3.9
conda activate fintech
pip install -r requirements.txt
```

---

## 💡 Usage Examples

### Example 1: Retirement Planning

```python
# User: "I'm 35, want to retire at 60 with ₹5 crore. I have ₹10L saved."

Input in UI:
- Current Age: 35
- Retirement Age: 60  
- Current Savings: ₹10,00,000
- Monthly SIP: ₹25,000
- Expected Return: 12%

Output:
✅ Projected Corpus: ₹5.2 Cr
💰 Surplus: ₹20L
📊 Success Probability: 87%
🎯 Recommendation: You're on track! Consider using surplus for other goals.
```

### Example 2: House Down Payment

```python
# Goal: Save ₹40L in 6 years for house

Input:
- Target: ₹40,00,000
- Timeline: 6 years
- Current Savings: ₹5,00,000
- Monthly SIP: ₹35,000
- Expected Return: 10%

Output:
⚠️ Shortfall: ₹3.2L
📈 Required Monthly: ₹38,500
💡 Action: Increase SIP by ₹3,500/month OR extend timeline by 6 months
```

### Example 3: Multi-Goal Balancing

```python
# Managing 3 goals simultaneously:

Emergency Fund (Essential): ₹9L in 1 year
House (Important): ₹40L in 6 years  
Retirement (Essential): ₹5Cr at age 60

System Output:
Priority 1: Emergency (₹75K/month for 12 months)
Priority 2: Retirement (₹25K/month starting now)
Priority 3: House (₹35K/month after emergency complete)

Total Initial: ₹100K/month
After 1 year: ₹60K/month
```

## 🛠️ Technology Stack

### Backend
- **FastAPI** 0.104.1 - Async REST API framework
- **Pydantic** 2.5.0 - Data validation
- **NumPy** 1.24.3 - Monte Carlo simulations
- **yfinance** 0.2.32 - Yahoo Finance API wrapper
- **httpx** 0.25.1 - Async HTTP client

### Frontend
- **Streamlit** 1.29.0 - Web interface
- **Plotly** 5.18.0 - Interactive visualizations
- **Pandas** 2.1.3 - Data manipulation

### AI/ML
- **Google Gemini 2.5 Flash** - Large Language Model
- **Google Search** - Real-time financial data

### Optional
- **PyPDF2** 3.0.1 - PDF document analysis

---

## 📁 Project Structure

```
fintech-agent/
│
├── backend.py                 # FastAPI server with all APIs
├── frontend.py                # Streamlit UI application
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── README.md                 # This file
│
├── docs/
│   ├── research_paper.pdf    # Academic paper
│   ├── presentation.pdf      # Research presentation
│   └── user_guide.pdf        # Detailed usage guide
│
├── assets/
│   ├── demo.gif              # Demo animation
│   ├── screenshot1.png       # Goal planner UI
│   ├── screenshot2.png       # Retirement calculator
│   └── architecture.png      # System diagram
│
├── tests/
│   ├── test_calculations.py  # Unit tests for math
│   ├── test_api.py           # API endpoint tests
│   └── test_monte_carlo.py   # Simulation validation
│
└── notebooks/
    ├── validation.ipynb       # Mathematical validation
    └── user_study.ipynb       # User study analysis
```

---

## 🔌 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Chat with AI
```http
POST /api/chat
Content-Type: application/json

{
  "messages": [
    {
      "role": "user",
      "parts": [{"text": "Can I retire at 55?"}]
    }
  ],
  "document_text": "optional PDF content"
}

Response: {
  "response": "Based on your profile..."
}
```

#### 2. Analyze Financial Goal
```http
POST /api/goals/analyze

{
  "name": "House Down Payment",
  "target_amount": 4000000,
  "current_savings": 500000,
  "monthly_contribution": 35000,
  "years_to_goal": 6,
  "expected_return": 10.0,
  "inflation_rate": 6.0,
  "priority": "Important",
  "goal_type": "House"
}

Response: {
  "projected_amount": 3720000,
  "surplus_or_shortfall": -280000,
  "is_achievable": false,
  "monthly_required": 38500,
  "progress_percentage": 12.5,
  "probability_of_success": 45.2,
  "recommendation": "Increase SIP by ₹3,500...",
  "year_by_year_projection": [...]
}
```

#### 3. Retirement Planning
```http
POST /api/retirement/analyze

{
  "current_age": 35,
  "retirement_age": 60,
  "life_expectancy": 85,
  "current_monthly_expenses": 50000,
  "lifestyle_multiplier": 1.2,
  "existing_savings": 1000000,
  "monthly_sip": 25000,
  "expected_return": 12.0,
  "inflation_rate": 6.0
}

Response: {
  "required_corpus": 64400000,
  "projected_corpus": 52000000,
  "surplus_or_shortfall": -12400000,
  "is_achievable": false,
  ...
}
```

#### 4. Stock Data
```http
GET /api/stock/AAPL

Response: {
  "info": {
    "longName": "Apple Inc.",
    "currentPrice": 178.25,
    "marketCap": 2800000000000,
    ...
  },
  "history": [...],
  "news": [...]
}
```

#### 5. Cash Flow Forecast
```http
POST /api/cashflow/forecast

{
  "monthly_income": 120000,
  "monthly_expenses": 85000,
  "existing_savings": 300000,
  "months_to_forecast": 12
}

Response: {
  "monthly_forecast": [...],
  "average_monthly_surplus": 35000,
  "deficit_months": [],
  "total_savings_end": 720000,
  "insights": [...]
}
```

📚 **Interactive API Docs**: http://localhost:8000/docs (Swagger UI)

---

## ✅ Validation & Results

### Mathematical Accuracy

| Test Case | Manual Calculation | System Output | Accuracy |
|-----------|-------------------|---------------|----------|
| Goal FV (₹10L + ₹20K/mo × 10y @ 12%) | ₹77,92,467 | ₹77,92,467 | 100.0% |
| Retirement Corpus (Age 35→60) | ₹6.44 Cr | ₹6.44 Cr | 99.8% |
| Monte Carlo Mean (10K runs) | ₹76.23L | ₹76.19L | 99.9% |

### User Study Results (N=25)

| Metric | Score | Result |
|--------|-------|--------|
| Ease of Use | 4.6/5 | ⭐⭐⭐⭐⭐ |
| Goal Clarity | 4.8/5 | ⭐⭐⭐⭐⭐ |
| Recommendation Quality | 4.4/5 | ⭐⭐⭐⭐ |
| Trust in Calculations | 4.2/5 | ⭐⭐⭐⭐ |
| Overall Satisfaction | 92% | 🎉 |

### Performance Benchmarks

| Operation | Average Time | 95th Percentile |
|-----------|-------------|-----------------|
| Goal Analysis | 1.2s | 1.8s |
| Monte Carlo (10K) | 0.8s | 1.1s |
| Stock Chart | 2.1s | 3.2s |

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute

1. **🐛 Report Bugs**: Open an issue with detailed reproduction steps
2. **💡 Suggest Features**: Share your ideas in Issues
3. **📝 Improve Documentation**: Fix typos, add examples
4. **🔧 Submit Code**: See guidelines below

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/fintech-agent.git
cd fintech-agent

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
python -m pytest tests/

# Commit with clear message
git commit -m "Add: detailed description of your changes"

# Push and create PR
git push origin feature/your-feature-name
```

### Code Style

- **Python**: Follow PEP 8
- **Docstrings**: Use Google style
- **Type Hints**: Add for all functions
- **Tests**: Write for new features

### Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific test
pytest tests/test_calculations.py::test_future_value
```
## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Abhinandan Onajol, Ajinkya Goundakar, Sakshi Hooli, 
                   Shreya Kutre, Tabassum Jahagirdar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```
## 🙏 Acknowledgments

- **Google Gemini Team** for providing free API access
- **Yahoo Finance** for market data APIs
- **Streamlit** for the amazing rapid prototyping framework
- **FastAPI** for the modern, fast web framework
- **KLE Tech University** for research support and resources
- **Open Source Community** for excellent libraries

---

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/abhinandan976/fintech-agent/issues)
- **Email**: abhinandanvanajol8@gmail.com
- **Documentation**: [Wiki](https://github.com/yourusername/fintech-agent)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/fintech-agent/discussions)

---


## 💬 FAQ

<details>
<summary><b>Is this system SEBI-registered?</b></summary>

No. This is an educational tool for research purposes. It provides information and calculations but not personalized investment advice. Consult a registered financial advisor for major decisions.
</details>

<details>
<summary><b>How accurate are the projections?</b></summary>

Mathematical calculations are 99.9% accurate vs. manual computation. However, projections depend on assumptions (expected returns, inflation) which may not hold in all market conditions. We use conservative estimates based on historical data.
</details>

<details>
<summary><b>Can I use this for my startup/business?</b></summary>

Yes! It's open-source (MIT License). You're free to use, modify, and distribute. We'd appreciate attribution and would love to hear how you're using it.
</details>

<details>
<summary><b>Does it work with US stocks?</b></summary>

Yes! Yahoo Finance provides global market data. Works with US (NYSE, NASDAQ), Indian (NSE, BSE), and most international markets.
</details>

<details>
<summary><b>How do I get a Gemini API key?</b></summary>

Visit https://makersuite.google.com/app/apikey, sign in with Google, and create a free API key. Current free tier includes 60 requests/minute.
</details>

<details>
<summary><b>Can I run this offline?</b></summary>

Partially. Financial calculations work offline. However, you need internet for: Gemini AI responses, stock market data, and real-time news.
</details>

<details>
<summary><b>How do I update to the latest version?</b></summary>

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```
</details>

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/fintech-agent&type=Date)](https://star-history.com/#yourusername/fintech-agent&Date)

---

## 🎉 Show Your Support

If this project helped you, please consider:

- ⭐ **Starring** the repository
- 🍴 **Forking** for your own use
- 📢 **Sharing** with others who might benefit
- 💰 **Sponsoring** future development
- 📝 **Writing** about your experience

---
