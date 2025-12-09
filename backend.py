import os
import httpx
import yfinance as yf
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import math

load_dotenv()

app = FastAPI(
    title="Fintech Agent API",
    description="Backend for the Fintech Agent with Planning & Forecasting capabilities.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY environment variable not set.")
    
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={GEMINI_API_KEY}"

# --- Pydantic Models ---
class ChatMessage(BaseModel):
    role: str
    parts: List[Dict[str, str]]

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    document_text: Optional[str] = None

class StockData(BaseModel):
    info: Dict[str, Any]
    history: List[Dict[str, Any]]
    news: List[Dict[str, Any]]

class ChartRequest(BaseModel):
    ticker: str
    period: str = "1y"
    interval: str = "1d"

class ChartData(BaseModel):
    ticker: str
    period: str
    interval: str
    data: List[Dict[str, Any]]
    info: Dict[str, Any]

# --- Planning & Forecasting Models ---
class Goal(BaseModel):
    id: Optional[str] = None
    name: str
    target_amount: float
    current_savings: float
    monthly_contribution: float
    years_to_goal: int
    expected_return: float
    priority: str  # "Essential", "Important", "Aspirational"
    goal_type: str  # "Retirement", "House", "Education", "Emergency", "Custom"
    inflation_rate: float = 6.0  # Default 6% for India

class GoalAnalysisResult(BaseModel):
    goal: Goal
    projected_amount: float
    surplus_or_shortfall: float
    is_achievable: bool
    monthly_required: float
    progress_percentage: float
    probability_of_success: float
    recommendation: str
    year_by_year_projection: List[Dict[str, Any]]

class RetirementPlan(BaseModel):
    current_age: int
    retirement_age: int
    life_expectancy: int
    current_monthly_expenses: float
    lifestyle_multiplier: float  # 0.7 = frugal, 1.0 = same, 1.5 = comfortable
    existing_savings: float
    monthly_sip: float
    expected_return: float
    inflation_rate: float = 6.0

class RetirementAnalysisResult(BaseModel):
    required_corpus: float
    projected_corpus: float
    surplus_or_shortfall: float
    years_of_funds: float
    is_achievable: bool
    alternative_retirement_age: Optional[int]
    additional_monthly_sip_needed: float
    recommendation: str
    withdrawal_strategy: Dict[str, Any]

class ScenarioAnalysis(BaseModel):
    base_case: Dict[str, Any]
    optimistic_case: Dict[str, Any]
    pessimistic_case: Dict[str, Any]
    custom_scenarios: List[Dict[str, Any]]

class CashFlowForecast(BaseModel):
    monthly_income: float
    monthly_expenses: float
    existing_savings: float
    months_to_forecast: int = 12

class CashFlowResult(BaseModel):
    monthly_forecast: List[Dict[str, Any]]
    average_monthly_surplus: float
    deficit_months: List[int]
    total_savings_end: float
    insights: List[str]

# --- In-memory storage (replace with database in production) ---
goals_storage: Dict[str, Goal] = {}

# --- Helper Functions ---

def calculate_future_value(pv: float, rate: float, periods: int, pmt: float = 0) -> float:
    """Calculate future value with monthly contributions."""
    if rate == 0:
        return pv + (pmt * periods * 12)
    
    monthly_rate = rate / 12 / 100
    months = periods * 12
    
    # FV of lump sum
    fv_lumpsum = pv * math.pow(1 + monthly_rate, months)
    
    # FV of annuity (monthly contributions)
    if pmt > 0:
        fv_annuity = pmt * ((math.pow(1 + monthly_rate, months) - 1) / monthly_rate)
    else:
        fv_annuity = 0
    
    return fv_lumpsum + fv_annuity

def calculate_required_sip(fv: float, pv: float, rate: float, periods: int) -> float:
    """Calculate required monthly SIP to reach target."""
    if rate == 0:
        months = periods * 12
        return (fv - pv) / months if months > 0 else 0
    
    monthly_rate = rate / 12 / 100
    months = periods * 12
    
    # Subtract FV of existing amount
    fv_existing = pv * math.pow(1 + monthly_rate, months)
    remaining_fv = fv - fv_existing
    
    if remaining_fv <= 0:
        return 0
    
    # Calculate SIP
    sip = (remaining_fv * monthly_rate) / (math.pow(1 + monthly_rate, months) - 1)
    return max(0, sip)

def calculate_inflation_adjusted_amount(amount: float, years: int, inflation_rate: float) -> float:
    """Calculate inflation-adjusted future amount."""
    return amount * math.pow(1 + inflation_rate / 100, years)

def monte_carlo_simulation(pv: float, pmt: float, years: int, avg_return: float, 
                          volatility: float = 15.0, simulations: int = 10000) -> Dict[str, Any]:
    """Run Monte Carlo simulation for goal achievement probability."""
    np.random.seed(42)
    
    final_values = []
    for _ in range(simulations):
        value = pv
        for year in range(years):
            # Generate random return based on normal distribution
            annual_return = np.random.normal(avg_return, volatility)
            monthly_return = annual_return / 12 / 100
            
            # Apply monthly contributions and returns
            for month in range(12):
                value = value * (1 + monthly_return) + pmt
        
        final_values.append(value)
    
    return {
        "median": float(np.median(final_values)),
        "percentile_10": float(np.percentile(final_values, 10)),
        "percentile_25": float(np.percentile(final_values, 25)),
        "percentile_75": float(np.percentile(final_values, 75)),
        "percentile_90": float(np.percentile(final_values, 90)),
        "mean": float(np.mean(final_values)),
        "std_dev": float(np.std(final_values))
    }

# --- API Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Fintech Agent API with Planning & Forecasting is running."}

@app.post("/api/chat", response_model=Dict[str, Any])
async def chat_with_gemini(request_body: ChatRequest = Body(...)):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured on server.")

    system_instruction = """
    You are a 'Fintech Agent,' a world-class financial analyst and advisor with expertise in financial planning.
    Your tone is professional, insightful, and clear.
    - Answer questions based on conversation history and document context
    - Use Google Search for real-time data and financial news
    - Provide concise, actionable insights
    - For financial planning questions, guide users to use the Planning & Forecasting tools
    - Don't give direct buy/sell advice - provide data and analysis for informed decisions
    - When discussing goals, encourage users to create them in the Planning section
    """

    if request_body.document_text:
        doc_context = {
            "role": "user",
            "parts": [{
                "text": f"--- DOCUMENT CONTEXT START ---\n{request_body.document_text}\n--- DOCUMENT CONTEXT END --- \n\nBased on the document I just provided, and our conversation, please answer my next question."
            }]
        }
        messages_with_context = request_body.messages[:-1] + [ChatMessage.model_validate(doc_context)] + [request_body.messages[-1]]
    else:
        messages_with_context = request_body.messages
    
    payload = {
        "contents": [msg.model_dump() for msg in messages_with_context],
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
        }
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(GEMINI_API_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if (
                "candidates" in result
                and len(result["candidates"]) > 0
                and "content" in result["candidates"][0]
                and "parts" in result["candidates"][0]["content"]
                and len(result["candidates"][0]["content"]["parts"]) > 0
                and "text" in result["candidates"][0]["content"]["parts"][0]
            ):
                text_response = result["candidates"][0]["content"]["parts"][0]["text"]
                return {"response": text_response}
            else:
                raise HTTPException(status_code=500, detail="Error processing LLM response.")

    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Error from Gemini API: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/stock/{ticker}", response_model=StockData)
async def get_stock_data(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        
        if not info or info.get('quoteType') == 'NONE':
            raise HTTPException(status_code=404, detail=f"Stock ticker '{ticker}' not found.")

        history_df = tk.history(period="1y")
        history_df = history_df.reset_index()
        history_df['Date'] = history_df['Date'].dt.strftime('%Y-%m-%d')
        history = history_df.to_dict(orient='records')
        
        news = tk.news if isinstance(tk.news, list) else []

        return StockData(info=info, history=history, news=news)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stock data: {str(e)}")

@app.post("/api/chart", response_model=ChartData)
async def get_chart_data(request: ChartRequest):
    try:
        ticker = request.ticker.upper()
        tk = yf.Ticker(ticker)
        info = tk.info
        
        if not info or info.get('quoteType') == 'NONE':
            raise HTTPException(status_code=404, detail=f"Stock ticker '{ticker}' not found.")
        
        try:
            history_df = tk.history(period=request.period, interval=request.interval)
        except Exception:
            history_df = tk.history(period="1y", interval="1d")
        
        if history_df.empty:
            raise HTTPException(status_code=404, detail=f"No historical data available for {ticker}.")
        
        history_df = history_df.reset_index()
        
        if 'Date' in history_df.columns:
            history_df['Date'] = pd.to_datetime(history_df['Date'])
            history_df['timestamp'] = history_df['Date'].astype(int) // 10**6
            history_df['Date'] = history_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        elif 'Datetime' in history_df.columns:
            history_df['Date'] = pd.to_datetime(history_df['Datetime'])
            history_df['timestamp'] = history_df['Date'].astype(int) // 10**6
            history_df['Date'] = history_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        chart_data = history_df.to_dict(orient='records')
        
        essential_info = {
            'symbol': info.get('symbol', ticker),
            'longName': info.get('longName', info.get('shortName', ticker)),
            'currentPrice': info.get('currentPrice', info.get('regularMarketPrice', 'N/A')),
            'currency': info.get('currency', 'USD'),
            'marketCap': info.get('marketCap', 'N/A'),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 'N/A'),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 'N/A'),
        }
        
        return ChartData(
            ticker=ticker,
            period=request.period,
            interval=request.interval,
            data=chart_data,
            info=essential_info
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chart data: {str(e)}")

# --- Planning & Forecasting Endpoints ---

@app.post("/api/goals/analyze", response_model=GoalAnalysisResult)
async def analyze_goal(goal: Goal):
    """Analyze a financial goal and provide detailed recommendations."""
    try:
        # Calculate inflation-adjusted target
        inflation_adjusted_target = calculate_inflation_adjusted_amount(
            goal.target_amount, 
            goal.years_to_goal, 
            goal.inflation_rate
        )
        
        # Calculate projected amount
        projected_amount = calculate_future_value(
            goal.current_savings,
            goal.expected_return,
            goal.years_to_goal,
            goal.monthly_contribution
        )
        
        # Calculate surplus/shortfall
        surplus_or_shortfall = projected_amount - inflation_adjusted_target
        is_achievable = surplus_or_shortfall >= 0
        
        # Calculate required monthly SIP if not achievable
        if not is_achievable:
            monthly_required = calculate_required_sip(
                inflation_adjusted_target,
                goal.current_savings,
                goal.expected_return,
                goal.years_to_goal
            )
        else:
            monthly_required = goal.monthly_contribution
        
        # Calculate progress percentage
        progress_percentage = (goal.current_savings / inflation_adjusted_target) * 100
        
        # Run Monte Carlo simulation
        mc_results = monte_carlo_simulation(
            goal.current_savings,
            goal.monthly_contribution,
            goal.years_to_goal,
            goal.expected_return,
            volatility=15.0
        )
        
        # Calculate probability of success
        simulations_above_target = 0
        for _ in range(10000):
            sim_value = np.random.normal(mc_results['mean'], mc_results['std_dev'])
            if sim_value >= inflation_adjusted_target:
                simulations_above_target += 1
        probability_of_success = (simulations_above_target / 10000) * 100
        
        # Generate recommendation
        if is_achievable:
            if surplus_or_shortfall > inflation_adjusted_target * 0.2:
                recommendation = f"🎉 Excellent! You're on track to exceed your goal by ₹{surplus_or_shortfall/100000:.1f}L. Consider using surplus for other goals or early retirement."
            else:
                recommendation = f"✅ You're on track to achieve this goal! Projected amount: ₹{projected_amount/100000:.1f}L. Stay consistent with your contributions."
        else:
            additional_needed = monthly_required - goal.monthly_contribution
            recommendation = f"⚠️ Current plan falls short by ₹{abs(surplus_or_shortfall)/100000:.1f}L. Increase monthly contribution by ₹{additional_needed:,.0f} to achieve goal, or extend timeline by {math.ceil(abs(surplus_or_shortfall) / (goal.monthly_contribution * 12))} years."
        
        # Year-by-year projection
        year_by_year = []
        current_value = goal.current_savings
        for year in range(1, goal.years_to_goal + 1):
            current_value = calculate_future_value(
                current_value, 
                goal.expected_return, 
                1, 
                goal.monthly_contribution
            )
            year_by_year.append({
                "year": year,
                "age": datetime.now().year + year,
                "value": round(current_value, 2),
                "contribution_total": round(goal.monthly_contribution * 12 * year, 2)
            })
        
        return GoalAnalysisResult(
            goal=goal,
            projected_amount=round(projected_amount, 2),
            surplus_or_shortfall=round(surplus_or_shortfall, 2),
            is_achievable=is_achievable,
            monthly_required=round(monthly_required, 2),
            progress_percentage=round(progress_percentage, 2),
            probability_of_success=round(probability_of_success, 2),
            recommendation=recommendation,
            year_by_year_projection=year_by_year
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing goal: {str(e)}")

@app.post("/api/retirement/analyze", response_model=RetirementAnalysisResult)
async def analyze_retirement(plan: RetirementPlan):
    """Analyze retirement plan and provide detailed recommendations."""
    try:
        years_to_retirement = plan.retirement_age - plan.current_age
        years_in_retirement = plan.life_expectancy - plan.retirement_age
        
        if years_to_retirement <= 0:
            raise HTTPException(status_code=400, detail="Retirement age must be greater than current age")
        
        # Calculate inflation-adjusted monthly expenses at retirement
        future_monthly_expenses = calculate_inflation_adjusted_amount(
            plan.current_monthly_expenses * plan.lifestyle_multiplier,
            years_to_retirement,
            plan.inflation_rate
        )
        
        # Calculate required corpus (using 4% withdrawal rule adjusted for India)
        # Assuming 25x annual expenses for safe withdrawal
        annual_expenses_at_retirement = future_monthly_expenses * 12
        required_corpus = annual_expenses_at_retirement * 25
        
        # Calculate projected corpus
        projected_corpus = calculate_future_value(
            plan.existing_savings,
            plan.expected_return,
            years_to_retirement,
            plan.monthly_sip
        )
        
        # Calculate surplus/shortfall
        surplus_or_shortfall = projected_corpus - required_corpus
        is_achievable = surplus_or_shortfall >= 0
        
        # Calculate years of funds
        if projected_corpus > 0:
            safe_withdrawal_rate = 0.04  # 4% rule
            annual_withdrawal = projected_corpus * safe_withdrawal_rate
            years_of_funds = projected_corpus / annual_expenses_at_retirement
        else:
            years_of_funds = 0
        
        # Calculate alternative retirement age or additional SIP needed
        alternative_retirement_age = None
        additional_monthly_sip_needed = 0
        
        if not is_achievable:
            # Option 1: Calculate additional SIP needed
            additional_monthly_sip_needed = calculate_required_sip(
                required_corpus,
                plan.existing_savings,
                plan.expected_return,
                years_to_retirement
            ) - plan.monthly_sip
            
            # Option 2: Calculate alternative retirement age
            for extra_years in range(1, 11):
                alt_projected = calculate_future_value(
                    plan.existing_savings,
                    plan.expected_return,
                    years_to_retirement + extra_years,
                    plan.monthly_sip
                )
                if alt_projected >= required_corpus:
                    alternative_retirement_age = plan.retirement_age + extra_years
                    break
        
        # Generate recommendation
        if is_achievable:
            recommendation = f"🎉 You're on track for a comfortable retirement! Your projected corpus of ₹{projected_corpus/10000000:.2f} Cr will support {years_of_funds:.1f} years of retirement expenses."
        else:
            if alternative_retirement_age:
                recommendation = f"⚠️ Shortfall of ₹{abs(surplus_or_shortfall)/10000000:.2f} Cr. Options: 1) Increase SIP by ₹{additional_monthly_sip_needed:,.0f}/month, OR 2) Delay retirement to age {alternative_retirement_age}"
            else:
                recommendation = f"⚠️ Significant shortfall. Increase monthly SIP by ₹{additional_monthly_sip_needed:,.0f} to achieve goal."
        
        # Withdrawal strategy
        withdrawal_strategy = {
            "first_year_withdrawal": round(annual_expenses_at_retirement, 2),
            "monthly_withdrawal": round(future_monthly_expenses, 2),
            "safe_withdrawal_rate": "4%",
            "strategy": "Start with 4% withdrawal in year 1, adjust annually for inflation. Maintain 60% debt, 40% equity allocation during retirement."
        }
        
        return RetirementAnalysisResult(
            required_corpus=round(required_corpus, 2),
            projected_corpus=round(projected_corpus, 2),
            surplus_or_shortfall=round(surplus_or_shortfall, 2),
            years_of_funds=round(years_of_funds, 2),
            is_achievable=is_achievable,
            alternative_retirement_age=alternative_retirement_age,
            additional_monthly_sip_needed=round(additional_monthly_sip_needed, 2),
            recommendation=recommendation,
            withdrawal_strategy=withdrawal_strategy
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing retirement: {str(e)}")

@app.post("/api/cashflow/forecast", response_model=CashFlowResult)
async def forecast_cashflow(forecast: CashFlowForecast):
    """Generate cash flow forecast for specified months."""
    try:
        monthly_forecast = []
        running_balance = forecast.existing_savings
        deficit_months = []
        
        for month in range(1, forecast.months_to_forecast + 1):
            monthly_surplus = forecast.monthly_income - forecast.monthly_expenses
            running_balance += monthly_surplus
            
            if monthly_surplus < 0:
                deficit_months.append(month)
            
            monthly_forecast.append({
                "month": month,
                "income": forecast.monthly_income,
                "expenses": forecast.monthly_expenses,
                "surplus_deficit": round(monthly_surplus, 2),
                "balance": round(running_balance, 2)
            })
        
        average_monthly_surplus = sum(m['surplus_deficit'] for m in monthly_forecast) / len(monthly_forecast)
        
        # Generate insights
        insights = []
        if average_monthly_surplus > 0:
            insights.append(f"💰 Positive cash flow! Average monthly surplus: ₹{average_monthly_surplus:,.0f}")
            insights.append(f"📈 Annual savings potential: ₹{average_monthly_surplus * 12:,.0f}")
        else:
            insights.append(f"⚠️ Negative cash flow! Average monthly deficit: ₹{abs(average_monthly_surplus):,.0f}")
        
        if deficit_months:
            insights.append(f"📅 {len(deficit_months)} months show deficit. Build emergency fund.")
        else:
            insights.append("✅ No deficit months projected. Good financial health!")
        
        if running_balance < forecast.monthly_expenses * 3:
            insights.append(f"💡 Emergency fund low. Target: ₹{forecast.monthly_expenses * 6:,.0f} (6 months)")
        
        return CashFlowResult(
            monthly_forecast=monthly_forecast,
            average_monthly_surplus=round(average_monthly_surplus, 2),
            deficit_months=deficit_months,
            total_savings_end=round(running_balance, 2),
            insights=insights
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error forecasting cash flow: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)