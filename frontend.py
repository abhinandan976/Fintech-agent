import streamlit as st
import requests
import re
import pandas as pd
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# --- ROBUST PDF IMPORT ---
pypdf2 = None
PDF_AVAILABLE = False

try:
    import PyPDF2
    pypdf2 = PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pypdf
        pypdf2 = pypdf
        PDF_AVAILABLE = True
    except ImportError:
        pass

def extract_pdf_text(pdf_file):
    if not PDF_AVAILABLE or not pypdf2:
        st.error("❌ PDF library not available. Please install PyPDF2.")
        return None
    
    try:
        if hasattr(pdf_file, 'seek'):
            pdf_file.seek(0)
        
        reader = pypdf2.PdfReader(pdf_file)
        
        if reader.is_encrypted:
            st.error("❌ This PDF is encrypted.")
            return None
        
        text = ""
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except Exception:
                continue
        
        if not text.strip():
            st.warning("⚠️ No text could be extracted from PDF.")
            return None
        
        return text.strip()
    
    except Exception as e:
        st.error(f"❌ Error reading PDF: {str(e)}")
        return None

# --- Configuration ---
st.set_page_config(
    page_title="Fintech Agent - Planning & Forecasting",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://127.0.0.1:8000"

# --- Session State ---
def init_session_state():
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = []
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None
    if "doc_text_cache" not in st.session_state:
        st.session_state.doc_text_cache = None
    if "show_chart" not in st.session_state:
        st.session_state.show_chart = False
    if "chart_ticker" not in st.session_state:
        st.session_state.chart_ticker = None
    if "last_ticker" not in st.session_state:
        st.session_state.last_ticker = None
    if "current_page" not in st.session_state:
        st.session_state.current_page = "chat"
    
    if not st.session_state.chat_sessions and st.session_state.current_chat_id is None:
        create_new_chat()

def create_new_chat():
    chat_count = len(st.session_state.chat_sessions)
    new_chat_id = f"chat_{chat_count + 1}"
    new_chat_title = f"Chat {chat_count + 1}"
    
    new_session = {
        "id": new_chat_id,
        "title": new_chat_title,
        "messages": [],
        "doc_text": None
    }
    
    st.session_state.chat_sessions.insert(0, new_session)
    if len(st.session_state.chat_sessions) > 10:
        st.session_state.chat_sessions.pop()
    
    st.session_state.current_chat_id = new_chat_id
    st.session_state.doc_text_cache = None
    st.session_state.show_chart = False
    st.session_state.chart_ticker = None

def switch_chat(chat_id: str):
    st.session_state.current_chat_id = chat_id
    session = get_current_chat_session()
    if session:
        st.session_state.doc_text_cache = session.get("doc_text")
    else:
        st.session_state.doc_text_cache = None
    st.session_state.show_chart = False

def get_current_chat_session():
    if st.session_state.current_chat_id:
        for session in st.session_state.chat_sessions:
            if session["id"] == st.session_state.current_chat_id:
                return session
    return None

def update_current_chat_session(session_data: dict):
    if st.session_state.current_chat_id:
        for i, session in enumerate(st.session_state.chat_sessions):
            if session["id"] == st.session_state.current_chat_id:
                st.session_state.chat_sessions[i] = session_data
                return

def create_stock_chart(chart_data: dict, chart_type: str = "candlestick"):
    try:
        df = pd.DataFrame(chart_data['data'])
        info = chart_data['info']
        
        if df.empty:
            st.warning("No data available to plot.")
            return None
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3],
            subplot_titles=(f"{info['longName']} ({info['symbol']})", "Volume")
        )
        
        if chart_type == "candlestick":
            fig.add_trace(
                go.Candlestick(
                    x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price',
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                ),
                row=1, col=1
            )
        elif chart_type == "line":
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#2196F3', width=2)
                ),
                row=1, col=1
            )
        elif chart_type == "area":
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='#2196F3', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(33, 150, 243, 0.3)'
                ),
                row=1, col=1
            )
        
        colors = ['#ef5350' if df['Close'].iloc[i] < df['Open'].iloc[i] else '#26a69a' 
                  for i in range(len(df))]
        
        fig.add_trace(
            go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color=colors, showlegend=False),
            row=2, col=1
        )
        
        fig.update_layout(
            title=dict(text=f"{info['longName']} - Current Price: {info['currency']} {info['currentPrice']}", x=0.5, xanchor='center'),
            yaxis_title="Price",
            yaxis2_title="Volume",
            xaxis_rangeslider_visible=False,
            height=700,
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        if info.get('fiftyTwoWeekHigh') != 'N/A':
            fig.add_hline(y=info['fiftyTwoWeekHigh'], line_dash="dash", line_color="green", annotation_text="52W High", row=1, col=1)
        if info.get('fiftyTwoWeekLow') != 'N/A':
            fig.add_hline(y=info['fiftyTwoWeekLow'], line_dash="dash", line_color="red", annotation_text="52W Low", row=1, col=1)
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def format_stock_data(data: dict) -> str:
    try:
        info = data.get("info", {})
        news = data.get("news", [])
        
        name = info.get("longName", info.get("shortName", "N/A"))
        symbol = info.get("symbol", "N/A")
        price = info.get("currentPrice", info.get("previousClose", "N/A"))
        currency = info.get("currency", "")
        
        summary = f"**{name} ({symbol}):**\n\n"
        summary += f"- **Current Price:** {price} {currency}\n"
        summary += f"- **Market Cap:** {info.get('marketCap', 'N/A'):,}\n" if isinstance(info.get('marketCap'), (int, float)) else f"- **Market Cap:** N/A\n"
        summary += f"- **P/E Ratio:** {info.get('trailingPE', 'N/A')}\n"
        summary += f"- **52W Range:** {info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}\n"
        
        if news:
            summary += "\n**Recent News:**\n"
            for item in news[:3]:
                summary += f"- [{item.get('title', 'N/A')}]({item.get('link', '#')})\n"
        
        return summary
    except Exception as e:
        return f"Error: {str(e)}"

def convert_chat_history_for_api(messages: list) -> list:
    api_messages = []
    for msg in messages:
        role = "model" if msg["role"] == "assistant" else "user"
        api_messages.append({"role": role, "parts": [{"text": msg["content"]}]})
    return api_messages

def format_currency(amount: float) -> str:
    """Format currency in Indian style."""
    if amount >= 10000000:  # Crores
        return f"₹{amount/10000000:.2f} Cr"
    elif amount >= 100000:  # Lakhs
        return f"₹{amount/100000:.2f} L"
    else:
        return f"₹{amount:,.0f}"

# --- Planning UI Components ---

def render_goal_planner():
    st.title("🎯 Goal Planner")
    st.markdown("Create and analyze your financial goals with AI-powered insights.")
    
    with st.form("goal_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            goal_name = st.text_input("Goal Name", placeholder="e.g., House Down Payment")
            target_amount = st.number_input("Target Amount (₹)", min_value=10000, value=5000000, step=100000)
            current_savings = st.number_input("Current Savings (₹)", min_value=0, value=0, step=10000)
            monthly_contribution = st.number_input("Monthly Contribution (₹)", min_value=0, value=10000, step=1000)
        
        with col2:
            years_to_goal = st.number_input("Years to Goal", min_value=1, max_value=50, value=10)
            expected_return = st.slider("Expected Annual Return (%)", 5.0, 20.0, 12.0, 0.5)
            inflation_rate = st.slider("Inflation Rate (%)", 3.0, 10.0, 6.0, 0.5)
            goal_type = st.selectbox("Goal Type", ["Retirement", "House", "Education", "Emergency", "Custom"])
            priority = st.selectbox("Priority", ["Essential", "Important", "Aspirational"])
        
        submit_button = st.form_submit_button("🔍 Analyze Goal", use_container_width=True)
    
    if submit_button:
        if not goal_name:
            st.error("Please provide a goal name.")
            return
        
        with st.spinner("Analyzing your goal..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/goals/analyze",
                    json={
                        "name": goal_name,
                        "target_amount": target_amount,
                        "current_savings": current_savings,
                        "monthly_contribution": monthly_contribution,
                        "years_to_goal": years_to_goal,
                        "expected_return": expected_return,
                        "priority": priority,
                        "goal_type": goal_type,
                        "inflation_rate": inflation_rate
                    },
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                
                # Display Results
                st.markdown("---")
                st.subheader("📊 Goal Analysis Results")
                
                # Key Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Projected Amount", format_currency(result['projected_amount']))
                with col2:
                    surplus_color = "normal" if result['is_achievable'] else "inverse"
                    st.metric("Surplus/Shortfall", format_currency(result['surplus_or_shortfall']), 
                             delta_color=surplus_color)
                with col3:
                    st.metric("Progress", f"{result['progress_percentage']:.1f}%")
                with col4:
                    st.metric("Success Probability", f"{result['probability_of_success']:.1f}%")
                
                # Recommendation
                if result['is_achievable']:
                    st.success(result['recommendation'])
                else:
                    st.warning(result['recommendation'])
                    st.info(f"💡 Required Monthly SIP: {format_currency(result['monthly_required'])}")
                
                # Projection Chart
                st.markdown("### 📈 Year-by-Year Projection")
                df = pd.DataFrame(result['year_by_year_projection'])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['year'],
                    y=df['value'],
                    mode='lines+markers',
                    name='Projected Value',
                    line=dict(color='#2196F3', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(33, 150, 243, 0.2)'
                ))
                
                fig.add_hline(y=target_amount, line_dash="dash", line_color="green", 
                             annotation_text="Target (Today's Value)")
                
                fig.update_layout(
                    title="Goal Achievement Projection",
                    xaxis_title="Year",
                    yaxis_title="Amount (₹)",
                    hovermode='x unified',
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error analyzing goal: {str(e)}")

def render_retirement_planner():
    st.title("🏖️ Retirement Planner")
    st.markdown("Plan for a comfortable retirement with detailed projections.")
    
    with st.form("retirement_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            current_age = st.number_input("Current Age", min_value=18, max_value=80, value=30)
            retirement_age = st.number_input("Retirement Age", min_value=40, max_value=80, value=60)
            life_expectancy = st.number_input("Life Expectancy", min_value=60, max_value=100, value=85)
            current_monthly_expenses = st.number_input("Current Monthly Expenses (₹)", min_value=10000, value=50000, step=5000)
        
        with col2:
            lifestyle_multiplier = st.selectbox(
                "Retirement Lifestyle",
                [0.7, 1.0, 1.2, 1.5],
                format_func=lambda x: {0.7: "Frugal (70%)", 1.0: "Same (100%)", 1.2: "Comfortable (120%)", 1.5: "Luxurious (150%)"}[x],
                index=2
            )
            existing_savings = st.number_input("Existing Retirement Savings (₹)", min_value=0, value=500000, step=100000)
            monthly_sip = st.number_input("Monthly SIP (₹)", min_value=0, value=20000, step=5000)
            expected_return = st.slider("Expected Annual Return (%)", 8.0, 15.0, 12.0, 0.5)
            inflation_rate = st.slider("Inflation Rate (%)", 4.0, 8.0, 6.0, 0.5)
        
        analyze_button = st.form_submit_button("🔍 Analyze Retirement Plan", use_container_width=True)
    
    if analyze_button:
        with st.spinner("Analyzing retirement plan..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/retirement/analyze",
                    json={
                        "current_age": current_age,
                        "retirement_age": retirement_age,
                        "life_expectancy": life_expectancy,
                        "current_monthly_expenses": current_monthly_expenses,
                        "lifestyle_multiplier": lifestyle_multiplier,
                        "existing_savings": existing_savings,
                        "monthly_sip": monthly_sip,
                        "expected_return": expected_return,
                        "inflation_rate": inflation_rate
                    },
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                
                st.markdown("---")
                st.subheader("📊 Retirement Analysis")
                
                # Key Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Required Corpus", format_currency(result['required_corpus']))
                with col2:
                    st.metric("Projected Corpus", format_currency(result['projected_corpus']))
                with col3:
                    st.metric("Years of Funds", f"{result['years_of_funds']:.1f} years")
                
                # Recommendation
                if result['is_achievable']:
                    st.success(result['recommendation'])
                else:
                    st.warning(result['recommendation'])
                    if result['additional_monthly_sip_needed'] > 0:
                        st.info(f"💡 Additional Monthly SIP Needed: {format_currency(result['additional_monthly_sip_needed'])}")
                    if result['alternative_retirement_age']:
                        st.info(f"💡 Alternative: Retire at age {result['alternative_retirement_age']}")
                
                # Withdrawal Strategy
                st.markdown("### 💰 Recommended Withdrawal Strategy")
                ws = result['withdrawal_strategy']
                st.info(f"""
                **First Year Withdrawal:** {format_currency(ws['first_year_withdrawal'])} annually  
                **Monthly Income:** {format_currency(ws['monthly_withdrawal'])}  
                **Strategy:** {ws['strategy']}
                """)
                
                # Visual Comparison
                fig = go.Figure(data=[
                    go.Bar(name='Required', x=['Corpus'], y=[result['required_corpus']], marker_color='#ef5350'),
                    go.Bar(name='Projected', x=['Corpus'], y=[result['projected_corpus']], marker_color='#26a69a')
                ])
                
                fig.update_layout(
                    title="Required vs Projected Retirement Corpus",
                    yaxis_title="Amount (₹)",
                    barmode='group',
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error analyzing retirement: {str(e)}")

def render_cashflow_forecast():
    st.title("💵 Cash Flow Forecast")
    st.markdown("Project your monthly cash flow and identify surplus/deficit periods.")
    
    with st.form("cashflow_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_income = st.number_input("Monthly Income (₹)", min_value=10000, value=100000, step=10000)
            monthly_expenses = st.number_input("Monthly Expenses (₹)", min_value=5000, value=60000, step=5000)
        
        with col2:
            existing_savings = st.number_input("Existing Savings (₹)", min_value=0, value=200000, step=10000)
            months_to_forecast = st.slider("Months to Forecast", 6, 24, 12)
        
        forecast_button = st.form_submit_button("📊 Generate Forecast", use_container_width=True)
    
    if forecast_button:
        with st.spinner("Generating cash flow forecast..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/cashflow/forecast",
                    json={
                        "monthly_income": monthly_income,
                        "monthly_expenses": monthly_expenses,
                        "existing_savings": existing_savings,
                        "months_to_forecast": months_to_forecast
                    },
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                
                st.markdown("---")
                st.subheader("📊 Cash Flow Analysis")
                
                # Key Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    surplus_color = "normal" if result['average_monthly_surplus'] >= 0 else "inverse"
                    st.metric("Avg Monthly Surplus", format_currency(result['average_monthly_surplus']), 
                             delta_color=surplus_color)
                with col2:
                    st.metric("Deficit Months", len(result['deficit_months']))
                with col3:
                    st.metric("Final Balance", format_currency(result['total_savings_end']))
                
                # Insights
                st.markdown("### 💡 Key Insights")
                for insight in result['insights']:
                    st.info(insight)
                
                # Chart
                df = pd.DataFrame(result['monthly_forecast'])
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=("Monthly Surplus/Deficit", "Running Balance"),
                    vertical_spacing=0.15,
                    row_heights=[0.5, 0.5]
                )
                
                colors = ['#26a69a' if x >= 0 else '#ef5350' for x in df['surplus_deficit']]
                
                fig.add_trace(
                    go.Bar(x=df['month'], y=df['surplus_deficit'], marker_color=colors, name='Surplus/Deficit'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=df['month'], y=df['balance'], mode='lines+markers', 
                              line=dict(color='#2196F3', width=3), name='Balance'),
                    row=2, col=1
                )
                
                fig.update_xaxes(title_text="Month", row=2, col=1)
                fig.update_yaxes(title_text="Amount (₹)", row=1, col=1)
                fig.update_yaxes(title_text="Balance (₹)", row=2, col=1)
                
                fig.update_layout(height=600, template='plotly_white', showlegend=False)
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")

# --- Main App ---
init_session_state()

# --- Sidebar ---
with st.sidebar:
    st.title("🤖 Fintech Agent")
    
    # Navigation
    st.markdown("### 🧭 Navigation")
    nav_options = {
        "💬 Chat": "chat",
        "🎯 Goal Planner": "goal_planner",
        "🏖️ Retirement Planner": "retirement",
        "💵 Cash Flow Forecast": "cashflow",
        "📊 Stock Charts": "charts"
    }
    
    for label, page in nav_options.items():
        if st.button(label, key=f"nav_{page}", use_container_width=True, 
                    type="primary" if st.session_state.current_page == page else "secondary"):
            st.session_state.current_page = page
            st.rerun()
    
    st.markdown("---")
    
    if st.session_state.current_page == "chat":
        if st.button("➕ New Chat", use_container_width=True):
            create_new_chat()
            st.rerun()
        
        st.markdown("### 💬 Chat History")
        if st.session_state.chat_sessions:
            for session in st.session_state.chat_sessions:
                is_current = session["id"] == st.session_state.current_chat_id
                if st.button(session["title"], key=f"chat_btn_{session['id']}", 
                           use_container_width=True, type="primary" if is_current else "secondary"):
                    switch_chat(session["id"])
                    st.rerun()
    
    elif st.session_state.current_page == "charts":
        st.markdown("### 📊 Stock Chart")
        chart_ticker_input = st.text_input("Ticker", value=st.session_state.last_ticker or "", placeholder="AAPL, MSFT")
        
        if st.button("🚀 Show Chart", use_container_width=True, type="primary"):
            if chart_ticker_input:
                st.session_state.show_chart = True
                st.session_state.chart_ticker = chart_ticker_input.upper()
                st.session_state.last_ticker = chart_ticker_input.upper()
                st.rerun()
        
        st.caption("Quick access:")
        cols = st.columns(3)
        quick_tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA"]
        for idx, ticker in enumerate(quick_tickers):
            with cols[idx % 3]:
                if st.button(f"${ticker}", key=f"quick_{ticker}", use_container_width=True):
                    st.session_state.show_chart = True
                    st.session_state.chart_ticker = ticker
                    st.session_state.last_ticker = ticker
                    st.rerun()
    
    st.markdown("---")
    st.caption("💡 AI-powered financial planning & analysis")

# --- Main Content Area ---
if st.session_state.current_page == "goal_planner":
    render_goal_planner()

elif st.session_state.current_page == "retirement":
    render_retirement_planner()

elif st.session_state.current_page == "cashflow":
    render_cashflow_forecast()

elif st.session_state.current_page == "charts":
    if st.session_state.show_chart and st.session_state.chart_ticker:
        st.subheader(f"📊 {st.session_state.chart_ticker}")
        
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        with col1:
            period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "ytd", "max"], index=5)
        with col2:
            interval_map = {
                "1d": ["1m", "5m", "15m", "30m"],
                "5d": ["5m", "15m", "30m", "1d"],
                "1mo": ["30m", "1d"],
                "3mo": ["1d", "1wk"],
                "6mo": ["1d", "1wk"],
                "1y": ["1d", "1wk"],
                "2y": ["1d", "1wk", "1mo"],
                "5y": ["1d", "1wk", "1mo"],
                "ytd": ["1d", "1wk"],
                "max": ["1d", "1wk", "1mo"]
            }
            interval = st.selectbox("Interval", interval_map.get(period, ["1d"]), index=0)
        with col3:
            chart_type = st.selectbox("Type", ["candlestick", "line", "area"], index=0)
        with col4:
            st.write("")
            st.write("")
            if st.button("✖️ Close", use_container_width=True):
                st.session_state.show_chart = False
                st.rerun()
        
        with st.spinner("Loading chart..."):
            try:
                response = requests.post(f"{API_BASE_URL}/api/chart", json={"ticker": st.session_state.chart_ticker, "period": period, "interval": interval}, timeout=30)
                response.raise_for_status()
                chart_data = response.json()
                fig = create_stock_chart(chart_data, chart_type)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.info("👈 Enter a ticker in the sidebar to view charts")

else:  # Chat page
    current_session = get_current_chat_session()
    
    if current_session:
        st.title(f"💬 {current_session['title']}")
        
        if PDF_AVAILABLE:
            uploaded_file = st.file_uploader("📄 Upload Document", type=["pdf", "txt"], key=f"uploader_{current_session['id']}")
            
            if uploaded_file:
                with st.spinner("Processing..."):
                    try:
                        if uploaded_file.type == "application/pdf":
                            doc_text = extract_pdf_text(BytesIO(uploaded_file.read()))
                        else:
                            uploaded_file.seek(0)
                            doc_text = uploaded_file.read().decode("utf-8")
                        
                        if doc_text:
                            st.session_state.doc_text_cache = doc_text
                            current_session["doc_text"] = doc_text
                            update_current_chat_session(current_session)
                            st.success(f"✅ Processed ({len(doc_text)} chars)")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            elif st.session_state.doc_text_cache:
                st.info(f"📄 Document loaded ({len(st.session_state.doc_text_cache)} chars)")
                if st.button("🗑️ Clear"):
                    st.session_state.doc_text_cache = None
                    current_session["doc_text"] = None
                    update_current_chat_session(current_session)
                    st.rerun()
        
        if not current_session["messages"]:
            st.markdown("""
            ### 👋 Welcome! 
            
            **Try:**
            - "Tell me about $AAPL"
            - "How should I plan for retirement?"
            - Upload financial documents
            - Use Planning tools in sidebar
            """)
        else:
            for message in current_session["messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask about stocks, planning, or upload documents..."):
            current_session["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            stock_ticker_match = re.search(r'\$(\b[A-Z]{1,5}\b)', prompt)
            
            if stock_ticker_match:
                ticker = stock_ticker_match.group(1)
                st.session_state.last_ticker = ticker
                
                with st.chat_message("assistant"):
                    with st.spinner(f"Fetching ${ticker}..."):
                        try:
                            response = requests.get(f"{API_BASE_URL}/api/stock/{ticker}", timeout=10)
                            response.raise_for_status()
                            stock_data = response.json()
                            formatted_response = format_stock_data(stock_data)
                            st.markdown(formatted_response)
                            current_session["messages"].append({"role": "assistant", "content": formatted_response})
                        except Exception as e:
                            error_msg = f"❌ Error: {str(e)}"
                            st.error(error_msg)
                            current_session["messages"].append({"role": "assistant", "content": error_msg})
            else:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            api_messages = convert_chat_history_for_api(current_session["messages"])
                            payload = {"messages": api_messages, "document_text": st.session_state.doc_text_cache}
                            response = requests.post(f"{API_BASE_URL}/api/chat", json=payload, timeout=120)
                            response.raise_for_status()
                            chat_response = response.json().get("response", "Sorry, no response.")
                            st.markdown(chat_response)
                            current_session["messages"].append({"role": "assistant", "content": chat_response})
                        except Exception as e:
                            error_msg = f"❌ Error: {str(e)}"
                            st.error(error_msg)
                            current_session["messages"].append({"role": "assistant", "content": error_msg})
            
            update_current_chat_session(current_session)
            st.rerun()