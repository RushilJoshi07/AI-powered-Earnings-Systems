import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.rag import analyze_company, answer_question

st.set_page_config(
    page_title="EarningsEdge",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp {
        background-color: #060810;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 38px;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #00d4ff 60%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 12px;
        color: #475569;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-bottom: 28px;
    }
    .company-header {
        background: linear-gradient(135deg, #0d1117, #0f1923);
        border: 1px solid #1e2d3d;
        border-left: 4px solid #00d4ff;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 24px;
    }
    .company-ticker {
        font-size: 32px;
        font-weight: 800;
        color: #ffffff;
    }
    .company-quarter {
        font-size: 13px;
        color: #64748b;
        margin-top: 4px;
    }
    .data-badge {
        background: rgba(0,212,255,0.1);
        border: 1px solid rgba(0,212,255,0.2);
        color: #00d4ff;
        font-size: 11px;
        padding: 4px 12px;
        border-radius: 20px;
        letter-spacing: 1px;
    }
    .section-title {
        font-size: 10px;
        letter-spacing: 4px;
        color: #00d4ff;
        text-transform: uppercase;
        margin-bottom: 14px;
        margin-top: 28px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e2d3d;
    }
    .metric-card {
        background: #0d1117;
        border: 1px solid #1e2d3d;
        border-radius: 10px;
        padding: 20px 16px;
        text-align: center;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-label {
        font-size: 10px;
        color: #475569;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        line-height: 1;
    }
    .metric-sub {
        font-size: 11px;
        color: #475569;
        margin-top: 6px;
    }
    .reasoning-box {
        background: #0d1117;
        border: 1px solid #1e2d3d;
        border-left: 3px solid #00d4ff;
        border-radius: 8px;
        padding: 24px;
        font-size: 14px;
        line-height: 1.8;
        color: #cbd5e1;
    }
    .disclaimer {
        font-size: 11px;
        color: #334155;
        margin-top: 14px;
        padding-top: 14px;
        border-top: 1px solid #1e2d3d;
        font-style: italic;
    }
    .theme-tag {
        display: inline-block;
        background: rgba(0,212,255,0.08);
        border: 1px solid rgba(0,212,255,0.2);
        color: #7dd3fc;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 12px;
        margin: 3px;
        font-weight: 500;
    }
    .chat-container {
        background: #0d1117;
        border: 1px solid #1e2d3d;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .chat-message-user {
        background: #1e2d3d;
        border-radius: 8px 8px 2px 8px;
        padding: 10px 16px;
        margin: 8px 0;
        font-size: 13px;
        color: #e2e8f0;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-message-bot {
        background: #131920;
        border: 1px solid #1e2d3d;
        border-left: 2px solid #00d4ff;
        border-radius: 2px 8px 8px 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 13px;
        color: #cbd5e1;
        line-height: 1.7;
        max-width: 90%;
    }
    .stTextInput input {
        background-color: #0d1117;
        border: 1px solid #1e2d3d;
        color: #e2e8f0;
        border-radius: 6px;
        font-size: 14px;
    }
    .stButton button {
        background: linear-gradient(135deg, #00d4ff, #0088cc);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        letter-spacing: 1px;
        height: 42px;
    }
    .positive-badge {
        background: rgba(16,185,129,0.15);
        color: #10b981;
        border: 1px solid rgba(16,185,129,0.3);
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
    }
    .negative-badge {
        background: rgba(244,63,94,0.15);
        color: #f43f5e;
        border: 1px solid rgba(244,63,94,0.3);
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
    }
    .neutral-badge {
        background: rgba(245,158,11,0.15);
        color: #f59e0b;
        border: 1px solid rgba(245,158,11,0.3);
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=86400)
def get_company_name(ticker):
    # fetches real company name from yfinance
    # cached for 24 hours so we only call yfinance once per ticker per day
    try:
        info = yf.Ticker(ticker).info
        return info.get("longName") or info.get("shortName") or ticker
    except Exception:
        return ticker


def get_tone_from_polarity(polarity):
    if polarity > 0.03:
        return "Positive"
    elif polarity < -0.03:
        return "Negative"
    else:
        return "Neutral"


def get_available_tickers():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "cleaned", "training_data_sentiment.csv"))
    tickers = sorted(df["symbol"].unique().tolist())
    options = []
    for t in tickers:
        name = get_company_name(t)
        if name != t:
            options.append(f"{t} — {name}")
        else:
            options.append(t)
    return tickers, options


def run_analysis(ticker):
    return analyze_company(ticker)


def get_stock_chart_data(symbol, earnings_date_str):
    try:
        date = datetime.strptime(earnings_date_str, "%Y-%m-%d")
        start = date - timedelta(days=10)
        end = date + timedelta(days=10)

        stock_df = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False
        )
        spy_df = yf.download(
            "SPY",
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            progress=False
        )

        if stock_df.empty or spy_df.empty:
            return None, None

        return stock_df, spy_df

    except Exception:
        return None, None


def build_stock_chart(symbol, earnings_date_str, stock_df, spy_df):
    stock_close = stock_df["Close"].dropna()
    spy_close = spy_df["Close"].dropna()

    if hasattr(stock_close.iloc[0], 'iloc'):
        stock_close = stock_close.iloc[:, 0]
    if hasattr(spy_close.iloc[0], 'iloc'):
        spy_close = spy_close.iloc[:, 0]

    stock_normalized = (stock_close / float(stock_close.iloc[0])) * 100
    spy_normalized = (spy_close / float(spy_close.iloc[0])) * 100

    stock_final = float(stock_normalized.iloc[-1])
    excess = stock_final - float(spy_normalized.iloc[-1])
    stock_color = "#10b981" if stock_final >= 100 else "#f43f5e"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=stock_normalized.index,
        y=stock_normalized.values,
        name=symbol,
        line=dict(color=stock_color, width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba({'16,185,129' if stock_final >= 100 else '244,63,94'},0.05)",
        hovertemplate=f"{symbol}: %{{y:.1f}}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=spy_normalized.index,
        y=spy_normalized.values,
        name="S&P 500",
        line=dict(color="#475569", width=1.5, dash="dot"),
        hovertemplate="S&P 500: %{y:.1f}<extra></extra>"
    ))

    earnings_dt = datetime.strptime(earnings_date_str, "%Y-%m-%d")
    fig.add_vline(
        x=earnings_dt.timestamp() * 1000,
        line_dash="dash",
        line_color="#f59e0b",
        line_width=1.5,
        annotation_text="Earnings Call",
        annotation_font_color="#f59e0b",
        annotation_font_size=11
    )

    fig.add_hline(y=100, line_color="#1e2d3d", line_width=1)

    fig.update_layout(
        paper_bgcolor="#060810",
        plot_bgcolor="#060810",
        font_color="#e2e8f0",
        legend=dict(bgcolor="#0d1117", bordercolor="#1e2d3d", borderwidth=1),
        xaxis=dict(gridcolor="#0d1117", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#0d1117", showgrid=True, title="Normalized Price (Base 100)", zeroline=False),
        margin=dict(l=0, r=0, t=10, b=0),
        height=320,
        hovermode="x unified"
    )

    return fig, excess


def build_trend_chart(trend_data):
    quarters = [t["quarter"] for t in trend_data]
    polarities = [t["polarity"] for t in trend_data]
    pos_scores = [t["positive_score"] * 100 for t in trend_data]
    neg_scores = [t["negative_score"] * 100 for t in trend_data]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12,
        subplot_titles=("Sentiment Polarity per Quarter", "Positive vs Negative Language (%)")
    )

    bar_colors = []
    for p in polarities:
        if p > 0.03:
            bar_colors.append("#10b981")
        elif p < -0.03:
            bar_colors.append("#f43f5e")
        else:
            bar_colors.append("#f59e0b")

    fig.add_trace(go.Bar(
        x=quarters,
        y=polarities,
        marker_color=bar_colors,
        name="Polarity",
        hovertemplate="Quarter: %{x}<br>Polarity: %{y:.4f}<extra></extra>"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=quarters,
        y=pos_scores,
        name="Positive %",
        line=dict(color="#10b981", width=2),
        hovertemplate="Positive language: %{y:.1f}%<extra></extra>"
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=quarters,
        y=neg_scores,
        name="Negative %",
        line=dict(color="#f43f5e", width=2),
        hovertemplate="Negative language: %{y:.1f}%<extra></extra>"
    ), row=2, col=1)

    fig.update_layout(
        paper_bgcolor="#060810",
        plot_bgcolor="#060810",
        font_color="#e2e8f0",
        legend=dict(bgcolor="#0d1117", bordercolor="#1e2d3d", borderwidth=1),
        margin=dict(l=0, r=0, t=30, b=0),
        height=400
    )

    fig.update_xaxes(gridcolor="#0d1117")
    fig.update_yaxes(gridcolor="#0d1117")

    for annotation in fig.layout.annotations:
        annotation.font.color = "#64748b"
        annotation.font.size = 11

    return fig


def render_sentiment_cards(company_data):
    polarity = company_data["polarity"]
    tone = get_tone_from_polarity(polarity)

    tone_color = {
        "Positive": "#10b981",
        "Negative": "#f43f5e",
        "Neutral": "#f59e0b"
    }.get(tone, "#f59e0b")

    tone_emoji = {
        "Positive": "📈",
        "Negative": "📉",
        "Neutral": "➡️"
    }.get(tone, "➡️")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 2px solid {tone_color}">
            <div class="metric-label">Overall Tone</div>
            <div class="metric-value" style="color: {tone_color}">{tone_emoji} {tone}</div>
            <div class="metric-sub">Based on full transcript</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 2px solid #10b981">
            <div class="metric-label">Positive Language</div>
            <div class="metric-value" style="color: #10b981">{company_data['positive_score']:.1%}</div>
            <div class="metric-sub">of words were optimistic</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 2px solid #f43f5e">
            <div class="metric-label">Negative Language</div>
            <div class="metric-value" style="color: #f43f5e">{company_data['negative_score']:.1%}</div>
            <div class="metric-sub">of words were cautious</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        polarity_color = "#10b981" if polarity > 0 else "#f43f5e"
        polarity_label = (
            "More positive than negative" if polarity > 0.03
            else "More negative than positive" if polarity < -0.03
            else "Balanced — roughly equal"
        )
        st.markdown(f"""
        <div class="metric-card" style="border-top: 2px solid {polarity_color}">
            <div class="metric-label">Tone Balance Score</div>
            <div class="metric-value" style="color: {polarity_color}">{polarity:+.4f}</div>
            <div class="metric-sub">{polarity_label}</div>
        </div>
        """, unsafe_allow_html=True)


def render_trend_insights(trend_data):
    if not trend_data or len(trend_data) < 2:
        return

    latest = trend_data[-1]
    previous = trend_data[-2]
    polarity_change = latest["polarity"] - previous["polarity"]
    direction = "Improved" if polarity_change > 0 else "Declined"
    change_color = "#10b981" if polarity_change > 0 else "#f43f5e"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Tone Trend</div>
            <div class="metric-value" style="color: {change_color}; font-size: 18px">
                {'↑' if polarity_change > 0 else '↓'} {direction}
            </div>
            <div class="metric-sub">vs previous quarter</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        quarters_positive = sum(1 for t in trend_data if t["polarity"] > 0.03)
        total = len(trend_data)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Positive Quarters</div>
            <div class="metric-value" style="color: #10b981">{quarters_positive}/{total}</div>
            <div class="metric-sub">historically positive tone</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_polarity = sum(t["polarity"] for t in trend_data) / len(trend_data)
        avg_color = "#10b981" if avg_polarity > 0 else "#f43f5e"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Tone Score</div>
            <div class="metric-value" style="color: {avg_color}">{avg_polarity:+.4f}</div>
            <div class="metric-sub">across all quarters</div>
        </div>
        """, unsafe_allow_html=True)


def main():
    st.markdown('<div class="main-header">EarningsEdge</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Earnings Intelligence Platform</div>', unsafe_allow_html=True)

    tickers, options = get_available_tickers()

    col_search, col_button = st.columns([4, 1])

    with col_search:
        selected_option = st.selectbox(
            "Search",
            options=["Select a company..."] + options,
            index=0,
            label_visibility="collapsed"
        )

    with col_button:
        analyze_clicked = st.button("Analyze →", use_container_width=True)

    selected_ticker = None
    if selected_option and selected_option != "Select a company...":
        selected_ticker = selected_option.split(" — ")[0].strip()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_ticker" not in st.session_state:
        st.session_state.current_ticker = None
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "input_value" not in st.session_state:
        st.session_state["input_value"] = ""

    if selected_ticker != st.session_state.current_ticker:
        st.session_state.chat_history = []
        st.session_state.current_ticker = selected_ticker
        st.session_state.analysis_result = None

    if analyze_clicked and selected_ticker:
        with st.spinner(f"Analyzing {selected_ticker} earnings call..."):
            result = run_analysis(selected_ticker)
            st.session_state.analysis_result = result

    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        company_data = result["company_data"]
        polarity = company_data["polarity"]
        tone = get_tone_from_polarity(polarity)
        company_name = get_company_name(company_data["symbol"])

        tone_badge_class = {
            "Positive": "positive-badge",
            "Negative": "negative-badge",
            "Neutral": "neutral-badge"
        }.get(tone, "neutral-badge")

        st.markdown(f"""
        <div class="company-header">
            <div>
                <div class="company-ticker">{company_data['symbol']}
                    <span style="font-size: 18px; color: #64748b; font-weight: 400"> — {company_name}</span>
                </div>
                <div class="company-quarter">
                    {company_data['quarter']} Earnings Call &nbsp;·&nbsp; {company_data['date']}
                </div>
            </div>
            <div style="margin-top: 8px">
                <span class="{tone_badge_class}">{tone} Tone</span>
                &nbsp;
                <span class="data-badge">2019–2022 Data</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">📊 Sentiment Analysis</div>', unsafe_allow_html=True)
        render_sentiment_cards(company_data)

        st.markdown('<div class="section-title">📈 Stock Movement vs S&P 500</div>', unsafe_allow_html=True)

        with st.spinner("Loading price data..."):
            stock_df, spy_df = get_stock_chart_data(
                company_data["symbol"],
                company_data["date"]
            )

        if stock_df is not None:
            fig, excess_return = build_stock_chart(
                company_data["symbol"],
                company_data["date"],
                stock_df,
                spy_df
            )

            col_chart, col_stats = st.columns([3, 1])

            with col_chart:
                st.plotly_chart(fig, use_container_width=True)

            with col_stats:
                excess_color = "#10b981" if excess_return >= 0 else "#f43f5e"
                excess_label = "Outperformed S&P 500" if excess_return >= 0 else "Underperformed S&P 500"
                excess_arrow = "↑" if excess_return >= 0 else "↓"
                st.markdown(f"""
                <div class="metric-card" style="margin-top: 10px">
                    <div class="metric-label">20-Day Excess Return</div>
                    <div class="metric-value" style="color: {excess_color}">{excess_arrow} {abs(excess_return):.1f}%</div>
                    <div class="metric-sub">{excess_label} over 20 days</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="metric-card" style="margin-top: 10px">
                    <div class="metric-label">Earnings Date</div>
                    <div class="metric-value" style="color: #e2e8f0; font-size: 14px">{company_data['date']}</div>
                    <div class="metric-sub">{company_data['quarter']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="color: #475569; font-size: 13px; padding: 20px 0">Price data not available for this period.</div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="section-title">🤖 AI Earnings Summary</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="reasoning-box">
            {result['reasoning']}
            <div class="disclaimer">
                ⚠ This summary is based solely on the earnings call transcript.
                External market factors not mentioned in the call may have influenced stock movement.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">🏷️ Key Topics Discussed</div>', unsafe_allow_html=True)
        themes_html = "".join([f'<span class="theme-tag">{t}</span>' for t in result["themes"]])
        st.markdown(f'<div style="padding: 8px 0">{themes_html}</div>', unsafe_allow_html=True)

        if result["trend"]:
            st.markdown('<div class="section-title">📅 Historical Sentiment Trend</div>', unsafe_allow_html=True)
            render_trend_insights(result["trend"])
            st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
            trend_fig = build_trend_chart(result["trend"])
            st.plotly_chart(trend_fig, use_container_width=True)

            st.markdown('<div style="margin-top: 16px"></div>', unsafe_allow_html=True)
            recent_quarters = result["trend"][-4:]
            trend_cols = st.columns(len(recent_quarters))
            for i, quarter_data in enumerate(recent_quarters):
                with trend_cols[i]:
                    q_tone = get_tone_from_polarity(quarter_data["polarity"])
                    q_color = {
                        "Positive": "#10b981",
                        "Negative": "#f43f5e",
                        "Neutral": "#f59e0b"
                    }.get(q_tone, "#f59e0b")
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{quarter_data['quarter']}</div>
                        <div class="metric-value" style="color: {q_color}; font-size: 16px">{q_tone}</div>
                        <div class="metric-sub">Polarity: {quarter_data['polarity']:+.4f}</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="section-title">📅 Historical Sentiment Trend</div>', unsafe_allow_html=True)
            st.markdown(
                '<div style="color: #475569; font-size: 13px; padding: 12px 0">Less than 3 quarters available for trend analysis.</div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="section-title">💬 Ask the Transcript</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="color: #475569; font-size: 13px; margin-bottom: 12px;">Ask any question about what was said on this earnings call. Get answers directly from the transcript.</div>',
            unsafe_allow_html=True
        )

        if st.session_state.chat_history:
            chat_html = ""
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    chat_html += f'<div class="chat-message-user">🧑 {message["content"]}</div>'
                else:
                    chat_html += f'<div class="chat-message-bot">🤖 {message["content"]}</div>'
            st.markdown(f'<div class="chat-container">{chat_html}</div>', unsafe_allow_html=True)

        col_input, col_ask = st.columns([5, 1])

        with col_input:
            user_question = st.text_input(
                "Question",
                placeholder="e.g. What did management say about revenue growth?",
                label_visibility="collapsed",
                key=f"chat_input_{len(st.session_state.chat_history)}"
            )

        with col_ask:
            ask_clicked = st.button("Ask →", key="ask_button", use_container_width=True)

        if ask_clicked:
            if user_question and len(user_question.strip()) >= 5:
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_question
                })

                with st.spinner("Searching transcript..."):
                    answer = answer_question(
                        user_question,
                        result["index"],
                        result["chunks"]
                    )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

                st.rerun()
            else:
                st.warning("Please enter a question with at least 5 characters.")

    elif analyze_clicked and not selected_ticker:
        st.warning("Please select a company first.")

    else:
        st.markdown("""
        <div style="text-align: center; padding: 80px 0; color: #334155;">
            <div style="font-size: 56px; margin-bottom: 20px">📊</div>
            <div style="font-size: 18px; color: #475569; margin-bottom: 8px; font-weight: 500">
                Select a company to analyze its earnings call
            </div>
            <div style="font-size: 13px; color: #334155; margin-bottom: 32px">
                440 S&P 500 companies · 2019–2022 earnings data · Powered by FinBERT + Groq
            </div>
            <div style="display: flex; justify-content: center; gap: 24px; flex-wrap: wrap">
                <div style="color: #1e3a5f; font-size: 12px; letter-spacing: 1px">📈 SENTIMENT ANALYSIS</div>
                <div style="color: #1e3a5f; font-size: 12px; letter-spacing: 1px">🤖 AI REASONING</div>
                <div style="color: #1e3a5f; font-size: 12px; letter-spacing: 1px">💬 RAG CHATBOT</div>
                <div style="color: #1e3a5f; font-size: 12px; letter-spacing: 1px">📅 TREND ANALYSIS</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

# streamlit run /Users/rushil/PycharmProjects/PythonProject/earningsedge/modules/dashboard.py