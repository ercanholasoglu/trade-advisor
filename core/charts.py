"""
📈 Chart Builder — Plotly financial charts
============================================
Candlestick, indicators, equity curves — all Plotly.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_candlestick_chart(df, ticker: str, sma20=None, sma50=None):
    """Build candlestick chart with SMA overlays and volume."""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.03,
    )
    
    dates = df.index
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=dates,
        open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name=ticker, showlegend=False,
    ), row=1, col=1)
    
    # SMA overlays
    closes = df["Close"].values.astype(float)
    if len(closes) >= 20:
        sma20_vals = [None] * 19 + [float(np.mean(closes[i-19:i+1])) for i in range(19, len(closes))]
        fig.add_trace(go.Scatter(
            x=dates, y=sma20_vals,
            line=dict(color="#FF6B35", width=1.5),
            name="SMA 20",
        ), row=1, col=1)
    
    if len(closes) >= 50:
        sma50_vals = [None] * 49 + [float(np.mean(closes[i-49:i+1])) for i in range(49, len(closes))]
        fig.add_trace(go.Scatter(
            x=dates, y=sma50_vals,
            line=dict(color="#004E98", width=1.5),
            name="SMA 50",
        ), row=1, col=1)
    
    # Volume bars
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=dates, y=df["Volume"],
        marker_color=colors, name="Volume", showlegend=False,
    ), row=2, col=1)
    
    fig.update_layout(
        title=f"📊 {ticker} — Price & Volume",
        height=500,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
        font=dict(color="#e0e0e0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=60, b=20),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    
    return fig

def build_portfolio_equity_chart(snapshots: list[dict]):
    """Build portfolio equity curve from snapshots."""
    if not snapshots or len(snapshots) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Henüz trade yok — önce sinyal çalıştırın", showarrow=False,
                          font=dict(size=14, color="gray"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e", height=300)
        return fig

    equities = [s["equity"] for s in snapshots]
    initial = equities[0]
    final = equities[-1]
    ret = (final - initial) / initial * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=equities, mode='lines',
        line=dict(color="#03DAC6" if final >= initial else "#ef5350", width=2),
        fill='tozeroy', fillcolor='rgba(3,218,198,0.08)' if final >= initial else 'rgba(239,83,80,0.08)',
        name="Portföy",
    ))
    fig.add_hline(y=initial, line_dash="dash", line_color="gray", opacity=0.4,
                  annotation_text=f"Başlangıç: ${initial:,.0f}")
    fig.update_layout(
        title=f"💼 Portföy Değeri — {ret:+.1f}%",
        height=300, template="plotly_dark",
        paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e",
        font=dict(color="#e0e0e0"), yaxis_title="Değer ($)",
        margin=dict(l=50, r=20, t=60, b=30),
    )
    return fig


def build_pnl_distribution_chart(trade_history: list[dict]):
    """Build PnL distribution histogram."""
    if not trade_history:
        fig = go.Figure()
        fig.add_annotation(text="Henüz kapatılmış trade yok", showarrow=False,
                          font=dict(size=14, color="gray"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e", height=300)
        return fig

    pnls = [t["pnl_pct"] for t in trade_history]
    colors = ["#26a69a" if p > 0 else "#ef5350" for p in pnls]

    fig = go.Figure(data=[go.Bar(
        x=[f"{t['ticker']}" for t in trade_history],
        y=pnls,
        marker_color=colors,
        text=[f"{p:+.1f}%" for p in pnls],
        textposition='outside',
    )])
    fig.update_layout(
        title="📊 Trade PnL Dağılımı",
        height=300, template="plotly_dark",
        paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e",
        font=dict(color="#e0e0e0"), yaxis_title="PnL %",
        margin=dict(l=50, r=20, t=60, b=30),
    )
    return fig


def build_drawdown_chart(snapshots: list[dict]):
    """Build drawdown chart from equity snapshots."""
    if not snapshots or len(snapshots) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Yetersiz veri", showarrow=False, font=dict(size=14, color="gray"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e", height=250)
        return fig

    equities = np.array([s["equity"] for s in snapshots])
    peak = np.maximum.accumulate(equities)
    drawdown = (equities - peak) / peak * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=drawdown, mode='lines', fill='tozeroy',
        line=dict(color="#ef5350", width=1.5),
        fillcolor='rgba(239,83,80,0.15)',
        name="Drawdown",
    ))
    fig.update_layout(
        title=f"📉 Drawdown — Max: {float(np.min(drawdown)):.1f}%",
        height=250, template="plotly_dark",
        paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e",
        font=dict(color="#e0e0e0"), yaxis_title="Drawdown %",
        margin=dict(l=50, r=20, t=60, b=30),
    )
    return fig


def build_indicator_panel(df, ticker: str):
    """Build RSI + MACD indicator panel."""
    from core.signal_engine import _rsi_series, _ema
    
    closes = df["Close"].values.astype(float)
    dates = df.index
    
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.08,
        subplot_titles=["RSI (14)", "MACD (12,26,9)"],
    )
    
    # RSI
    rsi_vals = _rsi_series(closes)
    fig.add_trace(go.Scatter(
        x=dates, y=rsi_vals,
        line=dict(color="#BB86FC", width=2),
        name="RSI",
    ), row=1, col=1)
    
    # RSI zones
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=1, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=1, col=1)
    
    # MACD
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    histogram = macd_line - signal_line
    
    colors = ["#26a69a" if h >= 0 else "#ef5350" for h in histogram]
    fig.add_trace(go.Bar(
        x=dates, y=histogram,
        marker_color=colors, name="MACD Hist", showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=macd_line,
        line=dict(color="#03DAC6", width=1.5),
        name="MACD",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=dates, y=signal_line,
        line=dict(color="#FF6B35", width=1.5),
        name="Signal",
    ), row=2, col=1)
    
    fig.update_layout(
        height=400,
        template="plotly_dark",
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
        font=dict(color="#e0e0e0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=20, t=40, b=20),
        showlegend=True,
    )
    fig.update_yaxes(title_text="RSI", row=1, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    
    return fig


def build_equity_curve(backtest_result: dict):
    """Build equity curve chart from backtest results."""
    if "error" in backtest_result or "equity_curve" not in backtest_result:
        fig = go.Figure()
        fig.add_annotation(text="No backtest data", showarrow=False, font=dict(size=20, color="gray"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e")
        return fig
    
    equity = backtest_result["equity_curve"]
    initial = backtest_result.get("initial_capital", 100000)
    
    fig = go.Figure()
    
    # Equity curve
    fig.add_trace(go.Scatter(
        y=equity,
        mode='lines',
        line=dict(color="#03DAC6", width=2),
        name="Strategy",
        fill='tozeroy',
        fillcolor='rgba(3, 218, 198, 0.1)',
    ))
    
    # Initial capital line
    fig.add_hline(y=initial, line_dash="dash", line_color="gray", opacity=0.5,
                  annotation_text=f"Initial: ${initial:,.0f}")
    
    # Color: green if up, red if down
    final = equity[-1] if equity else initial
    color = "#26a69a" if final >= initial else "#ef5350"
    ret = (final - initial) / initial * 100
    
    fig.update_layout(
        title=f"💰 Equity Curve — {ret:+.1f}%",
        height=350,
        template="plotly_dark",
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
        font=dict(color="#e0e0e0"),
        yaxis_title="Portfolio Value ($)",
        xaxis_title="Trade #",
        margin=dict(l=50, r=20, t=60, b=40),
    )
    
    return fig


def build_backtest_signals_chart(backtest_result: dict):
    """Build chart showing buy/sell signals on price data."""
    if "error" in backtest_result or not backtest_result.get("signals"):
        fig = go.Figure()
        fig.add_annotation(text="No signals", showarrow=False, font=dict(size=20, color="gray"))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e")
        return fig
    
    signals = backtest_result["signals"]
    
    buy_dates = [s["date"] for s in signals if s["signal"] == "BUY"]
    buy_prices = [s["entry_price"] for s in signals if s["signal"] == "BUY"]
    buy_correct = [s["correct"] for s in signals if s["signal"] == "BUY"]
    
    sell_dates = [s["date"] for s in signals if s["signal"] == "SELL"]
    sell_prices = [s["entry_price"] for s in signals if s["signal"] == "SELL"]
    sell_correct = [s["correct"] for s in signals if s["signal"] == "SELL"]
    
    all_dates = [s["date"] for s in signals]
    all_prices = [s["entry_price"] for s in signals]
    
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=all_dates, y=all_prices,
        mode='lines', line=dict(color='gray', width=1),
        name='Price', showlegend=False,
    ))
    
    # Buy signals
    fig.add_trace(go.Scatter(
        x=buy_dates, y=buy_prices,
        mode='markers',
        marker=dict(
            symbol='triangle-up', size=12,
            color=["#26a69a" if c else "#ef5350" for c in buy_correct],
            line=dict(width=1, color='white'),
        ),
        name='BUY',
        text=[f"RSI: {s['rsi']}<br>Return: {s['return_pct']:+.1f}%<br>{'✅' if s['correct'] else '❌'}"
              for s in signals if s["signal"] == "BUY"],
        hoverinfo='text+x+y',
    ))
    
    # Sell signals
    fig.add_trace(go.Scatter(
        x=sell_dates, y=sell_prices,
        mode='markers',
        marker=dict(
            symbol='triangle-down', size=12,
            color=["#26a69a" if c else "#ef5350" for c in sell_correct],
            line=dict(width=1, color='white'),
        ),
        name='SELL',
        text=[f"RSI: {s['rsi']}<br>Return: {s['return_pct']:+.1f}%<br>{'✅' if s['correct'] else '❌'}"
              for s in signals if s["signal"] == "SELL"],
        hoverinfo='text+x+y',
    ))
    
    fig.update_layout(
        title=f"📍 Signal Map — {backtest_result['ticker']}",
        height=350,
        template="plotly_dark",
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#1e1e1e",
        font=dict(color="#e0e0e0"),
        yaxis_title="Price",
        margin=dict(l=50, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return fig
