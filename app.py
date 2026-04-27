"""
🤖 Trade Bot Advisor - Gradio UI
=================================
Multi-agent agentic trade advisor with interactive Gradio interface.
"""

import os
import gradio as gr
from agent import create_trade_advisor


# ============================================================
# Global agent instance (lazy init)
# ============================================================
_advisor = None


def get_advisor():
    """Lazy-init: agent'ı ilk kullanımda oluşturur."""
    global _advisor
    if _advisor is None:
        hf_token = os.environ.get("HF_TOKEN", "")
        model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-72B-Instruct")
        _advisor = create_trade_advisor(hf_token=hf_token, model_id=model_id)
    return _advisor


# ============================================================
# Quick Analysis (Structured)
# ============================================================
def quick_analysis(ticker: str, portfolio_value: float, risk_tolerance: str):
    """Structured quick analysis - ticker + portfolio bilgisi alır."""
    if not ticker or not ticker.strip():
        return "❌ Lütfen bir ticker sembolü girin (örn: AAPL, NVDA, BTC-USD)"

    ticker = ticker.strip().upper()
    query = (
        f"Analyze {ticker} for a potential trade entry. "
        f"My portfolio value is ${portfolio_value:,.0f}. "
        f"My risk tolerance is {risk_tolerance.lower()}. "
        f"Provide a comprehensive trade recommendation."
    )

    try:
        advisor = get_advisor()
        result = advisor.run(query)
        return str(result)
    except Exception as e:
        return f"❌ Analiz sırasında hata oluştu:\n\n{str(e)}"


# ============================================================
# Free-form Chat
# ============================================================
def chat_analysis(message: str, history: list):
    """Free-form sohbet - kullanıcı istediğini sorabilir."""
    if not message or not message.strip():
        return "Lütfen bir soru sorun. Örnek: 'AAPL hissesini analiz et, portföyüm $100K ve orta risk toleransım var.'"

    try:
        advisor = get_advisor()
        result = advisor.run(message)
        return str(result)
    except Exception as e:
        return f"❌ Hata: {str(e)}"


# ============================================================
# Gradio Interface
# ============================================================
DESCRIPTION = """
# 🤖 Agentic Trade Bot Advisor

**TradingAgents** paper (arxiv: 2412.20138) mimarisinden ilham alan multi-agent trade advisor.

### 🏗️ Mimari
| Agent | Görev |
|---|---|
| 📈 **Technical Analyst** | RSI, MACD, Bollinger Bands, SMA, ATR, Stochastic |
| 📰 **Sentiment Analyst** | Haber analizi + piyasa duyarlılığı |
| 📊 **Fundamental Analyst** | P/E, PEG, marjlar, büyüme, bilanço |
| ⚠️ **Risk Manager** | VaR, position sizing, stop-loss, piyasa rejimi |
| 🎯 **Fund Manager** | Tüm raporları sentezler, final trade kararı verir |

### ⚡ Nasıl Çalışır
1. Bir ticker girin (örn: AAPL, NVDA, BTC-USD)
2. 5 agent sırayla çalışır: Teknik → Sentiment → Fundamental → Risk → Final Karar
3. Yapılandırılmış trade raporu alırsınız: Sinyal, güven skoru, giriş/çıkış seviyeleri

> ⚠️ **Disclaimer**: Bu yapay zeka destekli bir analiz aracıdır. Yatırım tavsiyesi değildir.
"""

# --- Hızlı Analiz Tab ---
with gr.Blocks(
    title="🤖 Trade Bot Advisor",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
    ),
) as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        # ---- Tab 1: Hızlı Analiz ----
        with gr.Tab("⚡ Hızlı Analiz"):
            with gr.Row():
                with gr.Column(scale=1):
                    ticker_input = gr.Textbox(
                        label="📌 Ticker Sembolü",
                        placeholder="AAPL, NVDA, TSLA, BTC-USD...",
                        value="NVDA",
                    )
                    portfolio_input = gr.Number(
                        label="💰 Portföy Değeri ($)",
                        value=100000,
                        minimum=1000,
                    )
                    risk_input = gr.Radio(
                        label="⚖️ Risk Toleransı",
                        choices=["Conservative", "Moderate", "Aggressive"],
                        value="Moderate",
                    )
                    analyze_btn = gr.Button(
                        "🔍 Analiz Başlat",
                        variant="primary",
                        size="lg",
                    )

                    gr.Markdown("""
                    ### 📝 Örnek Ticker'lar
                    - **Hisseler**: AAPL, NVDA, TSLA, MSFT, GOOGL, AMZN, META
                    - **Kripto**: BTC-USD, ETH-USD, SOL-USD
                    - **ETF**: SPY, QQQ, IWM
                    - **Emtialar**: GC=F (Altın), CL=F (Petrol)
                    """)

                with gr.Column(scale=2):
                    output = gr.Textbox(
                        label="📋 Trade Advisor Raporu",
                        lines=35,
                        max_lines=60,
                        show_copy_button=True,
                    )

            analyze_btn.click(
                fn=quick_analysis,
                inputs=[ticker_input, portfolio_input, risk_input],
                outputs=output,
            )

        # ---- Tab 2: Serbest Sohbet ----
        with gr.Tab("💬 Serbest Sohbet"):
            gr.Markdown("""
            Doğal dilde soru sorun. Agent'lar otomatik olarak gerekli araçları kullanacak.
            
            **Örnek sorular:**
            - "AAPL ve MSFT'yi karşılaştır, hangisi daha iyi bir yatırım?"
            - "Piyasa genel durumu nasıl? Şu an hisse almak mantıklı mı?"
            - "BTC-USD için kısa vadeli teknik analiz yap"
            - "TSLA hissesi aşırı mı değerli?"
            """)
            chatbot = gr.ChatInterface(
                fn=chat_analysis,
                type="messages",
                examples=[
                    "Analyze AAPL for a swing trade. My portfolio is $50,000 with moderate risk tolerance.",
                    "What's the overall market sentiment right now? Is it a good time to buy stocks?",
                    "Compare NVDA and AMD - which is a better buy right now?",
                    "Give me a technical analysis of BTC-USD for the next week.",
                    "Is TSLA overvalued at current levels? What do fundamentals say?",
                ],
            )

    gr.Markdown("""
    ---
    **Built with** 🤗 smolagents + yfinance + pandas-ta | 
    **Inspired by** [TradingAgents Paper](https://arxiv.org/abs/2412.20138) |
    **⚠️ Not financial advice**
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
