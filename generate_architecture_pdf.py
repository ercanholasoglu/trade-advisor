#!/usr/bin/env python3
"""Mimari diyagram PDF oluşturucu — emoji-free, tüm fontlarda çalışır."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

C = {'bg':'#0d1117','text':'#e6edf3','dim':'#8b949e','blue':'#58a6ff','green':'#3fb950',
     'red':'#f85149','orange':'#d29922','purple':'#a371f7','yellow':'#f0c000','cyan':'#39d2c0',
     'box_bg':'#161b22','border':'#30363d'}

def box(ax,x,y,w,h,title,sub,color):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.4",facecolor=C['box_bg'],edgecolor=color,lw=2))
    ax.text(x+w/2,y+h/2+(1.5 if sub else 0),title,ha='center',va='center',fontsize=9,fontweight='bold',color=color)
    if sub: ax.text(x+w/2,y+h/2-2,sub,ha='center',va='center',fontsize=6.5,color=C['dim'])

def arrow(ax,x1,y1,x2,y2,c='#30363d',lw=1.2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color=c,lw=lw))

def label(ax,x,y,t,fs=14,c='#e6edf3'):
    ax.text(x,y,t,ha='center',va='center',fontsize=fs,fontweight='bold',color=c)

def create_pdf(path="trade_bot_architecture.pdf"):
    fig,axes=plt.subplots(3,1,figsize=(22,34)); fig.patch.set_facecolor(C['bg'])
    for a in axes: a.set_xlim(0,100);a.set_ylim(0,100);a.set_aspect('equal');a.axis('off');a.set_facecolor(C['bg'])

    # === PAGE 1: GENEL MIMARI ===
    ax=axes[0]
    label(ax,50,97,"AGENTIC TRADE BOT ADVISOR - GENEL MIMARI",16)
    label(ax,50,93,"9 Agent + 9 Tool + LLM Council + Fund Manager",10,C['dim'])

    box(ax,35,84,30,7,"[KULLANICI]","Ticker + Portfolio + Risk",C['cyan'])
    arrow(ax,50,84,50,79,C['cyan'],2)
    box(ax,25,71,50,8,"[FUND MANAGER] CodeAgent","Orchestrator - tum agent'lari yonetir",C['purple'])

    label(ax,50,67,"--- PHASE 1: VERI TOPLAMA ---",10,C['blue'])
    agents=[(3,"[TECHNICAL]\nAnalyst","RSI,MACD\nBollinger,SMA",C['blue']),(22,"[SENTIMENT]\nAnalyst","Haberler\n+KAP",C['green']),
            (41,"[FUNDAMENTAL]\nAnalyst","P/E,PEG\nMarjlar",C['orange']),(60,"[BIST]\nAnalyst","BIST30,Altin\nUSD/TRY",C['red']),
            (79,"[RISK]\nManager","VaR,SL/TP\nVIX",C['yellow'])]
    for x,lbl,sub,c in agents:
        box(ax,x,48,17,14,lbl,sub,c)
        arrow(ax,50,71,x+8.5,62,C['border'],1)

    label(ax,50,44,"--- PHASE 2: LLM COUNCIL ---",10,C['orange'])
    council=[(10,"[BULL] Bullish\nAdvisor","En guclu AL\nargumani",C['green']),
             (40,"[BEAR] Bearish\nAdvisor","En guclu SAT\nargumani",C['red']),
             (70,"[MEDIATOR]\nNeutral","Dengeyi kurar\nBlind spot bulur",C['yellow'])]
    for x,lbl,sub,c in council:
        box(ax,x,25,22,14,lbl,sub,c)

    for x,_,_,_ in agents:
        for cx,_,_,_ in council: arrow(ax,x+8.5,48,cx+11,39,C['border'],0.5)
    for cx,_,_,c in council: arrow(ax,cx+11,39,50,71,c,1.5)

    box(ax,25,6,50,12,"[FINAL RAPOR]","Council Vote -> Consensus -> BUY/SELL/HOLD",C['purple'])
    arrow(ax,50,71,50,18,C['purple'],2)

    # === PAGE 2: VERI AKISI ===
    ax=axes[1]
    label(ax,50,97,"VERI AKIS DIYAGRAMI - ADIM ADIM",16)
    steps=[
        (92,"Kullanici: 'THYAO.IS analiz et, $100K, moderate risk'",C['cyan']),
        (85,"Fund Manager: Workflow baslat",C['purple']),
        (78,"STEP 1: technical_analyst('THYAO.IS')",C['blue']),
        (73,"  -> get_price_history -> yfinance -> OHLCV",C['blue']),
        (68,"  -> get_technical_indicators -> pandas-ta -> RSI:53 MACD:BULL",C['blue']),
        (61,"STEP 2: sentiment_analyst('THYAO','THY')",C['green']),
        (56,"  -> get_news_sentiment -> DuckDuckGo -> 8 haber",C['green']),
        (51,"  -> get_kap_disclosures -> KAP API -> Bildirimler",C['green']),
        (44,"STEP 3: fundamental_analyst('THYAO.IS')",C['orange']),
        (39,"  -> get_fundamental_data -> yfinance -> P/E:4.4",C['orange']),
        (32,"STEP 4: bist_analyst",C['red']),
        (27,"  -> get_bist_scanner('commodities_try') -> Altin:6798 TRY/gr",C['red']),
        (20,"STEP 5: risk_manager('THYAO.IS',100000,'moderate')",C['yellow']),
        (15,"  -> get_market_overview -> VIX:18.5 BIST:BEARISH",C['yellow']),
        (10,"  -> calculate_risk_metrics -> SL:309 TP:337 176 hisse",C['yellow']),
    ]
    for y,t,c in steps:
        ax.text(50,y,t,ha='center',va='center',fontsize=8,color=c,family='monospace',
                bbox=dict(boxstyle='round,pad=0.3',facecolor=C['box_bg'],edgecolor=c,alpha=0.7,lw=1))
    for i in range(len(steps)-1):
        arrow(ax,6,steps[i][0]-1.5,6,steps[i+1][0]+1.5,C['border'],1)

    # === PAGE 3: LLM COUNCIL KARAR MEKANIZMASI ===
    ax=axes[2]
    label(ax,50,97,"LLM COUNCIL KARAR MEKANIZMASI",16)
    label(ax,50,93,"Phase 1 verileri -> 3 farkli perspektif -> Oylama -> Final Karar",10,C['dim'])

    box(ax,20,82,60,8,"[VERI PAKETI] (Phase 1 ciktisi)","Technical + Sentiment + Fundamental + BIST + Risk",C['blue'])

    # Bull detay
    box(ax,2,52,28,24,"[BULL] BULLISH ADVISOR","",C['green'])
    ax.text(16,60,"Gorevi: En guclu AL vakasi\n\n- Pozitif sinyalleri bulur\n- Neden AL'i aciklar\n- Bear argumanlarini curutur\n- Conviction: 1-10 skor\n\nCikti: STRONG_BUY / BUY\n       / LEAN_BUY + %hedef",
            ha='center',va='center',fontsize=6.5,color=C['green'],family='monospace')

    # Bear detay
    box(ax,36,52,28,24,"[BEAR] BEARISH ADVISOR","",C['red'])
    ax.text(50,60,"Gorevi: En guclu SAT vakasi\n\n- Negatif sinyalleri bulur\n- Neden SATMA'yi aciklar\n- Bull argumanlarini curutur\n- Conviction: 1-10 skor\n\nCikti: STRONG_SELL / SELL\n       / LEAN_SELL + %risk",
            ha='center',va='center',fontsize=6.5,color=C['red'],family='monospace')

    # Mediator detay
    box(ax,70,52,28,24,"[MEDIATOR] NEUTRAL","",C['yellow'])
    ax.text(84,60,"Gorevi: Hakem + Dengeleyici\n\n- Arguman kalitesini olcer\n- Zayif mantigi tespit eder\n- Kor noktalari bulur\n- Bull/Bear guc skoru verir\n\nCikti: BUY / HOLD / SELL\n       + Confidence %0-100",
            ha='center',va='center',fontsize=6.5,color=C['yellow'],family='monospace')

    arrow(ax,50,82,16,76,C['green'],1.5)
    arrow(ax,50,82,50,76,C['red'],1.5)
    arrow(ax,50,82,84,76,C['yellow'],1.5)
    arrow(ax,30,64,36,64,C['dim'],1); ax.text(33,66,"Bull+Bear ->",fontsize=6,color=C['dim'],ha='center')

    # Oylama
    box(ax,15,30,70,16,"[OYLAMA] COUNCIL VOTE","",C['purple'])
    ax.text(50,36,"3/3 ayni yon   -> STRONG consensus -> HIGH confidence (80%+)\n"
            "2/3 ayni yon   -> MODERATE consensus -> cogunlugu takip et\n"
            "Hepsi farkli   -> SPLIT -> HOLD oner, confidence dusuk (<50%)\n"
            "Mediator       -> Esitlikte mediator'in oyu belirler\n"
            "VIX > 30       -> OVERRIDE -> ne olursa olsun HOLD",
            ha='center',va='center',fontsize=7,color=C['text'],family='monospace')

    for cx in [16,50,84]: arrow(ax,cx,52,cx if cx!=84 else 65,46,C['border'],1.5)

    box(ax,20,12,60,12,"[FUND MANAGER] FINAL KARAR","Council oylari + Conviction + Risk verdict -> Sentez",C['purple'])
    arrow(ax,50,30,50,24,C['purple'],2)

    box(ax,25,2,50,7,"[TRADE ADVISOR REPORT]","Signal | Confidence | Council Debate | Trade Params",C['cyan'])
    arrow(ax,50,12,50,9,C['cyan'],2)

    plt.tight_layout(pad=2)
    plt.savefig(path,format='pdf',dpi=150,facecolor=C['bg'],bbox_inches='tight')
    plt.close()
    print(f"PDF: {path}")

if __name__=="__main__": create_pdf()
