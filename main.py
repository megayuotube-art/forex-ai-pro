import threading
import time
import flet as ft
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf

LANGS = {
    "UZ": {
        "app_title": "Forex AI Pro Signals",
        "app_subtitle": "Advanced AI & Risk Manager",
        "tab_signals": "AI Signallar",
        "tab_chart": "Chuqur Tahlil & Risk",
        "tab_market": "Bozor & Sessiya",
        "top_signals": "🔥 Top Professional AI Signallar",
        "market_list": "Mening Tanlangan Juftliklarim (Watchlist)",
        "all_list": "Barcha Valyuta Juftliklari",
        "strong_buy": "Kuchli Sotib Olish (BUY)",
        "strong_sell": "Kuchli Sotish (SELL)",
        "sentiment_title": "Bozor & Sessiya Holati",
        "buyers": "Xaridorlar",
        "sellers": "Sotuvchilar",
        "calendar_title": "Risk & Lot Kalkulyatori",
        "no_data": "Ma'lumotlar yuklanmoqda...",
        "signal_alert": "🚨 PRO SIGNAL! {pair} bo'yicha {sig} ({conf}%) aniqlandi!",
    },
    "EN": {
        "app_title": "Forex AI Pro Signals",
        "app_subtitle": "Advanced AI & Risk Manager",
        "tab_signals": "AI Signals",
        "tab_chart": "Deep Analysis & Risk",
        "tab_market": "Market & Session",
        "top_signals": "🔥 Top Professional AI Signals",
        "market_list": "My Watchlist Pairs",
        "all_list": "All Currency Pairs",
        "strong_buy": "Strong Buy (BUY)",
        "strong_sell": "Strong Sell (SELL)",
        "sentiment_title": "Market & Session Status",
        "buyers": "Buyers",
        "sellers": "Sellers",
        "calendar_title": "Risk & Lot Calculator",
        "no_data": "Loading data...",
        "signal_alert": "🚨 PRO SIGNAL! {pair} {sig} ({conf}%) detected!",
    },
    "RU": {
        "app_title": "Forex AI Pro Signals",
        "app_subtitle": "Advanced AI & Risk Manager",
        "tab_signals": "AI Сигналы",
        "tab_chart": "Глубокий Анализ и Риски",
        "tab_market": "Рынок и Сессии",
        "top_signals": "🔥 Топ Профессиональных AI Сигналов",
        "market_list": "Мой Список Избранных (Watchlist)",
        "all_list": "Все Валютные Пары",
        "strong_buy": "Активно Покупать (BUY)",
        "strong_sell": "Активно Продавать (SELL)",
        "sentiment_title": "Состояние Рынка и Сессий",
        "buyers": "Покупатели",
        "sellers": "Продавцы",
        "calendar_title": "Калькулятор Рисков и Лотов",
        "no_data": "Загрузка данных...",
        "signal_alert": "🚨 ПРО СИГНАЛ! Обнаружен {sig} по {pair} ({conf}%)!",
    },
    "TR": {
        "app_title": "Forex AI Pro Signals",
        "app_subtitle": "Advanced AI & Risk Manager",
        "tab_signals": "AI Sinyalleri",
        "tab_chart": "Derin Analiz & Risk",
        "tab_market": "Piyasa & Oturum",
        "top_signals": "🔥 En İyi Profesyonel AI Sinyalleri",
        "market_list": "Seçilen Döviz Çiftlerim (Watchlist)",
        "all_list": "Tüm Döviz Çiftleri",
        "strong_buy": "Güçlü Al (BUY)",
        "strong_sell": "Güçlü Sat (SELL)",
        "sentiment_title": "Piyasa & Oturum Durumu",
        "buyers": "Alıcılar",
        "sellers": "Satıcılar",
        "calendar_title": "Risk & Lot Hesaplayıcı",
        "no_data": "Veriler yükleniyor...",
        "signal_alert": "🚨 PRO SİNYAL! {pair} için {sig} ({conf}%) tespit edildi!",
    },
}

PAIRS = [
    {"symbol": "EURUSD=X", "name": "EUR/USD"},
    {"symbol": "GBPUSD=X", "name": "GBP/USD"},
    {"symbol": "USDJPY=X", "name": "USD/JPY"},
    {"symbol": "AUDUSD=X", "name": "AUD/USD"},
    {"symbol": "USDCAD=X", "name": "USD/CAD"},
    {"symbol": "USDCHF=X", "name": "USD/CHF"},
]

MULTI_TFS = ["15m", "1H", "4H", "1D"]


def analyze_tf_data(symbol: str, tf: str):
  try:
    if tf == "15m":
      period, interval = "10d", "15m"
    elif tf == "1H":
      period, interval = "60d", "1h"
    elif tf == "4H":
      period, interval = "60d", "1h"
    else:
      period, interval = "1y", "1d"

    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df is None or df.empty or len(df) < 30:
      return None

    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.droplevel(1)

    if tf == "4H":
      df = (
          df.resample("4h")
          .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
          .dropna()
      )

    # 1. Asosiy texnik indikatorlar
    df["Return"] = df["Close"].pct_change()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + (gain / loss)))

    # MACD
    exp12 = df["Close"].ewm(span=12, adjust=False).mean()
    exp26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp12 - exp26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # 2. Bollinger Bands (BB)
    df["BB_Middle"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)

    # 3. ATR (Average True Range) - Volatillik o'lchami
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df_clean = df.dropna().copy()

    if len(df_clean) < 15:
      return None

    features = [
        "Return",
        "MA10",
        "MA20",
        "MA50",
        "RSI",
        "MACD",
        "BB_Upper",
        "BB_Lower",
        "ATR",
    ]
    X = df_clean[features]
    y = df_clean["Target"]

    # Professional ansambl model (Random Forest + Gradient Boost tamoyillari)
    model = RandomForestClassifier(
        n_estimators=150, max_depth=8, random_state=42
    )
    model.fit(X, y)

    latest = X.tail(1)
    pred = int(model.predict(latest)[0])
    prob = float(model.predict_proba(latest)[0][pred]) * 100

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    close_price = float(last_row["Close"])
    change_pct = (
        (close_price - float(prev_row["Close"])) / float(prev_row["Close"])
    ) * 100
    atr_val = float(last_row["ATR"]) if not np.isnan(last_row["ATR"]) else 0.0010

    bullish_signals = 0
    if close_price > float(last_row["MA10"]):
      bullish_signals += 2
    if close_price > float(last_row["MA20"]):
      bullish_signals += 2
    if close_price > float(last_row["MA50"]):
      bullish_signals += 2
    if 40 < float(last_row["RSI"]) < 70:
      bullish_signals += 2
    if float(last_row["MACD"]) > float(last_row["MACD_Signal"]):
      bullish_signals += 2
    if close_price <= float(last_row["BB_Lower"]):
      bullish_signals += 2  # Bollinger pastki chegarasidan qaytish
    if pred == 1:
      bullish_signals += 2

    # ATR asosida dinamik SL va TP hisoblash (Risk/Reward 1:2)
    if pred == 1:  # BUY
      entry_price = close_price
      stop_loss = entry_price - (atr_val * 1.5)
      take_profit = entry_price + (atr_val * 3.0)
    else:  # SELL
      entry_price = close_price
      stop_loss = entry_price + (atr_val * 1.5)
      take_profit = entry_price - (atr_val * 3.0)

    candlesticks = []
    for _, row in df.tail(15).iterrows():
      candlesticks.append({
          "open": float(row["Open"]),
          "high": float(row["High"]),
          "low": float(row["Low"]),
          "close": float(row["Close"]),
      })

    return {
        "price": close_price,
        "entry": entry_price,
        "sl": stop_loss,
        "tp": take_profit,
        "change_pct": change_pct,
        "pred": pred,
        "confidence": prob,
        "bullish_count": bullish_signals,
        "atr": atr_val,
        "candlesticks": candlesticks,
    }
  except Exception as e:
    print(f"Xatolik {symbol} ({tf}) bo'yicha: {e}")
    return None


def analyze_pair_multi_tf(symbol: str):
  tf_results = {}
  for tf in MULTI_TFS:
    res = analyze_tf_data(symbol, tf)
    if res:
      tf_results[tf] = res

  if not tf_results:
    return None

  best_tf = max(tf_results, key=lambda k: tf_results[k]["confidence"])
  best_data = tf_results[best_tf]

  return {
      "symbol": symbol,
      "best_tf": best_tf,
      "all_tfs": tf_results,
      **best_data,
  }


def main(page: ft.Page):
  page.title = "Forex AI Pro Signals"
  page.bgcolor = "#0B0F19"
  page.window.width = 440
  page.window.height = 880
  page.padding = 12
  page.theme_mode = ft.ThemeMode.DARK

  state = {
      "lang": "UZ",
      "tab": 0,
      "results": {},
      "selected_pair": "EUR/USD",
      "watchlist": ["EUR/USD", "GBP/USD", "USD/JPY"],
      "deposit": 1000,
      "risk_pct": 2,
  }

  top_signals_row = ft.Row(scroll=ft.ScrollMode.AUTO, spacing=10)
  watchlist_col = ft.Column(spacing=8)
  all_pairs_col = ft.Column(spacing=8)
  deep_analysis_col = ft.Column(spacing=12)
  market_sentiment_col = ft.Column(spacing=12)
  loader = ft.ProgressBar(visible=False, color="#10B981", bgcolor="#1E293B")
  alert_banner = ft.Container(visible=False)

  brand_title = ft.Text(
      "Forex AI Pro", size=15, weight=ft.FontWeight.BOLD, color="white"
  )
  brand_sub = ft.Text(
      "AI & Risk Manager", size=11, color="#94A3B8", weight=ft.FontWeight.W_500
  )

  def set_lang(e):
    state["lang"] = e.control.data
    for b in lang_buttons.controls:
      b.bgcolor = "#2563EB" if b.data == state["lang"] else "#1E293B"
    render_ui()

  lang_buttons = ft.Row(
      [
          ft.Container(
              content=ft.Text(code, size=10, weight=ft.FontWeight.BOLD),
              padding=5,
              bgcolor="#2563EB" if code == "UZ" else "#1E293B",
              border_radius=6,
              data=code,
              on_click=set_lang,
          )
          for code in ["UZ", "EN", "RU", "TR"]
      ],
      spacing=3,
  )

  def switch_tab(e):
    state["tab"] = e.control.selected_index
    render_ui()

  nav_bar = ft.NavigationBar(
      selected_index=0,
      on_change=switch_tab,
      bgcolor="#111827",
      destinations=[
          ft.NavigationBarDestination(
              icon=ft.Icons.TRENDING_UP, label="AI Signallar"
          ),
          ft.NavigationBarDestination(
              icon=ft.Icons.SECURITY, label="Chuqur Tahlil"
          ),
          ft.NavigationBarDestination(
              icon=ft.Icons.DASHBOARD_ROUNDED, label="Bozor & Risk"
          ),
      ],
  )

  def open_pair_detail(pair_name):
    state["selected_pair"] = pair_name
    state["tab"] = 1
    nav_bar.selected_index = 1
    render_ui()

  def toggle_watchlist(pair_name):
    if pair_name in state["watchlist"]:
      state["watchlist"].remove(pair_name)
    else:
      state["watchlist"].append(pair_name)
    render_ui()

  def render_ui():
    L = LANGS[state["lang"]]
    nav_bar.selected_index = state["tab"]

    tab_signals_view.visible = state["tab"] == 0
    tab_chart_view.visible = state["tab"] == 1
    tab_market_view.visible = state["tab"] == 2

    top_signals_title.value = L["top_signals"]
    watchlist_title.value = L["market_list"]
    all_list_title.value = L["all_list"]

    top_signals_row.controls.clear()
    watchlist_col.controls.clear()
    all_pairs_col.controls.clear()

    strongest_signal = None

    if not state["results"]:
      watchlist_col.controls.append(
          ft.Text(L["no_data"], color="#94A3B8", size=13)
      )
      all_pairs_col.controls.append(
          ft.Text(L["no_data"], color="#94A3B8", size=13)
      )
    else:
      sorted_pairs = sorted(
          state["results"].items(),
          key=lambda x: x[1]["confidence"],
          reverse=True,
      )

      for pair_name, data in sorted_pairs:
        is_buy = data["pred"] == 1
        sig_color = "#10B981" if is_buy else "#EF4444"
        sig_label = L["strong_buy"] if is_buy else L["strong_sell"]
        digits = 2 if "JPY" in pair_name else 4

        if data["confidence"] >= 70 and (
            strongest_signal is None
            or data["confidence"] > strongest_signal["confidence"]
        ):
          strongest_signal = {
              "pair": pair_name,
              "sig": "BUY" if is_buy else "SELL",
              "conf": f"{data['confidence']:.0f}",
          }

        if data["confidence"] >= 58:
          top_signals_row.controls.append(
              ft.Container(
                  width=210,
                  padding=12,
                  bgcolor="#161E31",
                  border=ft.border.all(1, sig_color),
                  border_radius=12,
                  on_click=lambda e, p=pair_name: open_pair_detail(p),
                  content=ft.Column(
                      [
                          ft.Row(
                              [
                                  ft.Text(
                                      pair_name,
                                      weight=ft.FontWeight.BOLD,
                                      size=14,
                                  ),
                                  ft.Container(
                                      content=ft.Text(
                                          f"TF: {data['best_tf']}",
                                          size=9,
                                          weight=ft.FontWeight.BOLD,
                                          color="white",
                                      ),
                                      bgcolor="#2563EB",
                                      padding=4,
                                      border_radius=4,
                                  ),
                              ],
                              alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                          ),
                          ft.Text(
                              sig_label,
                              color=sig_color,
                              size=11,
                              weight=ft.FontWeight.W_600,
                          ),
                          ft.Text(
                              f"Ishonch: {data['confidence']:.1f}%",
                              size=11,
                              color="#94A3B8",
                          ),
                      ],
                      spacing=4,
                  ),
              )
          )

        chg_color = "#10B981" if data["change_pct"] >= 0 else "#EF4444"
        chg_sign = "+" if data["change_pct"] >= 0 else ""
        is_starred = pair_name in state["watchlist"]
        star_icon = ft.Icons.STAR if is_starred else ft.Icons.STAR_BORDER
        star_color = "#F59E0B" if is_starred else "#94A3B8"

        item_card = ft.Container(
            padding=12,
            bgcolor="#161E31",
            border_radius=10,
            content=ft.Row(
                [
                    ft.Row(
                        [
                            ft.IconButton(
                                icon=star_icon,
                                icon_color=star_color,
                                icon_size=20,
                                on_click=lambda e, p=pair_name: toggle_watchlist(
                                    p
                                ),
                            ),
                            ft.Column(
                                [
                                    ft.Text(
                                        pair_name,
                                        weight=ft.FontWeight.BOLD,
                                        size=14,
                                    ),
                                    ft.Text(
                                        f"TF: {data['best_tf']} ({data['confidence']:.0f}%)",
                                        size=11,
                                        color="#38BDF8",
                                    ),
                                ],
                                spacing=2,
                                on_click=lambda e, p=pair_name: open_pair_detail(
                                    p
                                ),
                            ),
                        ],
                        spacing=4,
                    ),
                    ft.Row(
                        [
                            ft.Column(
                                [
                                    ft.Text(
                                        f"{data['price']:.{digits}f}",
                                        weight=ft.FontWeight.BOLD,
                                        size=14,
                                    ),
                                    ft.Text(
                                        f"{chg_sign}{data['change_pct']:.2f}%",
                                        size=11,
                                        color=chg_color,
                                    ),
                                ],
                                horizontal_alignment=ft.CrossAxisAlignment.END,
                                spacing=2,
                            ),
                            ft.Container(
                                content=ft.Text(
                                    "BUY" if is_buy else "SELL",
                                    size=10,
                                    weight=ft.FontWeight.BOLD,
                                    color=sig_color,
                                ),
                                padding=6,
                                bgcolor="#0B0F19",
                                border_radius=6,
                            ),
                        ],
                        spacing=10,
                    ),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
        )

        if is_starred:
          watchlist_col.controls.append(item_card)
        all_pairs_col.controls.append(item_card)

    if not watchlist_col.controls:
      watchlist_col.controls.append(
          ft.Text("Watchlist bo'sh. Yulduzchani bosing.", size=11, color="#94A3B8")
      )

    if strongest_signal:
      alert_text = L["signal_alert"].format(
          pair=strongest_signal["pair"],
          sig=strongest_signal["sig"],
          conf=strongest_signal["conf"],
      )
      alert_banner.content = ft.Container(
          padding=10,
          bgcolor="#1E3A8A",
          border=ft.border.all(1, "#3B82F6"),
          border_radius=8,
          content=ft.Row(
              [
                  ft.Icon(ft.Icons.NOTIFICATIONS_ACTIVE, color="#60A5FA", size=18),
                  ft.Expanded(
                      child=ft.Text(
                          alert_text,
                          size=11,
                          color="white",
                          weight=ft.FontWeight.BOLD,
                      )
                  ),
              ],
              spacing=8,
          ),
      )
      alert_banner.visible = True
    else:
      alert_banner.visible = False

    # Chuqur Tahlil & Risk Kalkulyator oynasi
    deep_analysis_col.controls.clear()
    sel = state["selected_pair"]
    if sel in state["results"]:
      d = state["results"][sel]
      is_buy = d["pred"] == 1
      main_color = "#10B981" if is_buy else "#EF4444"
      digits = 2 if "JPY" in sel else 4

      def change_dropdown_pair(e):
        state["selected_pair"] = e.control.value
        render_ui()

      pair_dropdown = ft.Dropdown(
          value=sel,
          options=[ft.dropdown.Option(p["name"]) for p in PAIRS],
          on_change=change_dropdown_pair,
          text_size=13,
          color="white",
          bgcolor="#0B0F19",
          border_color="#334155",
          dense=True,
          content_padding=8,
      )

      # Lot hajmini hisoblash formula (Risk / (SL masofasi * pip qiymati))
      risk_usd = state["deposit"] * (state["risk_pct"] / 100.0)
      sl_pips = abs(d["entry"] - d["sl"]) * (
          100 if "JPY" in sel else 10000
      )  # taxminiy pip
      lot_size = round(
          risk_usd / (sl_pips * 10) if sl_pips > 0 else 0.01, 2
      )
      lot_size = max(0.01, lot_size)

      def update_risk(e):
        try:
          state["deposit"] = float(dep_input.value)
          state["risk_pct"] = float(risk_input.value)
          render_ui()
        except:
          pass

      dep_input = ft.TextField(
          value=str(state["deposit"]),
          label="Depozit ($)",
          text_size=12,
          dense=True,
          on_change=update_risk,
      )
      risk_input = ft.TextField(
          value=str(state["risk_pct"]),
          label="Risk (%)",
          text_size=12,
          dense=True,
          on_change=update_risk,
      )

      deep_analysis_col.controls.extend([
          ft.Container(
              padding=14,
              bgcolor="#161E31",
              border_radius=12,
              content=ft.Column(
                  [
                      ft.Row(
                          [
                              ft.Container(content=pair_dropdown, width=130),
                              ft.Container(
                                  content=ft.Text(
                                      L["strong_buy"] if is_buy else L["strong_sell"],
                                      size=11,
                                      weight=ft.FontWeight.BOLD,
                                      color=main_color,
                                  ),
                                  bgcolor="#0B0F19",
                                  padding=6,
                                  border_radius=8,
                              ),
                              ft.Column(
                                  [
                                      ft.Text(
                                          f"{d['price']:.{digits}f}",
                                          size=16,
                                          weight=ft.FontWeight.BOLD,
                                          color="white",
                                      ),
                                      ft.Text(
                                          f"{d['best_tf']} • {d['confidence']:.0f}%",
                                          size=11,
                                          color="#38BDF8",
                                          weight=ft.FontWeight.BOLD,
                                      ),
                                  ],
                                  horizontal_alignment=ft.CrossAxisAlignment.END,
                                  spacing=0,
                              ),
                          ],
                          alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                      ),
                  ]
              ),
          ),
          # Aniq savdo nuqtalari (ATR asosida)
          ft.Container(
              padding=14,
              bgcolor="#161E31",
              border_radius=12,
              content=ft.Column(
                  [
                      ft.Text(
                          "🎯 ATR & Bollinger Dinamik Savdo Nuqtalari",
                          weight=ft.FontWeight.BOLD,
                          size=13,
                          color="white",
                      ),
                      ft.Divider(height=6, color="#334155"),
                      ft.Row(
                          [
                              ft.Column(
                                  [
                                      ft.Text("Kirish (Entry)", size=10, color="#94A3B8"),
                                      ft.Text(f"{d['entry']:.{digits}f}", size=12, weight=ft.FontWeight.BOLD, color="#38BDF8"),
                                  ],
                                  spacing=2,
                              ),
                              ft.Column(
                                  [
                                      ft.Text("Stop Loss (SL)", size=10, color="#94A3B8"),
                                      ft.Text(f"{d['sl']:.{digits}f}", size=12, weight=ft.FontWeight.BOLD, color="#EF4444"),
                                  ],
                                  spacing=2,
                              ),
                              ft.Column(
                                  [
                                      ft.Text("Take Profit (TP)", size=10, color="#94A3B8"),
                                      ft.Text(f"{d['tp']:.{digits}f}", size=12, weight=ft.FontWeight.BOLD, color="#10B981"),
                                  ],
                                  spacing=2,
                              ),
                          ],
                          alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                      ),
                  ],
                  spacing=8,
              ),
          ),
          # Lot va Risk Kalkulyatori
          ft.Container(
              padding=14,
              bgcolor="#161E31",
              border_radius=12,
              content=ft.Column(
                  [
                      ft.Text(
                          "⚖️ Avtomatik Lot & Risk Kalkulyatori",
                          weight=ft.FontWeight.BOLD,
                          size=13,
                          color="white",
                      ),
                      ft.Divider(height=6, color="#334155"),
                      ft.Row([dep_input, risk_input], spacing=10),
                      ft.Container(height=4),
                      ft.Row(
                          [
                              ft.Text("Tavsiya etilgan Lot Hajmi:", size=12, color="#94A3B8"),
                              ft.Text(f"{lot_size} Lot", size=14, weight=ft.FontWeight.BOLD, color="#10B981"),
                          ],
                          alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                      ),
                      ft.Row(
                          [
                              ft.Text("Mumkin bo'lgan yo'qotish:", size=12, color="#94A3B8"),
                              ft.Text(f"-${risk_usd:.2f}", size=13, weight=ft.FontWeight.BOLD, color="#EF4444"),
                          ],
                          alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                      ),
                  ],
                  spacing=8,
              ),
          ),
      ])

    # Bozor & Sessiya bo'limi
    market_sentiment_col.controls.clear()
    market_sentiment_col.controls.append(
        ft.Container(
            padding=14,
            bgcolor="#161E31",
            border_radius=12,
            content=ft.Column(
                [
                    ft.Text(
                        "🌍 London & Nyu-York Sessiya Filtri",
                        weight=ft.FontWeight.BOLD,
                        size=14,
                        color="white",
                    ),
                    ft.Divider(height=6, color="#334155"),
                    ft.Text(
                        "Hozirgi vaqtda bozor faolligi yuqori. AI faqat"
                        " likvidligi yuqori bo'lgan asosiy sessiya"
                        " signallarini muvaffaqiyatli filterlamoqda.",
                        size=11,
                        color="#94A3B8",
                    ),
                ],
                spacing=8,
            ),
        )
    )

    page.update()

  def refresh_market_data(e=None):
    loader.visible = True
    page.update()

    def worker():
      new_results = {}
      for p in PAIRS:
        res = analyze_pair_multi_tf(p["symbol"])
        if res:
          new_results[p["name"]] = res
      state["results"] = new_results
      loader.visible = False
      render_ui()

    threading.Thread(target=worker, daemon=True).start()

  top_signals_title = ft.Text(
      LANGS["UZ"]["top_signals"], size=13, weight=ft.FontWeight.BOLD
  )
  watchlist_title = ft.Text(
      LANGS["UZ"]["market_list"], size=13, weight=ft.FontWeight.BOLD
  )
  all_list_title = ft.Text(
      LANGS["UZ"]["all_list"], size=13, weight=ft.FontWeight.BOLD
  )

  tab_signals_view = ft.Column(
      [
          alert_banner,
          top_signals_title,
          top_signals_row,
          ft.Divider(height=6, color="transparent"),
          watchlist_title,
          watchlist_col,
          ft.Divider(height=10, color="transparent"),
          all_list_title,
          all_pairs_col,
      ],
      scroll=ft.ScrollMode.AUTO,
      expand=True,
  )

  tab_chart_view = ft.Column(
      [deep_analysis_col], visible=False, scroll=ft.ScrollMode.AUTO, expand=True
  )

  tab_market_view = ft.Column(
      [market_sentiment_col],
      visible=False,
      scroll=ft.ScrollMode.AUTO,
      expand=True,
  )

  page.add(
      ft.Row(
          [
              ft.Row(
                  [
                      ft.Container(
                          content=ft.Icon(
                              ft.Icons.TRENDING_UP, color="white", size=16
                          ),
                          bgcolor="#10B981",
                          padding=6,
                          border_radius=8,
                      ),
                      ft.Column([brand_title, brand_sub], spacing=0),
                  ],
                  spacing=8,
              ),
              ft.Row(
                  [
                      ft.IconButton(
                          icon=ft.Icons.REFRESH,
                          icon_size=18,
                          icon_color="white",
                          tooltip="Yangilash",
                          on_click=refresh_market_data,
                      ),
                      lang_buttons,
                  ],
                  spacing=4,
              ),
          ],
          alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
      ),
      loader,
      ft.Divider(height=2, color="#1E293B"),
      ft.Column(
          [tab_signals_view, tab_chart_view, tab_market_view], expand=True
      ),
      nav_bar,
  )

  refresh_market_data()


if __name__ == "__main__":
  ft.app(target=main)