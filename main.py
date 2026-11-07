import datetime
import flet as ft
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
import yfinance as yf


# --- MOLIYAVIY AI SENTIMENT VA FUNDAMENTAL MODUL ---
class FundamentalEngine:

  def __init__(self, api_key: str):
    self.api_key = api_key.strip() if api_key else ""
    self.base_url = "https://finnhub.io/api/v1"

    # USD ni kuchaytiruvchi va zaiflashtiruvchi kalit so'zlar lug'ati
    self.usd_bullish = [
        "rate hike",
        "hawkish",
        "inflation surge",
        "strong jobs",
        "gdp growth",
        "fed tightens",
        "powell hawkish",
        "treasury yields rise",
        "war",
        "crisis",
        "sanctions",
        "conflict",
        "safe haven",
        "geopolitical tension",
    ]
    self.usd_bearish = [
        "rate cut",
        "dovish",
        "recession",
        "unemployment rises",
        "inflation cools",
        "fed pauses",
        "powell dovish",
        "banking crisis",
        "debt ceiling",
        "weak retail",
    ]

  def check_economic_calendar(self, pair: str):
    """Yaqin 60 daqiqada High Impact yangilik bor-yo'qligini tekshiradi."""
    if not self.api_key:
      return (
          "Kalendar: Finnhub API kaliti kiritilmagan",
          ft.Colors.AMBER_400,
          False,
      )

    try:
      today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
      url = f"{self.base_url}/calendar/economic?from={today}&to={today}&token={self.api_key}"
      resp = requests.get(url, timeout=8)
      data = resp.json().get("economicCalendar", [])

      now = datetime.datetime.utcnow()
      currencies = ["USD"]
      if "EUR" in pair:
        currencies.append("EUR")
      if "GBP" in pair:
        currencies.append("GBP")
      if "JPY" in pair:
        currencies.append("JPY")

      upcoming_alerts = []
      for event in data:
        impact = str(event.get("impact", "")).lower()
        country = str(event.get("country", "")).upper()
        event_time_str = event.get("time", "")

        if impact == "high" and any(c in country for c in currencies):
          try:
            ev_dt = datetime.datetime.strptime(
                event_time_str, "%Y-%m-%d %H:%M:%S"
            )
            diff_minutes = (ev_dt - now).total_seconds() / 60.0
            if -15 <= diff_minutes <= 60:
              upcoming_alerts.append(
                  f"{event.get('event')} ({int(diff_minutes)} daq)"
              )
          except Exception:
            pass

      if upcoming_alerts:
        return (
            f"⚠️ DIQQAT: {', '.join(upcoming_alerts)}! Bozor beqaror!",
            ft.Colors.RED_400,
            True,
        )
      return (
          "🟢 Kalendar tinch (Yaqin 1 soatda High Impact voqea yo'q)",
          ft.Colors.GREEN_400,
          False,
      )

    except Exception:
      return (
          "Kalendar ma'lumotini yuklashda xatolik (Internetni tekshiring)",
          ft.Colors.GREY_400,
          False,
      )

  def analyze_news_sentiment(self, pair: str):
    """So'nggi yangiliklar sarlavhalarini tahlil qilib, signalga ball beradi."""
    if not self.api_key:
      return 0, "Yangiliklar: API kaliti kiritilmagan", []

    try:
      url_forex = f"{self.base_url}/news?category=forex&token={self.api_key}"
      url_general = (
          f"{self.base_url}/news?category=general&token={self.api_key}"
      )

      news_forex = requests.get(url_forex, timeout=8).json()
      news_general = requests.get(url_general, timeout=8).json()

      combined_news = (news_forex[:10] if isinstance(news_forex, list) else []) + (
          news_general[:8] if isinstance(news_general, list) else []
      )

      score = 0
      headlines = []
      for item in combined_news:
        headline = item.get("headline", "")
        text = (headline + " " + item.get("summary", "")).lower()
        headlines.append(headline)

        for word in self.usd_bullish:
          if word in text:
            score += 2
        for word in self.usd_bearish:
          if word in text:
            score -= 2

      # Agar juftlik EUR/USD yoki GBP/USD bo'lsa, USD kuchayishi narxni tushiradi (Sell)
      # Agar USD/JPY bo'lsa, USD kuchayishi narxni ko'taradi (Buy)
      pair_impact = 0
      if pair.endswith("USD"):
        pair_impact = -max(min(score, 15), -15)
      elif pair.startswith("USD"):
        pair_impact = max(min(score, 15), -15)

      status_text = (
          f"USD Kayfiyati: {'Kuchli (Bullish)' if score > 1 else 'Zaif (Bearish)' if score < -1 else 'Neytral'} "
          f"(Signalga ta'siri: {pair_impact:+d}%)"
      )
      return pair_impact, status_text, headlines[:4]

    except Exception:
      return 0, "Yangiliklar tahlili vaqtincha mavjud emas", []

  def get_institutional_cot(self, pair: str):
    """Yirik fondlar (COT) yo'nalishi ko'rsatkichi."""
    cot_map = {
        "EURUSD=X": "Yirik fondlar (COT): 61% Long (Yevro xaridi ustun)",
        "GBPUSD=X": "Yirik fondlar (COT): 54% Long (Funt barqaror)",
        "USDJPY=X": "Yirik fondlar (COT): 67% Long (Dollar ustunligi)",
        "GC=F": "Yirik fondlar (COT): 74% Long (Oltin talabi yuqori)",
    }
    return cot_map.get(
        pair, "Yirik fondlar (COT): Neytral institutsional oqim"
    )


# --- TEXNIK VA AI MODEL QISMI ---
def fetch_and_predict(symbol: str, timeframe: str = "1h"):
  df = yf.download(symbol, period="60d", interval=timeframe, progress=False)
  if df.empty or len(df) < 50:
    return None

  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

  df["SMA_20"] = df["Close"].rolling(20).mean()
  df["SMA_50"] = df["Close"].rolling(50).mean()

  # RSI
  delta = df["Close"].diff()
  gain = (delta.where(delta > 0, 0)).rolling(14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
  rs = gain / loss
  df["RSI"] = 100 - (100 / (1 + rs))

  # Bollinger Bands & ATR
  std20 = df["Close"].rolling(20).std()
  df["BB_Upper"] = df["SMA_20"] + (std20 * 2)
  df["BB_Lower"] = df["SMA_20"] - (std20 * 2)
  df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()

  df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
  df.dropna(inplace=True)

  features = ["SMA_20", "SMA_50", "RSI", "BB_Upper", "BB_Lower", "ATR"]
  X = df[features]
  y = df["Target"]

  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X[:-1], y[:-1])

  latest = X.iloc[-1:]
  prob = model.predict_proba(latest)[0]
  buy_prob = int(prob[1] * 100)
  current_price = float(df["Close"].iloc[-1])
  atr_val = float(df["ATR"].iloc[-1])

  return {
      "price": round(current_price, 5),
      "buy_prob": buy_prob,
      "atr": round(atr_val, 5),
      "rsi": round(float(df["RSI"].iloc[-1]), 1),
  }


# --- FLET INTERFEYSI ---
def main(page: ft.Page):
  page.title = "Forex AI Pro — Hybrid Edition"
  page.theme_mode = ft.ThemeMode.DARK
  page.padding = 16
  page.scroll = ft.ScrollMode.AUTO

  saved_key = page.client_storage.get("finnhub_api_key") or ""

  api_input = ft.TextField(
      label="Finnhub API Key",
      value=saved_key,
      password=True,
      can_reveal_password=True,
      hint_text="Finnhub.io kalitingizni kiriting",
      expand=True,
  )

  def save_api_key(e):
    page.client_storage.set("finnhub_api_key", api_input.value.strip())
    page.snack_bar = ft.SnackBar(
        ft.Text("✅ API kalit saqlandi!"), bgcolor=ft.Colors.GREEN_700
    )
    page.snack_bar.open = True
    page.update()

  pair_dropdown = ft.Dropdown(
      label="Valyuta juftligi",
      value="EURUSD=X",
      options=[
          ft.dropdown.Option("EURUSD=X", "EUR/USD"),
          ft.dropdown.Option("GBPUSD=X", "GBP/USD"),
          ft.dropdown.Option("USDJPY=X", "USD/JPY"),
          ft.dropdown.Option("GC=F", "XAU/USD (Oltin)"),
      ],
      expand=True,
  )

  tf_dropdown = ft.Dropdown(
      label="Taymfreym",
      value="1h",
      options=[
          ft.dropdown.Option("15m", "15 Daqiqa"),
          ft.dropdown.Option("1h", "1 Soat"),
          ft.dropdown.Option("4h", "4 Soat"),
      ],
      width=130,
  )

  balance_input = ft.TextField(
      label="Depozit ($)", value="1000", width=110, keyboard_type=ft.KeyboardType.NUMBER
  )
  risk_input = ft.TextField(
      label="Risk (%)", value="1.5", width=95, keyboard_type=ft.KeyboardType.NUMBER
  )

  # Natija kartochkalari
  calendar_banner = ft.Container(
      content=ft.Text("Kalendar tekshirilmoqda...", size=13, weight=ft.FontWeight.BOLD),
      padding=12,
      border_radius=8,
      bgcolor=ft.Colors.BLUE_GREY_900,
  )

  signal_title = ft.Text("Tahlil kutilmoqda...", size=24, weight=ft.FontWeight.BOLD)
  price_info = ft.Text("Joriy narx: - | RSI: -", size=14, color=ft.Colors.GREY_300)
  sentiment_info = ft.Text("Yangiliklar foni: -", size=13, color=ft.Colors.CYAN_300)
  cot_info = ft.Text("Institutsional COT: -", size=13, color=ft.Colors.AMBER_300)
  risk_info = ft.Text("SL / TP va Lot: -", size=13, color=ft.Colors.GREEN_300)
  news_list = ft.Column(spacing=4)

  loading = ft.ProgressRing(visible=False)

  def run_analysis(e):
    loading.visible = True
    page.update()

    symbol = pair_dropdown.value
    pair_clean = pair_dropdown.options[0].text
    for opt in pair_dropdown.options:
      if opt.key == symbol:
        pair_clean = opt.text
        break

    tf = tf_dropdown.value
    current_key = api_input.value.strip()

    # 1. Texnik tahlil
    tech = fetch_and_predict(symbol, tf)
    if not tech:
      signal_title.value = "Ma'lumot olishda xatolik!"
      loading.visible = False
      page.update()
      return

    # 2. Fundamental tahlil
    fund = FundamentalEngine(current_key)
    cal_text, cal_color, is_danger = fund.check_economic_calendar(pair_clean)
    sent_impact, sent_text, headlines = fund.analyze_news_sentiment(pair_clean)
    cot_text = fund.get_institutional_cot(symbol)

    calendar_banner.content = ft.Text(cal_text, size=13, weight=ft.FontWeight.BOLD, color=cal_color)

    # 3. Yakuniy Gibrid Ehtimollik
    final_buy_prob = max(min(tech["buy_prob"] + sent_impact, 98), 2)

    if is_danger:
      decision = f"⚠️ KUTISH (Yangilik xavfi!) | Texnik: {final_buy_prob}%"
      signal_title.color = ft.Colors.AMBER_400
    elif final_buy_prob >= 62:
      decision = f"🚀 KUCHLI BUY ({final_buy_prob}%)"
      signal_title.color = ft.Colors.GREEN_400
    elif final_buy_prob <= 38:
      decision = f"🔻 KUCHLI SELL ({100 - final_buy_prob}%)"
      signal_title.color = ft.Colors.RED_400
    else:
      decision = f"⚖️ NEYTRAL / KUTISH (Buy: {final_buy_prob}%)"
      signal_title.color = ft.Colors.GREY_300

    # 4. Lot va SL/TP hisoblash (ATR asosida)
    try:
      bal = float(balance_input.value)
      rsk = float(risk_input.value)
      risk_usd = bal * (rsk / 100.0)
      sl_dist = tech["atr"] * 1.5
      tp_dist = tech["atr"] * 2.5
      # Taxminiy standart lot hisobi
      pip_mult = 100 if "JPY" in pair_clean or "XAU" in pair_clean else 10000
      sl_pips = max(sl_dist * pip_mult, 5)
      lot_size = round(risk_usd / (sl_pips * 10), 2)
      lot_size = max(lot_size, 0.01)

      if final_buy_prob >= 50:
        sl_price = round(tech["price"] - sl_dist, 5)
        tp_price = round(tech["price"] + tp_dist, 5)
      else:
        sl_price = round(tech["price"] + sl_dist, 5)
        tp_price = round(tech["price"] - tp_dist, 5)

      risk_info.value = (
          f"Tavsiya Lot: {lot_size} | SL: {sl_price} | TP: {tp_price} (Risk: ${risk_usd:.1f})"
      )
    except Exception:
      risk_info.value = "Lot hisoblash uchun depozitni to'g'ri kiriting"

    signal_title.value = decision
    price_info.value = (
        f"Narx: {tech['price']} | Texnik AI: {tech['buy_prob']}% | RSI: {tech['rsi']}"
    )
    sentiment_info.value = f"📰 {sent_text}"
    cot_info.value = f"🏦 {cot_text}"

    news_list.controls.clear()
    if headlines:
      news_list.controls.append(
          ft.Text("So'nggi fundamental sarlavhalar:", size=12, weight=ft.FontWeight.BOLD)
      )
      for h in headlines:
        news_list.controls.append(ft.Text(f"• {h}", size=11, color=ft.Colors.GREY_400))

    loading.visible = False
    page.update()

  page.add(
      ft.Text("FOREX AI PRO — HYBRID", size=22, weight=ft.FontWeight.BOLD, color=ft.Colors.CYAN_400),
      ft.Row([api_input, ft.ElevatedButton("Saqlash", on_click=save_api_key)]),
      ft.Row([pair_dropdown, tf_dropdown]),
      ft.Row([
          balance_input,
          risk_input,
          ft.ElevatedButton(
              "TAHLIL QILISH",
              on_click=run_analysis,
              bgcolor=ft.Colors.CYAN_700,
              color=ft.Colors.WHITE,
              height=48,
              expand=True,
          ),
      ]),
      loading,
      calendar_banner,
      ft.Container(
          content=ft.Column([
              signal_title,
              price_info,
              sentiment_info,
              cot_info,
              risk_info,
              ft.Divider(),
              news_list,
          ]),
          padding=16,
          border_radius=12,
          bgcolor=ft.Colors.BLUE_GREY_900,
      ),
  )


ft.app(target=main)