import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from advanced_ta import LorentzianClassification
from ta.momentum import RSIIndicator
from ta.trend import CCIIndicator, ADXIndicator, MACD
from ta.volume import money_flow_index as MFI
from alpaca_trade_api import REST
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from timedelta import Timedelta
from API_DATA import API_DATA

API_KEY , API_SECRET ,BASE_URL = API_DATA()
ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}
print(BASE_URL)
# Fetch stock data from Yahoo Finance
def fetch_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df.columns = df.columns.str.lower()
    df2 = pd.DataFrame()
    df2 = df[['date', 'open', 'high', 'low', 'close', 'volume']]
    print(df2.head())
    del df
    return df2

# Initialize the Lorentzian Classification
def initialize_lc(df):
    lc = LorentzianClassification(
        df,
        features=[
            LorentzianClassification.Feature("RSI_14", 14, 2),
            LorentzianClassification.Feature("WT_10", 10, 11),
            LorentzianClassification.Feature("CCI_20", 20, 2),
            LorentzianClassification.Feature("ADX_20", 20, 2),
            LorentzianClassification.Feature("RSI_9", 9, 2),
            LorentzianClassification.Feature("MFI_14", 14, 2)
        ],
        settings=LorentzianClassification.Settings(
            source=None,
            neighborsCount=8,
            maxBarsBack=2000,
            useDynamicExits=False
        ),
        filterSettings=LorentzianClassification.FilterSettings(
            useVolatilityFilter=True,
            useRegimeFilter=True,
            useAdxFilter=False,
            regimeThreshold=-0.1,
            adxThreshold=20,
            kernelFilter=LorentzianClassification.KernelFilter(
                useKernelSmoothing=False,
                lookbackWindow=8,
                relativeWeight=8.0,
                regressionLevel=25,
                crossoverLag=2
            )
        )
    )
    return lc

# CA MACD SMA strategy
def ca_macd_sma_strategy(df, short_window=12, long_window=26, signal_window=9, sma_window=200):
    macd_indicator = MACD(close=df['close'], window_slow=long_window, window_fast=short_window)
    df['MACD'] = macd_indicator.macd()
    df['MACD_Signal'] = macd_indicator.macd_signal()
    df['SMA'] = df['close'].rolling(window=sma_window).mean()
    
    df['MACD_SMA_Line'] = np.where(df['MACD'] > df['MACD_Signal'], 1, -1)  # 1 for green, -1 for red
    return df

# V-pattern strategy
def v_pattern_strategy(df, decline_threshold=0.05, recovery_threshold=0.05, window=10):
    df['Pct_Change'] = df['close'].pct_change(window)
    signals = pd.DataFrame(index=df.index)
    signals['V_Buy_Signal'] = 0

    for i in range(window, len(df) - window):
        if df['Pct_Change'].iloc[i] <= -decline_threshold:
            recovery = (df['close'].iloc[i + window] - df['close'].iloc[i]) / df['close'].iloc[i]
            if recovery >= recovery_threshold:
                signals['V_Buy_Signal'].iloc[i + window] = 1
    
    df = df.join(signals)
    return df

# Define the trading strategy
class LorentzianTrader(Strategy):
    def initialize(self, symbol="SPY", cash_at_risk=0.5):
        self.symbol = symbol
        self.sleeptime = "24H"
        self.last_trade = None
        self.cash_at_risk = cash_at_risk
        self.api = REST(base_url=BASE_URL, key_id=API_KEY, secret_key=API_SECRET)

    def position_sizing(self):
        cash = self.get_cash()
        last_price = self.get_last_price(self.symbol)
        quantity = round(cash * self.cash_at_risk / last_price, 0)
        return cash, last_price, quantity

    def get_dates(self):
        today = self.get_datetime()
        three_days_prior = today - Timedelta(days=3)
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    def on_trading_iteration(self):
        cash, last_price, quantity = self.position_sizing()

        df = fetch_data(self.symbol, start_date='2020-01-01', end_date=datetime.now().strftime('%Y-%m-%d'))
        lc = initialize_lc(df)
        df['classification'] = lc.classify()

        df = ca_macd_sma_strategy(df)
        df = v_pattern_strategy(df)

        last_classification = df['classification'].iloc[-1]
        last_macd_sma_line = df['MACD_SMA_Line'].iloc[-1]
        last_v_signal = df['V_Buy_Signal'].iloc[-1]

        if cash > last_price:
            # Condition 1: Long position with green Lorentzian line and green MACD SMA line
            if last_classification == "positive" and last_macd_sma_line == 1:
                if self.last_trade == "sell":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "buy",
                    type="bracket",
                    take_profit_price=last_price * 1.20,
                    stop_loss_price=last_price * 0.95
                )
                self.submit_order(order)
                self.last_trade = "buy"
            # Condition 2: Short position with red Lorentzian line and red MACD SMA line
            elif last_classification == "negative" and last_macd_sma_line == -1:
                if self.last_trade == "buy":
                    self.sell_all()
                order = self.create_order(
                    self.symbol,
                    quantity,
                    "sell",
                    type="bracket",
                    take_profit_price=last_price * 0.80,
                    stop_loss_price=last_price * 1.05
                )
                self.submit_order(order)
                self.last_trade = "sell"

        # Condition 3: Close long position if Lorentzian line turns red
        if self.last_trade == "buy" and last_classification == "negative":
            self.sell_all()
            self.last_trade = None

        # Condition 4: Close short position if Lorentzian line turns green
        if self.last_trade == "sell" and last_classification == "positive":
            self.sell_all()
            self.last_trade = None

        # Condition 5: Follow V-pattern signals
        if self.last_trade == "buy" and last_v_signal == 1:
            self.sell_all()
            self.last_trade = None
        if self.last_trade == "sell" and last_v_signal == 1:
            self.sell_all()
            self.last_trade = None

        # Condition 6: Close long position if MACD SMA line turns red
        if self.last_trade == "buy" and last_macd_sma_line == -1:
            self.sell_all()
            self.last_trade = None

        # Condition 7: Close short position if MACD SMA line turns green
        if self.last_trade == "sell" and last_macd_sma_line == 1:
            self.sell_all()
            self.last_trade = None

# Run backtest or live trading
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 12, 31)
broker = Alpaca(ALPACA_CREDS)
strategy = LorentzianTrader(name='lorentzian_trader', broker=broker, parameters={"symbol": "TSLA", "cash_at_risk": 0.5})

# Uncomment to run backtest
strategy.backtest(YahooDataBacktesting, start_date, end_date, parameters={"symbol": "TSLA", "cash_at_risk": 0.5})

# Uncomment to run live trading
trader = Trader()
trader.add_strategy(strategy)
trader.run_all()
