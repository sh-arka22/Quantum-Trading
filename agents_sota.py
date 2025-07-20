import backtrader as bt
import pandas as pd
import numpy as np

class RSIMeanReversion(bt.Strategy):
    params = dict(rsi_period=14, oversold=30, overbought=70)

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)

    def next(self):
        if not self.position:
            if self.rsi < self.p.oversold:
                self.buy()
            elif self.rsi > self.p.overbought:
                self.sell()
        else:
            if (self.position.size > 0 and self.rsi > 50) or \
               (self.position.size < 0 and self.rsi < 50):
                self.close()

class MACDTrend(bt.Strategy):
    def __init__(self):
        self.macd = bt.indicators.MACD(self.data.close)

    def next(self):
        if not self.position and self.macd.macd > self.macd.signal:
            self.buy()
        elif not self.position and self.macd.macd < self.macd.signal:
            self.sell()
        elif self.position and (self.macd.macd * self.macd.signal < 0):
            self.close()

class BollingerBreak(bt.Strategy):
    params = dict(period=20, dev=2)

    def __init__(self):
        self.bol = bt.indicators.BollingerBands(self.data.close,
                                                period=self.p.period,
                                                devfactor=self.p.dev)

    def next(self):
        if not self.position:
            if self.data.close > self.bol.lines.top:
                self.buy()
            elif self.data.close < self.bol.lines.bot:
                self.sell()
        else:
            if abs(self.data.close - self.bol.lines.mid) < 0.01:
                self.close()

class ATRTrailing(bt.Strategy):
    params = dict(atr_period=14, mult=3)

    def __init__(self):
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.order = None

    def next(self):
        if self.order: return
        if not self.position:
            self.order = self.buy()
        else:
            trail = self.data.close - self.p.mult * self.atr[0]
            if self.data.close[0] < trail:
                self.close()

def run_backtest(df, strategy_class):
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df.set_index('Date'))
    cerebro.adddata(data)
    cerebro.addstrategy(strategy_class)
    cerebro.broker.setcash(100000)
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)
    cerebro.run()
    return cerebro.broker.getvalue() - 100000  # profit
