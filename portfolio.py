from typing import Dict, List
from data import StockData
from trader import Trader


class Portfolio:
    def __init__(self, trader: Trader, initial_capital: float):
        self.trader = trader
        self.holdings: Dict[str, int] = {}
        self.capital = initial_capital
        self.profits: List[float] = []

    def buy(self, ticker: str, amount: int) -> bool:
        stock_price = StockData.get_stock_price(ticker)
        cost = stock_price * amount
        if cost > self.capital:
            return False
        else:
            self.capital -= cost
            if ticker in self.holdings:
                self.holdings[ticker] += amount
            else:
                self.holdings[ticker] = amount
            return True

    def sell(self, ticker: str, amount: int) -> bool:
        if ticker not in self.holdings or self.holdings[ticker] < amount:
            return False
        else:
            stock_price = StockData.get_stock_price(ticker)
            revenue = stock_price * amount
            self.capital += revenue
            self.holdings[ticker] -= amount
            if self.holdings[ticker] == 0:
                del self.holdings[ticker]
            return True

    def update(self) -> None:
        holdings_value = 0
        for ticker, amount in self.holdings.items():
            stock_price = StockData.get_stock_price(ticker)
            holdings_value += stock_price * amount
        total_value = holdings_value + self.capital
        self.profits.append(total_value - self.trader.initial_capital)

    def get_holdings(self) -> Dict[str, int]:
        return self.holdings

    def get_capital(self) -> float:
        return self.capital

    def get_profits(self) -> List[float]:
        return self.profits

    def get_total_value(self) -> float:
        holdings_value = 0
        for ticker, amount in self.holdings.items():
            stock_price = StockData.get_stock_price(ticker)
            holdings_value += stock_price * amount
        total_value = holdings_value + self.capital
        return total_value