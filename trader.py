import random

class Trader:
    def __init__(self, cash, max_stocks, data, model):
        self.cash = cash
        self.max_stocks = max_stocks
        self.stocks = {}
        self.data = data
        self.model = model

    def buy_stock(self, stock, num_shares):
        price = self.data.get_current_price(stock)
        if price * num_shares <= self.cash and len(self.stocks) + 1 <= self.max_stocks:
            self.cash -= price * num_shares
            if stock in self.stocks:
                self.stocks[stock] += num_shares
            else:
                self.stocks[stock] = num_shares
            return True
        return False

    def sell_stock(self, stock, num_shares):
        if stock in self.stocks and self.stocks[stock] >= num_shares:
            price = self.data.get_current_price(stock)
            self.cash += price * num_shares
            self.stocks[stock] -= num_shares
            if self.stocks[stock] == 0:
                del self.stocks[stock]
            return True
        return False

    def evaluate_portfolio(self):
        portfolio_value = self.cash
        for stock, num_shares in self.stocks.items():
            price = self.data.get_current_price(stock)
            portfolio_value += price * num_shares
        return portfolio_value

    def make_trade(self):
        stocks_to_trade = self.data.get_stocks_to_trade()
        if len(stocks_to_trade) == 0:
            return
        stock = random.choice(stocks_to_trade)
        prediction = self.model.predict(stock)
        if prediction == 'Buy':
            self.buy_stock(stock, 1)
        elif prediction == 'Sell':
            if stock in self.stocks:
                self.sell_stock(stock, 1)