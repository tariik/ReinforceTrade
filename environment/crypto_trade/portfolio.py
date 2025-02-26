from environment.crypto_trade.position import Position


class Portfolio:
    def __init__(self, asset: float = 0, fiat: float = 0,
                 interest_asset: float = 0, interest_fiat: float = 0, buy_fee: float = 0, sell_fee: float = 0):
        self.asset = asset
        self.fiat = fiat

        self.interest_asset = interest_asset
        self.interest_fiat = interest_fiat

        self.transaction_history = []

        self.is_long = False
        self.is_short = False

        self.buy_fee = buy_fee
        self.sell_fee = sell_fee

        self.long_inventory = Position(side='long', max_position=1, buy_fee=self.buy_fee, sell_fee=self.sell_fee)
        self.short_inventory = Position(side='short', max_position=1, buy_fee=self.buy_fee, sell_fee=self.sell_fee)

    def valorisation(self, price):
        return sum([
            self.asset * price,
            self.fiat,
            - self.interest_asset * price,
            - self.interest_fiat
        ])

    def real_position(self, price):
        return (self.asset - self.interest_asset) * price / self.valorisation(price)

    def position(self, price):
        return self.asset * price / self.valorisation(price)

    def open_long(self, price, investment, fee):
        if self.fiat >= investment:
            self.asset += investment / price * (1 - fee)
            self.fiat -= investment
            self.is_long = True
            self.transaction_history.append({
                'action': 'open_long',
                'price': price,
                'investment_amount': investment,
                'fee': fee
            })
            return True
        else:
            print("Error: No hay suficiente fiat para abrir una posición larga")
        return False

    def close_long(self, price, fee):
        if self.asset > 0:
            fiat_return = self.asset * price * (1 - fee)
            self.fiat += fiat_return
            self.is_long = False
            self.transaction_history.append({
                'action': 'close_long',
                'price': price,
                'fiat_return': fiat_return,
                'fee': fee
            })
            self.asset = 0
        else:
            print("Error: No hay suficientes activos para cerrar una posición larga")

    def open_short(self, price, investment, fee):
        if self.fiat >= investment:
            self.asset -= investment / price * (1 + fee)
            self.fiat += investment
            self.is_short = True
            self.transaction_history.append({
                'action': 'open_short',
                'price': price,
                'investment_amount': investment,
                'fee': fee
            })
        else:
            print("Error: No hay suficiente fiat para abrir una posición corta")

    def close_short(self, price, fee):
        if self.asset < 0:
            fiat_return = -self.asset * price * (1 - fee)
            self.fiat -= fiat_return
            self.is_short = False
            self.transaction_history.append({
                'action': 'close_short',
                'price': price,
                'fiat_return': fiat_return,
                'fee': fee
            })
            self.asset = 0
        else:
            print("Error: No hay suficientes activos para cerrar una posición corta")

    def update_interest(self, borrow_interest):
        self.interest_asset = max(0.0, -self.asset) * borrow_interest
        self.interest_fiat = max(0.0, -self.fiat) * borrow_interest

    def __str__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    def describe(self, price):
        print("Value : ", self.valorisation(price), "Position : ", self.position(price))

    def get_portfolio_distribution(self):
        return {
            "asset": max(0.0, self.asset),
            "fiat": max(0.0, self.fiat),
            "borrowed_asset": max(0.0, -self.asset),
            "borrowed_fiat": max(0.0, -self.fiat),
            "interest_asset": self.interest_asset,
            "interest_fiat": self.interest_fiat,
        }

    def get_transaction_history(self):
        return self.transaction_history

    def reset(self) -> None:
        """
        Reset long and short inventories.

        :return: (void)
        """
        self.long_inventory.reset()
        self.short_inventory.reset()


if __name__ == "__main__":
    # Ejemplo de uso
    portfolio = Portfolio(asset=0, fiat=1000.0)

    # Precio actual del activo
    current_price = 50000.0

    # Mostrar la descripción inicial del portafolio
    portfolio.describe(current_price)

    # Abrir una posición larga
    investment_amount = portfolio.fiat * 0.05  # 5% del fiat disponible
    buy_fee = 0.001  # 0.1% de comisión para compra
    sell_fee = 0.0015  # 0.15% de comisión para venta
    portfolio.open_long(current_price, investment_amount, buy_fee)

    # Mostrar la descripción del portafolio después de abrir una posición larga
    portfolio.describe(current_price)

    # Cerrar la posición larga
    portfolio.close_long(current_price, sell_fee)

    # Mostrar la descripción del portafolio después de cerrar la posición larga
    portfolio.describe(current_price)

    # Abrir una posición corta
    portfolio.open_short(current_price, investment_amount, sell_fee)

    # Mostrar la descripción del portafolio después de abrir una posición corta
    portfolio.describe(current_price)

    # Cerrar la posición corta
    portfolio.close_short(current_price, buy_fee)

    # Mostrar la descripción del portafolio después de cerrar la posición corta
    portfolio.describe(current_price)

    # Actualizar intereses basados en una tasa de préstamo del 5%
    borrow_interest_rate = 0.05
    portfolio.update_interest(borrow_interest_rate)

    # Mostrar la descripción final del portafolio
    portfolio.describe(current_price)

    # Obtener la distribución del portafolio
    distribution = portfolio.get_portfolio_distribution()
    print(distribution)

    # Obtener el historial de transacciones
    transaction_history = portfolio.get_transaction_history()
    print(transaction_history)
