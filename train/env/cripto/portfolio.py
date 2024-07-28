class Portfolio:
    def __init__(self, asset: float = 0, fiat: float = 0, interest_asset: float = 0, interest_fiat: float = 0,
                 max_position: float = 1):
        self.last_investment = None
        self.asset = asset
        self.fiat = fiat
        self.interest_asset = interest_asset
        self.interest_fiat = interest_fiat
        self.transaction_history = []
        self.is_long = False
        self.is_short = False
        self.realized_pnl = 0
        self.pct_scale = 100  # Se puede ajustar según sea necesario
        self.max_position = max_position
        self.net_inventory_count = 0  # Inicializar el contador de posiciones abiertas

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

    def open_long(self, price, investment, fee, volatility, market_depth):
        self.last_investment = investment
        slippage = self.calculate_slippage(price, investment, volatility, market_depth)
        execution_price = price * (1 + slippage)
        if self.fiat >= investment:
            self.asset += investment / execution_price * (1 - fee)
            self.fiat -= investment
            self.is_long = True
            self.net_inventory_count += 1  # Incrementar el contador de posiciones abiertas
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

    def close_long(self, price, fee, volatility, market_depth):
        slippage = self.calculate_slippage(price,  self.last_investment, volatility, market_depth)
        execution_price = price * (1 + slippage)
        if self.asset > 0:
            fiat_return = self.asset * execution_price * (1 - fee)
            self.realized_pnl += fiat_return - (self.asset * execution_price)  # Actualizar PnL realizado
            self.fiat += fiat_return
            self.is_long = False
            self.net_inventory_count -= 1  # Decrementar el contador de posiciones abiertas
            self.transaction_history.append({
                'action': 'close_long',
                'price': execution_price,
                'fiat_return': fiat_return,
                'fee': fee
            })
            self.asset = 0
        else:
            print("Error: No hay suficientes activos para cerrar una posición larga")

    def open_short(self, price, investment, fee, volatility, market_depth):
        self.last_investment = investment
        slippage = self.calculate_slippage(price, investment, volatility, market_depth)
        execution_price = price * (1 + slippage)
        if self.fiat >= investment:
            self.asset -= investment / execution_price * (1 + fee)
            self.fiat += investment
            self.is_short = True
            self.net_inventory_count += 1  # Incrementar el contador de posiciones abiertas
            self.transaction_history.append({
                'action': 'open_short',
                'price': execution_price,
                'investment_amount': investment,
                'fee': fee
            })
        else:
            print("Error: No hay suficiente fiat para abrir una posición corta")

    def close_short(self, price, fee, volatility, market_depth):
        slippage = self.calculate_slippage(price, self.last_investment, volatility, market_depth)
        execution_price = price * (1 + slippage)
        if self.asset < 0:
            fiat_return = -self.asset * execution_price * (1 - fee)
            self.realized_pnl += self.asset * execution_price + fiat_return  # Actualizar PnL realizado
            self.fiat -= fiat_return
            self.is_short = False
            self.net_inventory_count -= 1  # Decrementar el contador de posiciones abiertas
            self.transaction_history.append({
                'action': 'close_short',
                'price': execution_price,
                'fiat_return': fiat_return,
                'fee': fee
            })
            self.asset = 0
        else:
            print("Error: No hay suficientes activos para cerrar una posición corta")

    def update_interest(self, borrow_interest):
        self.interest_asset = max(0.0, -self.asset) * borrow_interest
        self.interest_fiat = max(0.0, -self.fiat) * borrow_interest

    def get_unrealized_pnl(self, current_price):
        return (self.asset * current_price) - self.realized_pnl

    @staticmethod
    def calculate_slippage(price, amount, volatility, market_depth):
        # Suponer que el deslizamiento aumenta con la volatilidad y el tamaño de la orden
        size_impact = amount / market_depth
        volatility_impact = volatility / price
        slippage = (size_impact + volatility_impact) * 0.05  # Ajustar el coeficiente según sea necesario
        return slippage

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

    def get_scaled_metrics(self, current_price):
        return {
            'net_inventory_ratio': self.net_inventory_count / self.max_position,
            'realized_pnl_scaled': self.realized_pnl * self.pct_scale,
            'unrealized_pnl_scaled': self.get_unrealized_pnl(current_price) * self.pct_scale
        }

    def reset(self, asset, fiat, interest_asset: float = 0, interest_fiat: float = 0, max_position: float = 1):
        self.asset = asset
        self.fiat = fiat
        self.interest_asset = interest_asset
        self.interest_fiat = interest_fiat
        self.transaction_history = []
        self.is_long = False
        self.is_short = False
        self.realized_pnl = 0
        self.pct_scale = 100  # Se puede ajustar según sea necesario
        self.max_position = max_position
        self.net_inventory_count = 0  # Inicializar el contador de posiciones abiertas


if __name__ == "__main__":
    # Ejemplo de uso
    portfolio = Portfolio(asset=0, fiat=1000.0, max_position=10)

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

    # Obtener métricas escaladas
    scaled_metrics = portfolio.get_scaled_metrics(current_price)
    print(scaled_metrics)
