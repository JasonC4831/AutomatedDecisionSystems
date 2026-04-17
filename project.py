import numpy as np
import pandas as pd
from scipy.stats import norm
from yahooquery import Ticker
import datetime

class RiskManagementSystem:
    def __init__(self, portfolio, total_value=100000):
        """
        portfolio: dict mapping ticker to (weight, currency)
        e.g., {'AAPL': (0.4, 'USD'), 'BMW.DE': (0.2, 'EUR'), 'TSLA': (0.4, 'USD')}
        total_value: Total USD value of the portfolio
        """
        self.portfolio = portfolio
        self.total_value = total_value
        self.tickers = list(portfolio.keys())
        self.weights = np.array([val[0] for val in portfolio.values()])
        self.data = None
        self.returns = None

    def fetch_and_adjust_data(self):
        """
        Fetches 2 years of daily data and standardizes everything to USD 
        to account for Foreign Exchange (FX) risk.
        """
        print("Fetching historical data and FX rates...")
        end = datetime.date.today()
        start = end - datetime.timedelta(days=730) # 2 years
        
        # Determine necessary FX pairs (e.g., EURUSD=X)
        currencies = {val[1] for val in self.portfolio.values() if val[1] != 'USD'}
        fx_tickers = [f"{c}USD=X" for c in currencies]
        
        all_tickers = self.tickers + fx_tickers
        t = Ticker(all_tickers)
        hist = t.history(start=start.isoformat(), end=end.isoformat())
        
        # Pivot to get a clean dataframe of Adjusted Close prices
        prices = hist.reset_index().pivot(index='date', columns='symbol', values='adjclose')
        prices = prices.dropna(axis=1, how='all').ffill()

        # Apply FX Conversion dynamically
        for ticker in self.tickers:
            currency = self.portfolio[ticker][1]
            if currency != 'USD':
                fx_pair = f"{currency}USD=X"
                if fx_pair in prices.columns:
                    # Stock Price in EUR * EUR/USD rate = Stock Price in USD
                    prices[ticker] = prices[ticker] * prices[fx_pair]

        self.data = prices[self.tickers]
        self.returns = self.data.pct_change().dropna()
        return self.returns

    def calculate_var(self, confidence_level=0.95, days=5):
        """
        Calculates 5-Day Parametric Value at Risk (VaR) and Conditional VaR (CVaR).
        """
        # 1. Portfolio Volatility (Covariance Matrix)
        cov_matrix = self.returns.cov()
        port_variance = np.dot(self.weights.T, np.dot(cov_matrix, self.weights))
        daily_volatility = np.sqrt(port_variance)
        
        # 2. Time Scaling
        volatility_n_days = daily_volatility * np.sqrt(days)
        
        # 3. VaR Calculation
        z_score = norm.ppf(confidence_level)
        var_pct = z_score * volatility_n_days
        var_usd = self.total_value * var_pct

        # 4. CVaR (Expected Shortfall) Calculation
        cvar_pct = (norm.pdf(z_score) / (1 - confidence_level)) * volatility_n_days
        cvar_usd = self.total_value * cvar_pct

        return {
            '5d_volatility_pct': volatility_n_days * 100,
            '5d_var_usd': var_usd,
            '5d_var_pct': var_pct * 100,
            '5d_cvar_usd': cvar_usd
        }

    def stress_test(self):
        """
        Applies historical and prospective shocks to the portfolio.
        """
        # Scenario 1: Historical - "Black Monday" equivalent (assume -10% market drop)
        # We estimate shock impact by multiplying market drop by portfolio Beta
        port_returns = self.returns.dot(self.weights)
        market_proxy = Ticker('^GSPC').history(period='2y')['adjclose'].pct_change().dropna()
        
        # Align indices to calculate Beta
        aligned = pd.concat([port_returns, market_proxy.xs('^GSPC', level=0)], axis=1).dropna()
        cov = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]
        market_var = np.var(aligned.iloc[:, 1])
        beta = cov / market_var if market_var > 0 else 1.0

        prospective_shock_pct = 0.15 # Scenario: Tech/Market drops 15%
        shock_loss_usd = self.total_value * (prospective_shock_pct * beta)

        return {'beta': beta, 'scenario_15_pct_crash_usd': shock_loss_usd}

    def risk_mitigation(self):
        """
        Identifies the highest risk contributor and suggests a hedge.
        """
        cov_matrix = self.returns.cov()
        port_variance = np.dot(self.weights.T, np.dot(cov_matrix, self.weights))
        
        # Calculate Marginal Contribution to Risk (MCR)
        marginal_contrib = np.dot(cov_matrix, self.weights) / np.sqrt(port_variance)
        component_contrib = self.weights * marginal_contrib
        percentage_contrib = component_contrib / np.sum(component_contrib)

        highest_risk_idx = np.argmax(percentage_contrib)
        culprit = self.tickers[highest_risk_idx]
        contribution = percentage_contrib[highest_risk_idx] * 100

        suggestion = (f"SELL 50% of {culprit} position. It contributes {contribution:.1f}% "
                      f"of your total portfolio variance. Reallocate to TLT (US Treasuries) or GLD (Gold) to dampen beta.")
        return culprit, suggestion

    def generate_heatmap(self):
        """
        Synthesizes metrics into a Red/Yellow/Green Heatmap.
        """
        self.fetch_and_adjust_data()
        metrics = self.calculate_var()
        stress = self.stress_test()
        
        var_pct = metrics['5d_var_pct']
        
        print("\n" + "="*50)
        print("PORTFOLIO RISK HEATMAP & SUMMARY")
        print("="*50)
        
        if var_pct < 3.0:
            print("HEATMAP: GREEN (Low Risk)")
        elif var_pct < 7.0:
            print("HEATMAP: YELLOW (Moderate Risk)")
        else:
            print("HEATMAP: RED (High Risk)")
            
        print(f"\n--- Key Metrics (Total Value: ${self.total_value:,.2f}) ---")
        print(f"5-Day Value at Risk (95%):  ${metrics['5d_var_usd']:,.2f} ({var_pct:.2f}%)")
        print(f"5-Day Expected Shortfall:   ${metrics['5d_cvar_usd']:,.2f} (Worst 5% of cases)")
        print(f"Portfolio Beta vs S&P 500:  {stress['beta']:.2f}")
        print(f"Stress Test (15% Crash):    -${stress['scenario_15_pct_crash_usd']:,.2f}")
        
        if var_pct >= 7.0:
            print("\AUTOMATED RISK MITIGATION")
            culprit, advice = self.risk_mitigation()
            print(advice)
        print("="*50)


if __name__ == "__main__":
    # Test Portfolio: 40% Apple, 40% Tesla, 20% BMW (Priced in Euros)
    my_portfolio = {
        'AAPL': (0.4, 'USD'),
        'TSLA': (0.4, 'USD'),
        'BMW.DE': (0.2, 'EUR') # Adding international FX risk
    }
    
    engine = RiskManagementSystem(portfolio=my_portfolio, total_value=250000)
    engine.generate_heatmap()