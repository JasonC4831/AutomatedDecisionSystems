import numpy as np
import pandas as pd
from scipy.stats import norm
from yahooquery import Ticker
import datetime

class RiskManagementSystem:
    def __init__(self, portfolio, total_value=100000, threshold=10):
        """
        portfolio: dict mapping ticker to (weight, currency)
        e.g., {'AAPL': (0.4, 'USD'), 'BMW.DE': (0.2, 'EUR'), 'TSLA': (0.4, 'USD')}
        total_value: Total USD value of the portfolio
        """
        self.portfolio = portfolio
        self.total_value = total_value
        self.threshold = threshold
        self.tickers = list(portfolio.keys())
        self.weights = np.array([val[0] for val in portfolio.values()])
        self.data = None
        self.returns = None
        self.fx_analysis = None

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
    
    def _get_asset_metadata(self):
        """
        Calculates pure FX volatility vs. Stock volatility.
        """
        metadata = {}
        for ticker in self.tickers:
            # Expert Rule: Use market cap to determine 'Spectrum'
            info = Ticker(ticker).summary_detail.get(ticker, {})
            m_cap = info.get('marketCap', 0)
            
            if m_cap >= 10e9: # > $10B
                cap_size = 'Large'
            elif m_cap >= 2e9: # $2B - $10B
                cap_size = 'Mid'
            else:
                cap_size = 'Small'
            
            metadata[ticker] = {
                'cap_size': cap_size,
                'currency': self.portfolio[ticker][1]
            }
        return metadata

    def isolate_fx_risk(self):
        """
        Measures how much of the portfolio variance comes from 
        Currency fluctuations vs. Asset price fluctuations.
        """
        # Logic: Compare USD-adjusted returns vs. Local-currency returns
        # using same method as fetch_and_adjust_data exept without FX multipliers
        if self.returns is None:
            return
            
        analysis_results = {}
        end = datetime.date.today()
        start = end - datetime.timedelta(days=730)

        for ticker, (weight, currency) in self.portfolio.items():
            if currency == 'USD':
                continue
            
            fx_pair = f"{currency}USD=X"
            # Fetch raw local data to compare against our USD-adjusted data
            t = Ticker([ticker, fx_pair])
            hist = t.history(start=start.isoformat(), end=end.isoformat())
            df = hist.reset_index().pivot(index='date', columns='symbol', values='adjclose').ffill().dropna()
            
            local_rets = df[ticker].pct_change().dropna()
            fx_rets = df[fx_pair].pct_change().dropna()
            usd_rets = (1 + local_rets) * (1 + fx_rets) - 1

            var_usd = usd_rets.var()
            var_fx = fx_rets.var()
            
            analysis_results[ticker] = {
                'currency': currency,
                'fx_contribution_pct': (var_fx / var_usd) * 100 if var_usd > 0 else 0,
                'is_fx_heavy': (var_fx / var_usd) > 0.20 # 20% Threshold used; can be adjusted later to depend on user risk tolerance
            }
        
        self.fx_analysis = analysis_results
        return self.fx_analysis


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
    
    def perform_scenario_analysis(self):
        """
        Applies historical shocks and dynamic FX crash scenarios 
        for every international currency present in the portfolio.
        """
        scenarios = {
            "2020 Covid Crash": (-0.30, "A 30% sudden market deleveraging event."),
            "High Inflation/Rate Hike": (-0.15, "15% market drop due to discount rate adjustments.")
        }
        
        results = {}
        beta = self.stress_test()['beta']
        
        # 1. Standard Market Scenarios
        for name, (shock, desc) in scenarios.items():
            impact = self.total_value * (shock * beta)
            results[name] = {"loss": impact, "description": desc}
        
        # 2. Dynamic FX Crash Scenarios
        # Identify unique international currencies in the portfolio
        unique_currencies = {val[1] for val in self.portfolio.values() if val[1] != 'USD'}
        
        fx_shock_magnitude = -0.20 # Simulating a 20% crash of the local currency vs USD
        
        for curr in unique_currencies:
            # Calculate the total weight of assets exposed to this specific currency
            curr_weight = sum(val[0] for val in self.portfolio.values() if val[1] == curr)
            
            # Impact = Total Value * Weight in Currency * Magnitude of FX Drop
            # Note: This assumes a 'ceteris paribus' scenario where asset local prices stay flat
            fx_impact = self.total_value * curr_weight * fx_shock_magnitude
            
            scenario_name = f"{curr} Currency Crash"
            results[scenario_name] = {
                "loss": fx_impact,
                "description": f"A {abs(fx_shock_magnitude)*100:.0f}% drop in {curr} value against the USD."
            }
        
        return results
    
    def evaluate_risk_state(self):
        """ 
        Uses Rule-Based logic to determine the Heatmap color.
        """
        trace = []
        scores = [] # 0 for Green, 1 for Yellow, 2 for Red
        
        metrics = self.calculate_var()
        metadata = self._get_asset_metadata()
        
        # Rule 1: Absolute VaR Tolerance
        if metrics['5d_var_pct'] > self.threshold:
            trace.append(f"RED: 5-day VaR ({metrics['5d_var_pct']:.1f}%) exceeds max tolerance of 10%.")
            scores.append(2)
        elif metrics['5d_var_pct'] > self.threshold / 2:
            trace.append(f"YELLOW: 5-day VaR is elevated at {metrics['5d_var_pct']:.1f}%.")
            scores.append(1)
            
        # Rule 2: Concentration in Small-Cap / International
        small_cap_weight = sum(w for t, w in zip(self.tickers, self.weights) if metadata[t]['cap_size'] == 'Small')
        if small_cap_weight > 0.30:
            trace.append(f"RED: Small-Cap exposure is {small_cap_weight*100:.1f}%, exceeding 30% limit.")
            scores.append(2)

        # Final Decision Logic
        max_score = max(scores) if scores else 0
        color = {0: "GREEN", 1: "YELLOW", 2: "RED"}[max_score]
        
        return color, trace

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
        
        # Marginal Contribution to Risk
        mcr = np.dot(cov_matrix, self.weights) / np.sqrt(port_variance)
        pct_contrib = (self.weights * mcr) / np.sum(self.weights * mcr)

        highest_risk_idx = np.argmax(pct_contrib)
        culprit = self.tickers[highest_risk_idx]
        contribution = pct_contrib[highest_risk_idx] * 100
        
        # Check the pre-stored fx_analysis attribute
        fx_data = self.fx_analysis.get(culprit, {})
        
        if fx_data.get('is_fx_heavy'):
            curr = fx_data['currency']
            suggestion = (f"ALERT: {culprit} has a high FX risk and contributes to {contribution:.1f}% of your total portfolio variance, driven by {curr} volatility. "
                      f"Hedge using {curr}/USD shorts rather than selling the asset.")
        else:
            suggestion = (f"ALERT: {culprit} has a high asset risk, contributing to {contribution:.1f}% of your total portfolio variance. "
                      f"Reduce position size and reallocate to defensive assets such as TLT (US Treasuries) or GLD (Gold).")
        return culprit, suggestion

    def generate_heatmap(self):
        """
        Now calls all expert system components.
        """
        # 1. Prepare Data
        self.fetch_and_adjust_data()
        self.isolate_fx_risk()
        
        # 2. Run Quantitative Engines
        metrics = self.calculate_var()
        stress_basic = self.stress_test()
        scenarios = self.perform_scenario_analysis() # <--- CALLING SCENARIOS
        
        # 3. Run Inference Engine (Expert Logic)
        color, trace = self.evaluate_risk_state() # <--- CALLING BRAIN
        
        # 4. Visual Output
        print("\n" + "="*50)
        print(f"PORTFOLIO RISK STATE: {color}")
        print("="*50)
        
        print(f"\n--- Scenario Stress Tests ---")
        for name, data in scenarios.items():
            print(f"{name}: ")
            print(f"  Details: {data['description']}")
            print(f"  Projected Impact: ${data['loss']:,.2f}")
            print("-" * 30)
            
        # 5. Explainability Trace
        self.print_explainable_report(color, trace) # <--- CALLING EXPLAINER
        
        # 6. Mitigation (if risky)
        if color == "RED" or color == "YELLOW":
            print("\n--- AUTOMATED RISK MITIGATION ---")
            culprit, advice = self.risk_mitigation()
            print(advice)
        print("="*50)

    def print_explainable_report(self, color, trace):
        """
        Outputs the 'Trace' for human auditors.
        """
        print(f"\nFINAL SYSTEM STATE: {color}")
        print("-" * 30)
        for entry in trace:
            print(f"TRACER: {entry}")
            
        print("\nEXPLAINABILITY NOTE:")
        print("(a) WHAT: This system uses a Hybrid Rule-Based model.")
        print("(b) HOW: Rules check VaR (Parametric) and Concentration against thresholds.")
        print("(c) WHY: Ensures the portfolio remains within user-defined mandates.")


if __name__ == "__main__":
    # Test Portfolio: 25% Apple, 25% Tesla, 20% BMW (Priced in Euros), 15% PBR (in BRL), 15% 7203.T (in JPY)
    my_portfolio = {
        'AAPL': (0.25, 'USD'),
        'TSLA': (0.25, 'USD'),
        'BMW.DE': (0.2, 'EUR'),
        'PBR': (0.15, 'BRL'),
        '7203.T': (0.15, 'JPY')
    }
    
    engine = RiskManagementSystem(portfolio=my_portfolio, total_value=250000, threshold=10)
    engine.generate_heatmap()