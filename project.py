import numpy as np
import pandas as pd
from scipy.stats import norm
from yahooquery import Ticker
import datetime
import random

class RiskManagementSystem:
    def __init__(self, portfolio, currencies=['USD'], total_value=100000, threshold=10):
        """
        currencies: list of 'accepted' currencies. 
                    currencies[0] is the Base Currency for conversion.
        """
        self.portfolio = portfolio
        self.currencies = currencies 
        self.base_currency = currencies[0] 
        self.unheld_currencies = [
            val[1] for val in portfolio.values() 
            if val[1] not in self.currencies
        ]
        
        self.total_value = total_value
        self.threshold = threshold
        self.tickers = list(portfolio.keys())
        self.weights = np.array([val[0] for val in portfolio.values()])
        self.data = None
        self.returns = None
        self.fx_analysis = {}

    def fetch_and_adjust_data(self):
        """
        Fetches 2 years of daily data and standardizes everything to USD 
        to account for Foreign Exchange (FX) risk.
        """
        print("Fetching historical data and FX rates...")

        end = datetime.date.today()
        start = end - datetime.timedelta(days=730)
        
        # 1. Identify which stocks need conversion
        # Only convert if the stock currency is NOT in 'domestic' currencies list
        to_convert = {t: self.portfolio[t][1] for t in self.tickers 
                      if self.portfolio[t][1] not in self.currencies}
        
        # 2. Identify required FX pairs to get back to Base Currency
        # e.g., If Base is USD and stock is BRL, we need BRLUSD=X
        fx_tickers = [f"{c}{self.base_currency}=X" for c in set(to_convert.values())]
        
        all_tickers = self.tickers + fx_tickers
        t = Ticker(all_tickers)
        hist = t.history(start=start.isoformat(), end=end.isoformat())
        
        prices = hist.reset_index().pivot(index='date', columns='symbol', values='adjclose')
        prices = prices.dropna(axis=1, how='all').ffill()

        # 3. Apply Conditional FX Conversion
        for ticker in self.tickers:
            asset_curr = self.portfolio[ticker][1]
            
            if asset_curr not in self.currencies:
                # Convert to the first currency in the array
                fx_pair = f"{asset_curr}{self.base_currency}=X"
                if fx_pair in prices.columns:
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
            "High Inflation/Rate Hike": (-0.15, "15% drop due to discount rate adjustments.")
        }
        
        results = {}
        beta = self.stress_test().get('beta', 1.0)
        
        # 1. Market Scenarios
        for name, (shock, desc) in scenarios.items():
            impact = self.total_value * (shock * beta)
            results[name] = {"loss": impact, "description": desc}
        
        # 2. Dynamic FX Scenarios
        fx_shock_magnitude = -0.20 
        for curr in set(self.unheld_currencies):
            curr_weight = sum(val[0] for val in self.portfolio.values() if val[1] == curr)
            fx_impact = self.total_value * curr_weight * fx_shock_magnitude
            
            scenario_name = f"{curr} Depreciation vs {self.base_currency}"
            results[scenario_name] = {
                "loss": fx_impact,
                "description": f"Simulates a {abs(fx_shock_magnitude)*100:.0f}% drop in {curr} (Unheld)."
            }
        
        return results
            
    def evaluate_risk_state(self):
        """ 
        Returns (color, trace, rule_id) for the highest priority risk.
        """
        # candidates: list of (score, message, rule_id)
        candidates = []
        
        metrics = self.calculate_var()
        metadata = self._get_asset_metadata()
        
        # Rule 1: Absolute VaR (Priority 3/1)
        if metrics['5d_var_pct'] > self.threshold:
            candidates.append((3, f"RED: 5-day VaR ({metrics['5d_var_pct']:.1f}%) exceeds threshold.", "VAR_CRITICAL"))
        elif metrics['5d_var_pct'] > self.threshold / 2:
            candidates.append((1, f"YELLOW: 5-day VaR is elevated at {metrics['5d_var_pct']:.1f}%.", "VAR_ELEVATED"))

        # Rule 2: Small-Cap (Priority 2)
        small_cap_weight = sum(self.portfolio[t][0] for t in self.tickers if metadata[t]['cap_size'] == 'Small')
        if small_cap_weight > 0.30:
            candidates.append((2, f"RED: Small-Cap exposure is {small_cap_weight*100:.1f}%, exceeding 30% threshold.", "SMALL_CAP_CONCENTRATION"))

        # Rule 3: Unheld Currency (Priority 2/1)
        for curr in set(self.unheld_currencies):
            curr_weight = sum(val[0] for t, val in self.portfolio.items() if val[1] == curr)
            if curr_weight > 0.40:
                candidates.append((2, f"RED: Extreme exposure to unheld currency {curr} ({curr_weight*100:.1f}%).", f"FX_RED_{curr}"))
            elif curr_weight > 0.20:
                candidates.append((1, f"YELLOW: High exposure to unheld currency {curr} ({curr_weight*100:.1f}%).", f"FX_YELLOW_{curr}"))

        if not candidates:
            return "GREEN", ["GREEN: All metrics acceptable."], "LOW_RISK"

        candidates.sort(key=lambda x: x[0], reverse=True)
        top_score, top_message, rule_id = candidates[0]
        
        color_map = {0: "GREEN", 1: "YELLOW", 2: "RED", 3: "RED"}
        return color_map.get(top_score, "RED"), [top_message], rule_id

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

    def risk_mitigation(self, rule_id):
        """
        Identifies the highest risk contributor and suggests a hedge.
        """
        if rule_id == "LOW_RISK" or not rule_id:
            return None, "No changes required."

        culprit = None
        advice = ""
        metadata = self._get_asset_metadata()

        # 1. VAR RULES: PCR Logic
        if rule_id in ["VAR_CRITICAL", "VAR_ELEVATED"]:
            cov_matrix = self.returns.cov()
            port_variance = np.dot(self.weights.T, np.dot(cov_matrix, self.weights))
            port_vol = np.sqrt(port_variance)
            mcr = np.dot(cov_matrix, self.weights) / port_vol
            pcr = (self.weights * mcr) / port_vol
            
            highest_pcr_idx = np.argmax(pcr)
            culprit = self.tickers[highest_pcr_idx]
            culprit_pcr_pct = pcr[highest_pcr_idx] * 100
            advice = (f"STRATEGY: DIVERSIFICATION. {culprit} dominates {culprit_pcr_pct:.1f}% of "
                      f"your total portfolio risk. Reduce this position and rotate funds into assets with lower correlation such as US Treasuries or Gold to lower VaR.")

        # 2. SMALL-CAP RULE: Now with Random Large-Cap Reallocation
        elif rule_id == "SMALL_CAP_CONCENTRATION":
            # Identify the Small-Cap culprit
            small_caps = {t: self.portfolio[t][0] for t in self.tickers 
                        if metadata[t]['cap_size'] == 'Small'}
            culprit = max(small_caps, key=small_caps.get)
            
            # Find potential Large-Cap targets
            large_caps = [t for t in self.tickers if metadata[t]['cap_size'] == 'Large']
            
            if large_caps:
                target_stock = random.choice(large_caps)
                advice = (f"STRATEGY: REALLOCATION. Portfolio is overly concentrated in small-cap stocks. "
                        f"Small-cap stocks are generally more volatile than large-cap stocks. "
                        f"In the event of a market crash, small cap stocks often lose liquidity first, which could make it difficult to exit positions. "
                        f"Trim positions in the small cap stock {culprit} and reallocate funds into {target_stock} "
                        f"to stabilize the portfolio.")
            else:
                # Fallback if the user somehow has no large caps
                advice = (f"STRATEGY: REALLOCATION. Portfolio is overly concentrated in small-cap stocks. "
                        f"Small-cap stocks are generally more volatile than large-cap stocks. In the event of a market crash, "
                        f"small caps often lose liquidity first, which could make it difficult to exit positions. Trim positions in the small cap stock {culprit}. Since no Large-Caps exist "
                        f"in your current portfolio, consider opening a new position in a large-cap (Blue Chip) stock.")

        # 3. FX RULES
        elif "FX_" in rule_id:
            curr = rule_id.split('_')[-1]
            fx_stocks = {t: self.portfolio[t][0] for t in self.tickers if self.portfolio[t][1] == curr}
            culprit = max(fx_stocks, key=fx_stocks.get)
            advice = (f"STRATEGY: CURRENCY HEDGE. Having a high concentration in {curr} stocks, especially with {curr} not being a held currency, means "
                      f"a localized crisis like a regional war or central bank policy shift could wipe out portfolio gains regardless of how well the individual companies are performing. "
                      f"Hedge {culprit}'s currency risk with a {curr}/USD Forward or Future.")
        
        return culprit, advice

    def generate_heatmap(self):
        """
        Orchestrates the risk report generation.
        """
        # 1. Data and Diagnostics
        self.fetch_and_adjust_data()
        self.isolate_fx_risk()
        
        # 2. Quantitative Calculations
        metrics = self.calculate_var()
        scenarios = self.perform_scenario_analysis()
        
        # 3. Evaluation (Primary Risk Engine)
        color, trace, rule_id = self.evaluate_risk_state()

        # 4. Output Logic
        print("\n" + "="*60)
        print(f"PORTFOLIO RISK STATE: {color}")
        print("="*60)
        
        # Scenario output
        print(f"\n--- Scenario Stress Tests ---")
        for name, data in scenarios.items():
            print(f"{name}: ")
            print(f"  Details: {data['description']}")
            print(f"  Projected Impact: ${data['loss']:,.2f}")
            print("-" * 30)
            
        # 5. Explainability (Trace)
        self.print_explainable_report(color, trace)

        # 6. Mitigation (Passing evaluated parameters)
        if color != "GREEN":
            print("\n--- AUTOMATED RISK MITIGATION ---")
            # PASSING TRACE AND RULE_ID HERE
            culprit, advice = self.risk_mitigation(rule_id)
            print(advice)
            
        print("="*60 + "\n")

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
        print("(c) WHY: Ensures the portfolio remains within user-defined thresholds.")


if __name__ == "__main__":
    # Test Portfolio 1: 25% Apple, 25% Tesla, 20% BMW (Priced in Euros), 15% PBR (in BRL), 15% 7203.T (in JPY), with USD and EUR as domestic currencies
    portfolio_1 = {
        'AAPL': (0.25, 'USD'),
        'TSLA': (0.25, 'USD'),
        'BMW.DE': (0.2, 'EUR'),
        'PBR': (0.15, 'BRL'),
        '7203.T': (0.15, 'JPY')
    }
    engine_1 = RiskManagementSystem(portfolio=portfolio_1, currencies=['USD', 'EUR'], total_value=250000, threshold=10)
    engine_1.generate_heatmap()

    # Test Portfolio 2: 50% Apple, 15% Tesla, 20% REPX, 15% SHIP with threshold as 10% to trigger small caps concentration alert
    portfolio_2 = {
        'AAPL': (0.5, 'USD'),
        'TSLA': (0.15, 'USD'),
        'REPX': (0.2, 'USD'),
        'SHIP': (0.15, 'USD')
    }
    engine_2 = RiskManagementSystem(portfolio=portfolio_2, currencies=['USD'], total_value=250000, threshold=10)
    engine_2.generate_heatmap()

    # Test Portfolio 3: 50% Apple, 50% 7203.T (in JPY) with threshold as 10% to trigger international currency concentration alert
    portfolio_3 = {
        'AAPL': (0.5, 'USD'),
        '7203.T': (0.5, 'JPY')
    }
    engine_3 = RiskManagementSystem(portfolio=portfolio_3, currencies=['USD'], total_value=250000, threshold=10)
    engine_3.generate_heatmap()
