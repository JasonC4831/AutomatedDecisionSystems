import numpy as np
import pandas as pd
from scipy.stats import norm
from yahooquery import Ticker
import datetime
import random

class RiskManagementSystem:
    def __init__(self, portfolio, currencies=['USD'], total_value=100000, 
                 VaR_threshold=0.10, small_cap_threshold=0.30, fx_threshold=0.40):
        """
        portfolio: dict mapping ticker to (weight, currency)
        total_value: Total USD value of the portfolio
        VaR_threshold: 5-Day VaR decimal threshold (e.g., 0.10 for 10%)
        small_cap_threshold: Limit for small-cap concentration (default 0.30)
        fx_threshold: Limit for unheld currency concentration (default 0.40)
        """
        self.portfolio = portfolio
        self.currencies = currencies 
        self.base_currency = currencies[0] 
        self.total_value = total_value
        
        # Renamed and standardized thresholds
        self.VaR_threshold = VaR_threshold 
        self.small_cap_threshold = small_cap_threshold
        self.fx_threshold = fx_threshold
        
        self.unheld_currencies = [
            val[1] for val in portfolio.values() 
            if val[1] not in self.currencies
        ]
        
        self.tickers = list(portfolio.keys())
        self.weights = np.array([val[0] for val in portfolio.values()])
        self.data = None
        self.returns = None
        self._metadata_cache = None

    def fetch_and_adjust_data(self):
        """
        Fetches 2 years of daily data and standardizes everything to USD 
        to account for Foreign Exchange (FX) risk.
        """
        print("Fetching historical data and FX rates...")
        end = datetime.date.today()
        start = end - datetime.timedelta(days=730)
        
        to_convert = {t: self.portfolio[t][1] for t in self.tickers 
                      if self.portfolio[t][1] not in self.currencies}
        
        fx_tickers = [f"{c}{self.base_currency}=X" for c in set(to_convert.values())]
        all_tickers = self.tickers + fx_tickers + ['^GSPC'] # Added Market Proxy here
        
        t = Ticker(all_tickers)
        hist = t.history(start=start.isoformat(), end=end.isoformat())
        
        prices = hist.reset_index().pivot(index='date', columns='symbol', values='adjclose')
        prices = prices.ffill().dropna(axis=0, how='all')

        # Apply FX Conversion
        for ticker in self.tickers:
            asset_curr = self.portfolio[ticker][1]
            if asset_curr not in self.currencies:
                fx_pair = f"{asset_curr}{self.base_currency}=X"
                if fx_pair in prices.columns:
                    prices[ticker] = prices[ticker] * prices[fx_pair]

        self.data = prices[self.tickers]
        self.market_data = prices['^GSPC'] # Cached for stress_test
        self.returns = self.data.pct_change().dropna()
        return self.returns
    
    def _get_asset_metadata(self):
        """
        Fetches metadata for stocks once and caches it.
        """
        if self._metadata_cache is not None:
            return self._metadata_cache

        metadata = {}
        # Fetch all tickers in one batch call to be efficient
        t_info = Ticker(self.tickers).summary_detail
        
        for ticker in self.tickers:
            # Handle potential dictionary nesting from yahooquery
            info = t_info.get(ticker, {}) if isinstance(t_info, dict) else {}
            m_cap = info.get('marketCap', 0) if isinstance(info, dict) else 0
            
            if m_cap >= 10e9:
                cap_size = 'Large'
            elif m_cap >= 2e9:
                cap_size = 'Mid'
            else:
                cap_size = 'Small'
            
            metadata[ticker] = {
                'cap_size': cap_size,
                'currency': self.portfolio[ticker][1]
            }
        
        self._metadata_cache = metadata
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
        including both USD and Percentage loss impacts.
        """
        scenarios = {
            "2020 Covid Crash": (-0.30, "A 30% sudden market deleveraging event."),
            "High Inflation/Rate Hike": (-0.15, "15% market drop due to discount rate adjustments.")
        }
        
        results = {}
        beta = self.stress_test().get('beta', 1.0)
        
        # 1. Market Scenarios
        for name, (shock, desc) in scenarios.items():
            # Percentage impact on the total portfolio
            total_impact_pct = shock * beta 
            impact_usd = self.total_value * total_impact_pct
            
            results[name] = {
                "loss": impact_usd, 
                "loss_pct": total_impact_pct * 100,
                "description": desc
            }
        
        # 2. Dynamic FX Scenarios
        fx_shock_magnitude = -0.20 
        for curr in set(self.unheld_currencies):
            curr_weight = sum(val[0] for val in self.portfolio.values() if val[1] == curr)
            
            # Percentage impact on the total portfolio
            fx_impact_pct = curr_weight * fx_shock_magnitude
            fx_impact_usd = self.total_value * fx_impact_pct
            
            scenario_name = f"{curr} Depreciation vs {self.base_currency}"
            results[scenario_name] = {
                "loss": fx_impact_usd,
                "loss_pct": fx_impact_pct * 100,
                "description": f"Simulates a {abs(fx_shock_magnitude)*100:.0f}% drop in {curr} (Unheld)."
            }
        
        return results
            
    def evaluate_risk_state(self):
        """ 
        Returns (color, trace, rule_id) for the highest priority risk.
        """
        candidates = []
        metrics = self.calculate_var() # Returns pct (e.g., 12.0)
        metadata = self._get_asset_metadata()
        
        # Convert decimal threshold to percentage for metric comparison
        var_limit_pct = self.VaR_threshold * 100
        
        # Rule 1: Absolute VaR using the new parameter name
        if metrics['5d_var_pct'] > var_limit_pct:
            candidates.append((3, f"ALERT: 5-day VaR ({metrics['5d_var_pct']:.1f}%) exceeds threshold ({var_limit_pct:.1f}%).", "VAR_CRITICAL"))
        elif metrics['5d_var_pct'] > (var_limit_pct / 2):
            candidates.append((1, f"ALERT: 5-day VaR is elevated at {metrics['5d_var_pct']:.1f}%.", "VAR_ELEVATED"))

        # Rule 2: Small-Cap Concentration
        small_cap_weight = sum(self.portfolio[t][0] for t in self.tickers if metadata[t]['cap_size'] == 'Small')
        if small_cap_weight > self.small_cap_threshold:
            candidates.append((2, f"ALERT: Small-Cap exposure is {small_cap_weight*100:.1f}%, exceeding {self.small_cap_threshold*100:.0f}% limit.", "SMALL_CAP_CONCENTRATION"))

        # Rule 3: FX Concentration
        for curr in set(self.unheld_currencies):
            curr_weight = sum(val[0] for t, val in self.portfolio.items() if val[1] == curr)
            if curr_weight > self.fx_threshold:
                candidates.append((2, f"ALERT: Exposure to {curr} ({curr_weight*100:.1f}%) exceeds {self.fx_threshold*100:.0f}% limit.", f"FX_RED_{curr}"))

        if not candidates:
            return "GREEN", ["GREEN: All metrics acceptable."], "LOW_RISK"

        candidates.sort(key=lambda x: x[0], reverse=True)
        top_score, top_message, rule_id = candidates[0]
        
        color_map = {1: "YELLOW", 2: "RED", 3: "RED"}
        return color_map.get(top_score, "RED"), [top_message], rule_id

    def stress_test(self):
        """
        Applies historical and prospective shocks to the portfolio.
        """
        port_returns = self.returns.dot(self.weights)
        market_rets = self.market_data.pct_change().reindex(port_returns.index).fillna(0)
        
        # Calculate Beta correctly
        covariance = np.cov(port_returns, market_rets)[0, 1]
        market_variance = np.var(market_rets)
        beta = covariance / market_variance if market_variance > 0 else 1.0

        prospective_shock_pct = 0.15 
        shock_loss_usd = self.total_value * (prospective_shock_pct * beta)

        return {'beta': beta, 'scenario_15_pct_crash_usd': shock_loss_usd}

    def risk_mitigation(self, rule_id):
        """
        Identifies the highest risk contributor and provides a detailed strategy
        including an explanation of Percentage Contribution to Risk (PCR).
        """
        if rule_id == "LOW_RISK" or not rule_id:
            return None, "No changes required. Portfolio remains within risk thresholds."

        culprit = None
        advice = ""
        metadata = self._get_asset_metadata()

        # 1. VAR RULES: PCR Logic and Explanation
        if rule_id in ["VAR_CRITICAL", "VAR_ELEVATED"]:
            # Recalculate PCR
            cov_matrix = self.returns.cov()
            port_variance = np.dot(self.weights.T, np.dot(cov_matrix, self.weights))
            port_vol = np.sqrt(port_variance)
            mcr = np.dot(cov_matrix, self.weights) / port_vol
            pcr = (self.weights * mcr) / port_vol
            
            highest_pcr_idx = np.argmax(pcr)
            culprit = self.tickers[highest_pcr_idx]
            culprit_pcr_pct = pcr[highest_pcr_idx] * 100
            
            advice = (f"STRATEGY: DIVERSIFICATION. {culprit} currently accounts for {culprit_pcr_pct:.1f}% of "
                      f"your total portfolio risk. This is measured via Percentage Contribution to Risk (PCR), \n"
                      f"which calculates how much of the portfolio's total volatility is owned by a single asset, "
                      f"accounting for its size, individual volatility, and correlation with other holdings. \n"
                      f"Reduce this position size and rotate funds "
                      f"into low-beta assets like US Treasuries or Gold to rebalance the risk distribution.")

        # 2. SMALL-CAP RULE (Explanation remains the same)
        elif rule_id == "SMALL_CAP_CONCENTRATION":
            small_caps = {t: self.portfolio[t][0] for t in self.tickers 
                          if metadata[t]['cap_size'] == 'Small'}
            culprit = max(small_caps, key=small_caps.get)
            large_caps = [t for t in self.tickers if metadata[t]['cap_size'] == 'Large']
            
            header = (f"STRATEGY: REALLOCATION. Small-cap exposure exceeds the 30% safety threshold. "
                      f"Small-cap stocks are more sensitive to interest rate cycles and lack the capital \n"
                      f"depth of larger firms, often leading to liquidity traps during market panics.")

            if large_caps:
                target_stock = random.choice(large_caps)
                advice = (f"{header} Trim {culprit} and reallocate funds into {target_stock} to anchor "
                          f"the portfolio with large-cap stability.")
            else:
                advice = (f"{header} Trim {culprit}. As your portfolio currently lacks Large-Cap assets, "
                          f"consider initiating a position in a Blue Chip stock.")

        # 3. FX RULES (Explanation remains the same)
        elif "FX_" in rule_id:
            curr = rule_id.split('_')[-1]
            fx_stocks = {t: self.portfolio[t][0] for t in self.tickers if self.portfolio[t][1] == curr}
            culprit = max(fx_stocks, key=fx_stocks.get)
            
            advice = (f"STRATEGY: CURRENCY HEDGE. High concentration in {curr} assets presents translation risk. "
                      f"A drop in the value of the {curr} relative to {self.base_currency} will decrease your \n"
                      f"total returns regardless of stock performance. Use a {curr}/USD Forward to hedge {culprit}.")
        
        return culprit, advice

    def generate_heatmap(self, print_explanation=True):
        """
        Orchestrates the risk report generation with percentage-based stress testing.
        """
        # 1. Data and Diagnostics
        self.fetch_and_adjust_data()
        self.isolate_fx_risk()
        
        # 2. Quantitative Calculations
        metrics = self.calculate_var()
        scenarios = self.perform_scenario_analysis()
        
        # 3. Evaluation (Primary Risk Engine)
        color, trace, rule_id = self.evaluate_risk_state()

         # 4. Explainability
        if print_explanation:
            self.print_explainable_report()

        # 5. Output Logic
        print("\n" + "="*60)
        print(f"PORTFOLIO RISK STATE: {color}")
        print("="*60)

        print("-" * 30)
        for entry in trace:
            print(f"{entry}")
        print("-" * 30)

        # Scenario output
        print(f"\n--- Scenario Stress Tests ---")
        for name, data in scenarios.items():
            print(f"{name}:")
            print(f"  Description: {data['description']}")
            # OUTPUTTING PERCENTAGE LOSS HERE
            print(f"  Projected Impact: ${data['loss']:,.2f} ({data['loss_pct']:.2f}%)")
            print("-" * 30)

        # 6. Mitigation
        if color != "GREEN":
            print("\n--- Automated Risk Mitigation ---")
            culprit, advice = self.risk_mitigation(rule_id)
            print(advice)
            
        print("="*120 + "\n")

    def print_explainable_report(self):
        """
        Provides a clear summary of the VaR calculation and an overview
        of the other risk thresholds evaluated by the system.
        """
        print(f"\n" + "="*60)
        print("RISK METHODOLOGY SUMMARY")
        print("="*60)

        # 1. Simple VaR Summary
        print("This program calculates a 5-Day Value at Risk (VaR), taking into account changes to foreign exchange rates for stocks in currencies not held natively.")
        print("VaR estimates the 'worst-case' loss you could expect over the next week with 95% certainty. It looks at how volatile your stocks are")
        print("and how they move together. If your stocks tend to drop at the same time, the VaR increases, signaling a lack of diversification.")

        # 2. Overview of other checks
        print(f"The system compares the calculated VaR against a {self.VaR_threshold*100:.0f}% risk threshold. ")
        print("An alert is issued if your portfolio VaR is beyond or close to this threshold, and suggested trades are given that would mitigate risk.")
        print("Two other thresholds are defined and used in this system: ")
        print(f"    • small_cap_threshold: the percentage of smaller, volatile firms is capped at {self.small_cap_threshold*100:.0f}% before an alert is given.")
        print(f"    • fx_threshold: the percentage of stocks in foreign currencies you don't hold is limited to {self.fx_threshold*100:.0f}% before an alert is given.")
        print("Other than comparing risk thresholds, a number of scenarios are also run to give a summary of how your portfolio would fare under potential market crashes \nor localized economic changes.")


if __name__ == "__main__":
    # Test Portfolio 1: 25% Apple, 25% Tesla, 20% BMW (Priced in Euros), 15% PBR (in BRL), 15% 7203.T (in JPY), with USD and EUR as domestic currencies
    portfolio_1 = {
        'AAPL': (0.25, 'USD'),
        'TSLA': (0.25, 'USD'),
        'BMW.DE': (0.2, 'EUR'),
        'PBR': (0.15, 'BRL'),
        '7203.T': (0.15, 'JPY')
    }
    engine_1 = RiskManagementSystem(portfolio=portfolio_1, currencies=['USD', 'EUR'], total_value=250000)
    engine_1.generate_heatmap()

    # Test Portfolio 2: 50% Apple, 15% Tesla, 20% REPX, 15% SHIP with threshold as 10% to trigger small caps concentration alert
    portfolio_2 = {
        'AAPL': (0.5, 'USD'),
        'TSLA': (0.15, 'USD'),
        'REPX': (0.2, 'USD'),
        'SHIP': (0.15, 'USD')
    }
    engine_2 = RiskManagementSystem(portfolio=portfolio_2, currencies=['USD'], total_value=250000)
    engine_2.generate_heatmap(print_explanation=False)

    # Test Portfolio 3: 50% Apple, 50% 7203.T (in JPY) with threshold as 10% to trigger international currency concentration alert
    portfolio_3 = {
        'AAPL': (0.5, 'USD'),
        '7203.T': (0.5, 'JPY')
    }
    engine_3 = RiskManagementSystem(portfolio=portfolio_3, currencies=['USD'], total_value=250000)
    engine_3.generate_heatmap(print_explanation=False)
