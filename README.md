# README

* This project was written in Python and requires the numpy, scipy, and yahooquery libraries.

* Project Description (Including what it does, how the program reaches its decision (suggestion), and how it explains said decision)
    * This Risk Management System is a decision-support tool designed for monitoring and managing stock portfolio risk. It analyzes how market volatility and currency fluctuations can impact a given portfolio and suggests trades that would mitigate risk with a plain English explanation of why it gave that suggestion.
    * The program is a python RiskManagementSystem class which accepts 6 user defined parameters:
        * portfolio: a dict that maps stock tickers to tuples of (weight, currency), where weight is the percentage value of the stock in the portfolio (weights in the portfolio sum up to 1.0), and currency is the currency that the stock is traded in. An example of a portfolio is given below.
        * currencies (optional, default is ['USD']): a list of the user's domestic currencies, i.e. the currencies that the user holds themselves and does not care about the exchange rate of to another domestic currency. currencies[0] stores the base currency of the user, which is used to calculate the FX risk associated with stocks are traded in currencies not listed in the currencies array.
        * total_value (optional, default is 100000): the total nominal value of the portfolio in the base currency. E.g. if currencies[0] = 'EUR' and total_value = 100000, then the portfolio is worth €100,000.
        * VaR_threshold (optional, default is 0.1): The maximum allowable 5-day loss to portfolio value as a decimal (i.e. VaR_threshold=0.1 means the threshold is 10%).
        * small_cap_threshold (optional, default is 0.3): The maximum allowable portion (expressed as a decimal) of stock value in the portfolio that is categorized as small-cap (stocks below $2 billion in market cap).
        * fx_threshold (optional, default is 0.4): The maximum allowable portion (expressed as a decimal) of stock value in the portfolio that is of a single non-domestic (not in the currencies array) currency.
    * The RiskManagementSystem class has 1 main method generate_heatmap(self, print_explanation=True). This method is the backbone of the program. The other 9 methods are helper methods called by generate_heatmap (or by other helper methods) and each accomplishes one of the risk calculation, risk analysis, mitigation decision making, or explainibility portions of the program.
        * generate_heatmap(self, print_explanation=True) outputs a risk state, potential impact in market crashes, and suggested trades (with explanation) for a portfolio. It accomplishes these tasks by calling and outputting the return of helper methods in the following order:
            * fetch_and_adjust_data(self): fetches the daily returns of each stock in the portfolio and the S&P 500 over a time frame of 2 years and stores that as an attribute of the class. Historical prices are taken from yahooquery. The daily returns of a stock include changes in FX rates if the stock is traded in a non-domestic currency.
            * isolate_fx_risk(self): for each stock that is in a non-domestic currency, instead of converting daily returns to the base currency, daily changes in FX rates to the base currency are isolated from local returns (how the stock performed in its home market). The variance of the daily converted returns are then compared to the variance of daily changes in FX rates to get a percentage value for how much of the change in converted daily returns comes from changes in FX rates. This percentage is then stored along with a boolean is_fx_heavy that is true when the percentage is about 20%, and tells whether currency risk is a significant part of the volatility of a stock.
            * perform_scenario_analysis(self): the method calculates the projected losses of the portfolio in 2 market crash scenarios: one simulating the 2020 Covid Crash and another simulating sharp, short term crashes, like Black Tuesday. Potential losses associated with scenarios in which a foreign currency depreciates 20% relative to the base currency are also calculated.
                * stress_test(self): The perform_scenario_analysis method calls stress_test, which calculates a calculates the portfolio beta (volatility compared to the S&P 500 over the last 2 years) using the daily returns data fetched from fetch_and_adjust_data.
            * evaluate_risk_state(self): Calculates the 5-day VaR, the portion of stock value in the portfolio that is categorized as small-cap, and the max portion of stock value in the portfolio that is of a single non-domestic currency. The portfolio's risk state is then returned as a number id and an English alert based on if any of the calculated metrics exceeds their corresponding threshold (e.g. if the calculated VaR exceeds VaR_threshold). If multiple metrics exceed their threshold, then the risk state is determined based on the priority of each risk. Each risk state also corresponds to a heatmap color which is outputed by generate_heatmap. Exceeding VaR_threshold is of highest priority (RED); next is exceeding small_cap_threshold (RED), then exceeding fx_threshold (RED), and finally exceeding half of the VaR_threshold also corresponds to an elevated risk state (RED). If no thresholds are exceeded, the function returns a GREEN heatmap color and "LOW_RISK" risk state id.
                * calculate_var(self): this function is called by evaluate_risk_state to calculate the 5-day VaR. The parametric (variance-covariance) method is used, where the daily portfolio variance is first calculated by considering the volatility of each stock and covariance each stock has to other stocks, then the square root is taken to get the daily portfolio VaR at a 84% level (one standard deviation). That daily portfolio VaR at a 84% level is then multiplied by square root 5 and a Z-score of 1.645 to scale the VaR to be at a timeframe of 5-days and at a 95% level.
                * _get_asset_metadata(self): this function is called by evaluate_risk_state to get data on the market capitalization size of each stock, with stocks that have a market cap of $10 billion or greater being categorized as 'Large', stocks with a market cap of $2 billion to $10 billion being 'Mid', and stocks otherwise being 'Small'. Market cap for companies are converted to USD for standardized categorization.
            * print_explainable_report(self): the print_explanation argument decides whether to call the helper method print_explainable_report(self), which gives an explanation of how the Risk Management System reaches the decision it does. When print_explanation=False, a brief explanation is still given for why the suggested trade is a good idea, but the logic behind how the program suggests its trade is omitted.
            * risk_mitigation(self, rule_id): this function is called when the portfolio is in an elevated risk state (i.e. when the portfolio is not in the green range) and takes a risk state id outputed by evaluate_risk_state. It returns a 'culprit' stock that is most responsible for the portfolio being in the elevated risk state (I.e. the stock that is most responsible for the total volatility of the portfolio, is the small cap stock with the highest weight value, or is the stock in a non-domestic currency with the highest weight, depending on which risk state the portfolio is in). This function also returns a suggestion to buy or sell a stock, bond, or future in order to reduce portfolio risk and includes a brief explanation of why to follow the suggestion. The specific suggestion and explanation is determined by the 'culprit' stock and which risk state the portfolio is in. Examples of strategies that the system suggests can be found below. 
                * _get_asset_metadata(self): this function is also called by risk_mitigation to get data on the market capitalization size of each stock, for use in determining the 'culprit' stock when the risk state comes from the portfolio exceeding the small_cap_threshold.

* How to Run the Program
    * The program is run by creating an RiskManagementSystem object and calling the generate_heatmap method. 3 examples of the program running with outputs are given below:
        * Test Portfolio 1: 25% Apple, 25% Tesla, 20% BMW (Priced in Euros), 15% PBR (in BRL), 15% 7203.T (in JPY), with USD and EUR as domestic currencies
        * portfolio_1 = {
            'AAPL': (0.25, 'USD'),
            'TSLA': (0.25, 'USD'),
            'BMW.DE': (0.2, 'EUR'),
            'PBR': (0.15, 'BRL'),
            '7203.T': (0.15, 'JPY')
          }
          engine_1 = RiskManagementSystem(portfolio=portfolio_1, currencies=['USD', 'EUR'], total_value=250000)
          engine_1.generate_heatmap()
        * Output for Test Portfolio 1:
            *   Fetching historical data and FX rates...

                ============================================================
                RISK METHODOLOGY SUMMARY
                ============================================================
                This program calculates a 5-Day Value at Risk (VaR), taking into account changes to foreign exchange rates for stocks in currencies not held natively.
                VaR estimates the 'worst-case' loss you could expect over the next week with 95% certainty. It looks at how volatile your stocks are
                and how they move together. If your stocks tend to drop at the same time, the VaR increases, signaling a lack of diversification.
                The system compares the calculated VaR against a 10% risk threshold. 
                An alert is issued if your portfolio VaR is beyond or close to this threshold, and suggested trades are given that would mitigate risk.
                Two other thresholds are defined and used in this system: 
                    - small_cap_threshold: the percentage of smaller, volatile firms is capped at 30% before an alert is given.
                    - fx_threshold: the percentage of stocks in foreign currencies you don't hold is limited to 40% before an alert is given.
                Other than comparing risk thresholds, a number of scenarios are also run to give a summary of how your portfolio would fare under potential market crashes 
                or localized economic changes.

                ============================================================
                PORTFOLIO RISK STATE: YELLOW
                ============================================================
                ------------------------------
                ALERT: 5-day VaR is elevated at 5.4%.
                ------------------------------

                --- Scenario Stress Tests ---
                2020 Covid Crash:
                Description: A 30% sudden market deleveraging event.
                Projected Impact: $-77,482.53 (-30.99%)
                ------------------------------
                High Inflation/Rate Hike:
                Description: 15% market drop due to discount rate adjustments.
                Projected Impact: $-38,741.26 (-15.50%)
                ------------------------------
                BRL Depreciation vs USD:
                Description: Simulates a 20% drop in BRL (Unheld).
                Projected Impact: $-7,500.00 (-3.00%)
                ------------------------------
                JPY Depreciation vs USD:
                Description: Simulates a 20% drop in JPY (Unheld).
                Projected Impact: $-7,500.00 (-3.00%)
                ------------------------------

                --- Automated Risk Mitigation ---
                STRATEGY: DIVERSIFICATION. TSLA currently accounts for 53.6% of your total portfolio risk. This is measured via Percentage Contribution to Risk (PCR), 
                which calculates how much of the portfolio's total volatility is owned by a single asset, accounting for its size, individual volatility, and correlation with other holdings. 
                Even if a position's weight is small, a high PCR indicates it is disproportionately driving the portfolio's potential for loss. Reduce position size and rotate funds into 
                low-beta (less volatile) assets like Gold or Treasuries.
                ========================================================================================================================

        * Test Portfolio 2: 50% Apple, 15% Tesla, 20% REPX, 15% SHIP with threshold as 10% to trigger small caps concentration alert
        * portfolio_2 = {
            'AAPL': (0.5, 'USD'),
            'TSLA': (0.15, 'USD'),
            'REPX': (0.2, 'USD'),
            'SHIP': (0.15, 'USD')
          }
          engine_2 = RiskManagementSystem(portfolio=portfolio_2, currencies=['USD'], total_value=250000)
          engine_2.generate_heatmap(print_explanation=False)
        * Output for Test Portfolio 2:
            *   Fetching historical data and FX rates...

                ============================================================
                PORTFOLIO RISK STATE: RED
                ============================================================
                ------------------------------
                ALERT: Small-Cap exposure is 35.0%, exceeding 30% limit.
                ------------------------------

                --- Scenario Stress Tests ---
                2020 Covid Crash:
                Description: A 30% sudden market deleveraging event.
                Projected Impact: $-99,176.07 (-39.67%)
                ------------------------------
                High Inflation/Rate Hike:
                Description: 15% market drop due to discount rate adjustments.
                Projected Impact: $-49,588.03 (-19.84%)
                ------------------------------

                --- Automated Risk Mitigation ---
                STRATEGY: REALLOCATION. Small-cap exposure exceeds the 30% safety threshold. Small-cap stocks are more sensitive to interest rate cycles and lack the capital 
                depth of larger firms, often leading to liquidity traps during market panics. Trim REPX and reallocate funds into TSLA to anchor the portfolio with large-cap stability.
                ========================================================================================================================

        * Test Portfolio 3: 50% Apple, 50% 7203.T (in JPY) with threshold as 10% to trigger foreign currency concentration alert
        * portfolio_3 = {
            'AAPL': (0.5, 'USD'),
            '7203.T': (0.5, 'JPY')
          }
          engine_3 = RiskManagementSystem(portfolio=portfolio_3, currencies=['USD'], total_value=250000)
          engine_3.generate_heatmap(print_explanation=False)
        * Output for Test Portfolio 3:
            *   Fetching historical data and FX rates...

                ============================================================
                PORTFOLIO RISK STATE: RED
                ============================================================
                ------------------------------
                ALERT: Exposure to JPY (50.0%) exceeds 40% limit.
                ------------------------------

                --- Scenario Stress Tests ---
                2020 Covid Crash:
                Description: A 30% sudden market deleveraging event.
                Projected Impact: $-51,978.46 (-20.79%)
                ------------------------------
                High Inflation/Rate Hike:
                Description: 15% market drop due to discount rate adjustments.
                Projected Impact: $-25,989.23 (-10.40%)
                ------------------------------
                JPY Depreciation vs USD:
                Description: Simulates a 20% drop in JPY (Unheld).
                Projected Impact: $-25,000.00 (-10.00%)
                ------------------------------

                --- Automated Risk Mitigation ---
                STRATEGY: CURRENCY HEDGE. High concentration in JPY assets presents translation risk. A drop in the value of the JPY relative to USD will decrease your 
                total returns regardless of stock performance. Use a JPY/USD Forward to hedge 7203.T.
                ========================================================================================================================

* Why the Project is Interesting
    * This project tries to take into consideration the different global perspectives and risk tolerances of users when analyzing portfolio risk. For example, while the Risk Management System defaults to using USD as the base currency when calculating FX risk, the user can define which currencies it does not wish/need to consider foreign exchange rates for, and can even define a different base currency than USD. This opens up opportunities for non-US investors to use this program. To a European investor who defines their base currency as EUR and does not define USD as one of their domestic currencies, stocks traded in USD would have their foreign exchange rates integrated into the calculated 5-day VaR, which may lead to the program reaching a different suggestion than had the user been based in the US and trading in USD. Likewise, the ability for users to define the separate VaR, small cap, and FX thresholds used to determine portfolio risk states reflects the program's philosophy that trading suggestions should be reached based on information it knows about the user. A user who wishes to aim for high returns by trading mostly in small market cap stocks can still use this program without getting alerts that their portfolio is highly concentrated in small cap stocks, simply by defining a higher small_cap_threshold.

* Discussion of AI Usage
    * Google Gemini was used to figure out how to calculate 5-day VaR and beta values from daily returns of stocks in a portfolio, how to isolate foreign exchange risk from the daily returns of a stock in a non-domestic currency, and for generating starting code to calculate these metrics with USD as the base currency. We came up with the logic that there should be thresholds to determine when a portfolio is in a particular risk state and that different risk states should lead to different suggestions. We asked AI to help with the implementation of this logic, including the code that selects at most one top risk to explain and suggest a mitigation strategy to the user. We also asked how to add explainability to the program beyond explaining what Value at Risk is or why a high concentration in small cap stocks leads to risk in a portfolio. This led to us adding a "culprit" stock that contributes most to the portfolio's risk and which is the stock that is suggested to be sold or hedged against.