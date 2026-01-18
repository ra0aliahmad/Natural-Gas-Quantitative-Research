# Natural-Gas-Quantitative-Research
Prototype pricing model for Natural Gas storage contracts developed as part of the JPMorgan Chase Quantitative Research program. This project utilizes time-series analysis, seasonal trend modeling (Sine wave fitting), and operational constraint validation to provide indicative valuations for commodity derivatives.
# Natural Gas Storage Contract Pricing Model

## Project Overview
This project was developed to assist a natural gas trading desk in valuing storage contracts for clients. The model takes historical and snapshot price data to estimate future prices and provides a robust framework for calculating the Net Value of storage agreements.

## Key Features
* **Price Extrapolation:** Uses a combination of Linear Regression (for long-term trends) and Sine Wave fitting (to capture 12-month seasonality) to estimate gas prices for any future date.
* **Contract Valuation Engine:** A generalized function that accounts for multiple injection and withdrawal events.
* **Operational Constraints:** Automatically enforces physical limits such as maximum storage capacity and injection/withdrawal rates.
* **Cost Analysis:** Integrates fixed storage costs, per-unit injection/withdrawal fees, and transportation costs to provide a true net profit estimate.

## Technical Implementation
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scipy, Matplotlib
* **Methodology:** * `Trend Fit`: $y = Ax + B$
    * `Seasonality`: $A \sin(kt + z)$
    * `NPV Calculation`: Sum of cash flows (assuming 0% interest rate per instructions).

## How to Use
1. Ensure `nat_gas_data.csv` is in the root directory.
2. Run `NaturalGas.py` to see the indicative pricing and sample contract analysis.
3. Use the `calculate_contract_value()` function to test custom injection/withdrawal scenarios.

## Disclaimer
This model is a prototype developed for educational purposes and manual oversight during the JPMorgan Chase Quantitative Research Virtual Experience.