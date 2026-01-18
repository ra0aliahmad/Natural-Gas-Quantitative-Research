import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from datetime import datetime

class GasContractPricer:
    def __init__(self, data_csv):
        # Load and prepare data
        self.df = pd.read_csv(data_csv)
        self.df['Dates'] = pd.to_datetime(self.df['Dates'])
        self.df = self.df.sort_values('Dates')
        
        # Convert dates to numerical 'days' for interpolation
        self.start_date = self.df['Dates'].min()
        self.df['Days'] = (self.df['Dates'] - self.start_date).dt.days
        
        # Create Cubic Spline Interpolation (smooth curves for seasonality)
        self.price_model = interp1d(
            self.df['Days'], 
            self.df['Prices'], 
            kind='cubic', 
            fill_value="extrapolate"
        )

    def get_price(self, date_str):
        """Estimate the price for any given date."""
        target_date = pd.to_datetime(date_str)
        days = (target_date - self.start_date).days
        return float(self.price_model(days))

    def calculate_contract_value(self, injection_date, withdrawal_date, quantity, 
                                 storage_fee_per_month, injection_withdrawal_cost):
        """
        Calculates the Net Value of a storage contract.
        """
        # 1. Get prices from our extrapolated curve
        p_in = self.get_price(injection_date)
        p_out = self.get_price(withdrawal_date)
        
        # 2. Calculate time-based costs
        d1 = pd.to_datetime(injection_date)
        d2 = pd.to_datetime(withdrawal_date)
        months_in_storage = (d2 - d1).days / 30.44
        
        # 3. Formula: (Sell Price - Buy Price - Ops Costs) * Quantity - Monthly Fees
        spread = p_out - p_in
        total_ops_cost = injection_withdrawal_cost * 2 # Cost to put in AND take out
        storage_rent = storage_fee_per_month * months_in_storage
        
        net_profit = (spread - total_ops_cost) * quantity - (storage_rent * quantity)
        
        return {
            "Injection Price": round(p_in, 2),
            "Withdrawal Price": round(p_out, 2),
            "Spread": round(spread, 2),
            "Net Contract Value": round(net_profit, 2)
        }

# --- EXECUTION ---

# 1. Save your data to a local file
# (Assuming 'nat_gas_data.csv' exists with the Dates,Prices provided)

pricer = GasContractPricer('nat_gas_data.csv')

# 2. Example Trade:
# Buy 100,000 MMBtu in Summer (June 2022), Sell in Winter (January 2023)
# Storage fee: $0.02/MMBtu per month
# Injection/Withdrawal cost: $0.05/MMBtu
result = pricer.calculate_contract_value(
    injection_date='2022-06-15', 
    withdrawal_date='2023-01-15', 
    quantity=100000,
    storage_fee_per_month=0.02,
    injection_withdrawal_cost=0.05
)

print(f"Contract Analysis for Alex:")
for key, value in result.items():
    print(f"{key}: {value}")
    import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import date, timedelta

# 1. Load the data
# Ensure 'natgas_R.csv' is in your "D:\Quatitative Research" folder
df = pd.read_csv('nat_gas_data.csv', parse_dates=['Dates'])
prices = df['Prices'].values
dates = df['Dates'].values

# 2. Setup Time Variables
start_date = date(2020, 10, 31)
days_from_start = np.array([(d.to_pydatetime().date() - start_date).days for d in df['Dates']])

# 3. Simple Linear Regression (Trend: y = Ax + B)
def simple_regression(x, y):
    xbar, ybar = np.mean(x), np.mean(y)
    slope = np.sum((x - xbar) * (y - ybar)) / np.sum((x - xbar)**2)
    intercept = ybar - slope * xbar
    return slope, intercept

slope, intercept = simple_regression(days_from_start, prices)

# 4. Seasonal Fit (Sine Wave)
# Subtract the trend to find the seasonal residuals
sin_prices = prices - (days_from_start * slope + intercept)
sin_time = np.sin(days_from_start * 2 * np.pi / 365)
cos_time = np.cos(days_from_start * 2 * np.pi / 365)

def bilinear_regression(y, x1, x2):
    slope1 = np.sum(y * x1) / np.sum(x1 ** 2)
    slope2 = np.sum(y * x2) / np.sum(x2 ** 2)
    return slope1, slope2

slope1, slope2 = bilinear_regression(sin_prices, sin_time, cos_time)
amplitude = np.sqrt(slope1 ** 2 + slope2 ** 2)
shift = np.arctan2(slope2, slope1)

# 5. The Interpolation/Extrapolation Function
def get_gas_price(input_date):
    """Returns an indicative price for any date."""
    target_date = pd.to_datetime(input_date).date()
    days = (target_date - start_date).days
    
    # Formula: Trend + Seasonal Variation
    prediction = (days * slope + intercept) + (amplitude * np.sin(days * 2 * np.pi / 365 + shift))
    return round(prediction, 2)

# --- TASK OUTPUTS ---
# Estimate a price for the extrapolation period (e.g., Summer 2025)
test_date = "2025-07-15"
print(f"Indicative Price for {test_date}: ${get_gas_price(test_date)}")

# Visualize
plt.figure(figsize=(10, 5))
plt.plot(dates, prices, 'o', label='Actual Monthly Data')
continuous_dates = pd.date_range(start='2020-10-31', end='2025-09-30', freq='D')
plt.plot(continuous_dates, [get_gas_price(d) for d in continuous_dates], label='Extrapolated Model')
plt.title('Natural Gas Price: Trend + Seasonality')
plt.legend()
plt.show()
import pandas as pd
import numpy as np
from datetime import date

class GasStoragePricer:
    def __init__(self, csv_file):
        # 1. Load data and setup the regression/seasonality model
        df = pd.read_csv(csv_file, parse_dates=['Dates'])
        self.prices = df['Prices'].values
        self.start_date = date(2020, 10, 31)
        self.days_from_start = np.array([(d.to_pydatetime().date() - self.start_date).days for d in df['Dates']])
        
        # Fit Linear Trend (y = slope * x + intercept)
        xbar, ybar = np.mean(self.days_from_start), np.mean(self.prices)
        self.slope = np.sum((self.days_from_start - xbar) * (self.prices - ybar)) / np.sum((self.days_from_start - xbar)**2)
        self.intercept = ybar - self.slope * xbar
        
        # Fit Seasonality (Sine wave for the summer-winter cycle)
        sin_prices = self.prices - (self.days_from_start * self.slope + self.intercept)
        sin_t, cos_t = np.sin(self.days_from_start * 2 * np.pi / 365), np.cos(self.days_from_start * 2 * np.pi / 365)
        s1 = np.sum(sin_prices * sin_t) / np.sum(sin_t ** 2)
        s2 = np.sum(sin_prices * cos_t) / np.sum(cos_t ** 2)
        self.amplitude, self.shift = np.sqrt(s1**2 + s2**2), np.arctan2(s2, s1)

    def get_price_on_date(self, target_date_str):
        """Predicts the indicative price for any date."""
        days = (pd.to_datetime(target_date_str).date() - self.start_date).days
        return (days * self.slope + self.intercept) + (self.amplitude * np.sin(days * 2 * np.pi / 365 + self.shift))

    def calculate_contract_value(self, injection_dates, withdrawal_dates, 
                                 injection_volume, withdrawal_volume, 
                                 max_storage, storage_cost_per_month, fee_per_unit):
        """
        Prices the contract based on cash flows and physical constraints.
        """
        total_value = 0
        current_inventory = 0
        
        # Injection Cash Flows
        for i_date, vol in zip(injection_dates, injection_volume):
            if current_inventory + vol > max_storage:
                return "Error: Exceeds max storage capacity."
            price = self.get_price_on_date(i_date)
            total_value -= vol * (price + fee_per_unit) # Cash Out
            current_inventory += vol
            
        # Withdrawal Cash Flows
        for w_date, vol in zip(withdrawal_dates, withdrawal_volume):
            if current_inventory - vol < 0:
                return "Error: Insufficient gas in storage."
            price = self.get_price_on_date(w_date)
            total_value += vol * (price - fee_per_unit) # Cash In
            current_inventory -= vol
            
        # Fixed Storage Costs
        start, end = pd.to_datetime(min(injection_dates)), pd.to_datetime(max(withdrawal_dates))
        months = (end - start).days / 30.44
        total_value -= (months * storage_cost_per_month)
        
        return round(total_value, 2)

# --- TESTING ---
pricer = GasStoragePricer('nat_gas_data.csv')

# Sample: Buy 1M MMBtu in Summer 2023, Sell in Winter 2024
result = pricer.calculate_contract_value(
    injection_dates=['2023-06-01', '2023-07-01'],
    withdrawal_dates=['2024-01-01', '2024-02-01'],
    injection_volume=[500000, 500000],
    withdrawal_volume=[500000, 500000],
    max_storage=1500000,
    storage_cost_per_month=100000,
    fee_per_unit=0.01
)
print(f"Prototype Valuation: ${result:,.2f}")