import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as s
import random as r
import statistics as st
import os
import time
from datetime import datetime
from datetime import timedelta
from itertools import product
import polars as pl

os.chdir("C:/Users/Andy/Documents/Index_investing_Monte_Carlo_algo")

# Load custom functions
exec(open('functions.py').read())

# Read in data
raw_prices = pd.read_csv("FTSE250.csv")
raw_prices.columns = ["Date", "Open", "High", "Low", "Close"]

# Format the date
raw_prices["Date_ft"] = raw_prices["Date"].apply(lambda x: datetime.strptime(x, "%m/%d/%y"))
raw_prices["Year"] = raw_prices["Date_ft"].apply(lambda x: x.year)
raw_prices["Month"] = raw_prices["Date_ft"].apply(lambda x: x.month)

# Add all-time high
for date in raw_prices["Date_ft"]:
    raw_prices.loc[raw_prices["Date_ft"] == date, "ATH"] = raw_prices[raw_prices["Date_ft"] <= date]["High"].max()

raw_prices["discount"] = raw_prices["Open"] / raw_prices["ATH"] - 1

# Convert to a polars data frame for speed
#raw_prices = pl.DataFrame(raw_prices)

raw_prices[raw_prices["Date_ft"] <= date]["High"].max()

prices_by_month_year = raw_prices.groupby(by = ["Year", "Month"]).mean(numeric_only = True)
prices_by_month_year.head()

#plt.plot(list(prices_by_month_year["Open"]))

raw_prices["Low"].min()
raw_prices["High"].max()


##############################################
# Run algorithm

init_balance = 10000
global_assumed_annual_dividend = 0.03

# =============================================================================
# initial_buy_prices = list(np.linspace(10, 30, 8))
# initial_sell_prices = list(np.linspace(20, 80, 8))
# =============================================================================

# len(initial_buy_prices) * len(initial_sell_prices)

train_data = raw_prices[raw_prices["Year"] > 1995].reset_index(drop = True)
test_data = raw_prices[raw_prices["Year"] >= 2008].reset_index(drop = True)


    
##########################################
# First find a rough idea where alphaable trading rules are

start_time = time.time()

# Vectorised
initial_buy_triggers = list(np.linspace(-0.5, 0, 100))
initial_sell_triggers = list(np.linspace(0.5, 10, 1000))

results = pd.DataFrame(list(product(initial_buy_triggers, 
                                    initial_sell_triggers)),
                       columns = ["Buy", "Sell"])

# Add control to beginning
results.loc[-1] = [0, 100]

results["alpha"], results["trades"] = (
    calculate_alpha_vector(train_data,
                            results["Buy"],
                            results["Sell"],
                            initial_balance = init_balance,
                            assumed_annual_dividend = global_assumed_annual_dividend))

print("--- %s seconds ---" % (time.time() - start_time))

print(results.sort_values("alpha", ascending = False))
print(results.sort_values("alpha", ascending = False)[["Buy", "Sell", "alpha"]])
print(results[results["trades"]>1].sort_values("alpha", ascending = False))

# Search for any clusters of alpha
plt.subplot(1, 2, 1)
plt.scatter(results["Buy"], results["alpha"])
plt.title("Buy")
plt.subplot(1, 2, 2)
plt.scatter(results["Sell"], results["alpha"])
plt.title("Sell")
plt.show()

results["max_alpha"] = results["alpha"].max()
# Work out the range within 50%
best_results = results[results.alpha > 0.2 * results.max_alpha]

print(best_results["Buy"].min())
print(best_results["Buy"].max())
print(best_results["Sell"].min())
print(best_results["Sell"].max())


##########################################

# Test the model

# Benchmark

calculate_alpha_yearly(test_data, [0], [1e6], assumed_annual_dividend = global_assumed_annual_dividend, initial_balance = init_balance)

calculate_alpha_yearly(test_data, [-0.20], [4.5], assumed_annual_dividend = global_assumed_annual_dividend, initial_balance = init_balance)



##########################################

# Test the model with a monte carlo

# =============================================================================
# one_year_monte_carlo = monte_carlo_test_runs(data = raw_prices,
#                                              n_iterations = 1000,
#                                              n_years = 100,
#                                              buy_triggers = [-0.4], 
#                                              sell_triggers = [2],
#                                              initial_balance = init_balance, 
#                                              assumed_annual_dividend = global_assumed_annual_dividend)
# =============================================================================


monte_carlo = fast_monte_carlo_test_runs(
    data = raw_prices,
    n_iterations = 1000,
    min_years = 10,
    buy_trigger = -0.3, 
    sell_trigger = 4.0,
    initial_balance = init_balance, 
    assumed_annual_dividend = global_assumed_annual_dividend)

# Plot the results
monte_carlo["CAGR_alpha"].plot.hist(grid = True, bins = 20)

loser_info(monte_carlo)




test_buy_triggers = list(np.linspace(0, -0.4, 8)) * 2
test_sell_triggers = [2.0, 10.0] * 8

for i in range(1, len(test_buy_triggers) + 1):
    
    monte_carlo = fast_monte_carlo_test_runs(
        data = raw_prices,
        n_iterations = 1000,
        min_years = 10,
        buy_trigger = test_buy_triggers[i], 
        sell_trigger = test_sell_triggers[i],
        initial_balance = init_balance, 
        assumed_annual_dividend = global_assumed_annual_dividend)
    
    # Plot the results
    monte_carlo["CAGR_alpha"].plot.hist(grid = True, bins = 20)
    
    print(f"Buy trigger {test_buy_triggers[i]} and sell trigger {test_sell_triggers[i]} results")
    loser_info(monte_carlo)

