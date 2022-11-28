import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as s
import random as r
import statistics as st
import os
import time
from datetime import datetime
from itertools import product
import polars as pl

os.chdir("C:/Users/Andy/Documents/Index_trading")

# Load custom functions
execfile('functions.py')

# Read in data
raw_prices = pd.read_csv("FTSE250.csv")
raw_prices.columns = ["Date", "Open", "High", "Low", "Close"]

# Format the date
raw_prices["Date_ft"] = raw_prices["Date"].apply(lambda x: datetime.strptime(x, "%m/%d/%y"))
raw_prices["Year"] = raw_prices["Date_ft"].apply(lambda x: x.year)
raw_prices["Month"] = raw_prices["Date_ft"].apply(lambda x: x.month)

# Convert to a polars data frame for speed
#raw_prices = pl.DataFrame(raw_prices)

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

train_data = raw_prices[raw_prices["Year"] < 2007].reset_index(drop = True)
test_data = raw_prices[raw_prices["Year"] >= 2007].reset_index(drop = True)


    
##########################################
# First find a rough idea where profitable trading rules are

start_time = time.time()

# Vectorised
initial_buy_triggers = list(np.linspace(-0.5, 0, 100))
initial_sell_triggers = list(np.linspace(0.5, 10, 100))

results = pd.DataFrame(list(product(initial_buy_triggers, 
                                    initial_sell_triggers)),
                       columns = ["Buy", "Sell"])

# Add control to beginning
results.loc[-1] = [0, 1e6]

results["profit"], results["trades"] = (
    calculate_profit_vector(train_data,
                            results["Buy"],
                            results["Sell"],
                            initial_balance = init_balance,
                            assumed_annual_dividend = global_assumed_annual_dividend))

print("--- %s seconds ---" % (time.time() - start_time))

print(results.sort_values("profit", ascending = False))
print(results.sort_values("profit", ascending = False)[["Buy", "Sell", "Stop", "profit"]])
print(results[results["trades_lost"]<1000].sort_values("profit", ascending = False))

# Search for any clusters of profit
plt.subplot(1, 3, 1)
plt.scatter(results["Buy"], results["profit"])
plt.title("Buy")
plt.subplot(1, 3, 2)
plt.scatter(results["Sell"], results["profit"])
plt.title("Sell")
plt.subplot(1, 3, 3)
plt.scatter(results["Stop"], results["profit"])
plt.title("Stop")
plt.show()

results["max_profit"] = results["profit"].max()
# Work out the range within 5%
best_results = results[results.profit > 0.5 * results.max_profit]

print(best_results["Buy"].min())
print(best_results["Buy"].max())
print(best_results["Sell"].min())
print(best_results["Sell"].max())
print(best_results["Stop"].min())
print(best_results["Stop"].max())

##########################################
# Do a more focussed run on a more narrowed range

start_time = time.time()

# Vectorised
initial_buy_prices = list(np.linspace(28, 29, 100))
initial_sell_prices = list(np.linspace(35, 36.5, 100))
initial_stop_losses = list(np.linspace(27, 28, 100))

results = pd.DataFrame(list(product(initial_buy_prices, 
                                    initial_sell_prices,
                                    initial_stop_losses)),
                       columns = ["Buy", "Sell", "Stop"])

# Remove where buy > sell
results = results[results.Sell > (results.Buy + 0.2)] # spread added
results = results[results.Buy > (results.Stop + 0.2)] # spread added

results["profit"], results["trades_won"], results["trades_lost"] = (
    calculate_profit_vector(train_data,
                            results["Buy"],
                            results["Sell"],
                            results["Stop"],
                            max_exposure = max_exposure,
                            initial_balance = init_balance,
                            end_loss = global_end_loss))

print("--- %s seconds ---" % (time.time() - start_time))

print(results.sort_values("profit", ascending = False))
print(results.sort_values("profit", ascending = False)[["Buy", "Sell", "Stop", "profit"]])
print(results[results["trades_lost"]<1000].sort_values("profit", ascending = False))

results["max_profit"] = results["profit"].max()
# Work out the range within 5%
best_results = results[results.profit > 0.75 * results.max_profit]

print(best_results["Buy"].min())
print(best_results["Buy"].max())
print(best_results["Sell"].min())
print(best_results["Sell"].max())
print(best_results["Stop"].min())
print(best_results["Stop"].max())

##########################################

# Test the model

calculate_profit_yearly(test_data, [28.31], [35.6], [27.45], max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss)

calculate_profit_yearly(raw_prices, [10], [14], [0], max_exposure = max_exposure, initial_balance = init_balance, end_loss = global_end_loss)


##########################################

# Test the model with a monte carlo

one_year_monte_carlo = monte_carlo_test_runs(data = test_data,
                                             n_iterations = 1000,
                                             n_years = 1,
                                             buy_prices = [24.67], 
                                             sell_prices = [37.28], 
                                             stop_losses = [24.42],
                                             max_exposure = 1, 
                                             initial_balance = init_balance, 
                                             end_loss = True)

# Plot the results
one_year_monte_carlo["Percent_profit"].plot.hist(grid = True,
                                                 bins = 20)

loser_info(one_year_monte_carlo)


##########################################

# Test the model with a monte carlo

one_year_monte_carlo = monte_carlo_test_runs(data = raw_prices,
                                             n_iterations = 1000,
                                             n_years = 1,
                                             buy_prices = [24.67], 
                                             sell_prices = [37.28], 
                                             stop_losses = [24.42],
                                             max_exposure = max_exposure, 
                                             initial_balance = init_balance, 
                                             end_loss = True)

# Plot the results
one_year_monte_carlo["Percent_profit"].plot.hist(grid = True,
                                                 bins = 20)

loser_info(one_year_monte_carlo)



##########################################







