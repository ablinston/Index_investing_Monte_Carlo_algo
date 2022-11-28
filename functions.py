import numpy as np
import pandas as pd
import random as r
import statistics as st
from datetime import datetime
import polars as pl
import pdb

# =============================================================================
# def calculate_profit(data, buy_price, sell_price, max_exposure = 1e10, initial_balance = 1e4, end_loss = True, overnight_rate = 0):
#  
#     # Initial variables
#     hold = False
#     shares = 0
#     balance = initial_balance
#     
#     # Loop through each day
#     for i in range(0, len(data.index)):
#         # Check if we're on the last day and sell if so
#         if ((i == len(data.index) - 1) & hold & end_loss):
#             balance += (data["Open"][i] - buy_price) * shares
#         # If we're already holding, look to sell
#         elif hold:
#             if data["High"][i] >= sell_price:
#                 # Calculate the profit
#                 balance += (sell_price - buy_price) * shares
#                 # No long holding
#                 hold = False
#             else:
#                 # We have to charge the overnight rate
#                 pass
#         else:
#             # Check if there is an opportunity to buy
#             if data["Low"][i] <= buy_price:
#                 # Work out how many shares you have
#                 shares = min(max_exposure, balance) / buy_price
#                 hold = True
#     return balance - initial_balance
# =============================================================================

# The overnight rate is a daily % based on sonia + 2.5% usually
# SONIA has been between 4-8% until post 2008

# =============================================================================
# def calculate_profit(data, 
#                      buy_price, 
#                      sell_price, 
#                      max_exposure = 1, # a proportion of balance to be bet
#                      initial_balance = 1e4, 
#                      end_loss = False, 
#                      overnight_rate = (0.025 / 365),
#                      daily_balances = False):
#  
#     # Initial variables
#     hold = False
#     bet_per_pt = 0
#     balance = initial_balance
#     
#     daily_balance_data = pd.DataFrame({"Date":["1900-01-01"],
#                                        "High":[0.1],
#                                        "Low":[0.1],
#                                        "bet_per_pt":[0.1],
#                                        "balance":[0]})
#     
#     # Loop through each day
#     for i in range(1, len(data.index)):
#        
#         # If we're already holding, look to sell
#         if hold:
#             # We have to charge the overnight rate
#             balance -= (data["Date_ft"][i] - data["Date_ft"][i - 1]).days * data["Open"][i] * bet_per_pt * overnight_rate
#             if data["High"][i] > sell_price:
#                 # Calculate the profit
#                 balance += (sell_price - buy_price - 0.15) * bet_per_pt
#                 # No long holding
#                 bet_per_pt = 0
#                 hold = False
#                 # Check if we're on the last day and sell if so
#             elif ((i == len(data.index) - 1) & end_loss):
#                 balance += (data["Open"][i] - buy_price - 0.15) * bet_per_pt
#                 
#         else:
#             # Check if there is an opportunity to buy
#             if (data["Low"][i] < buy_price) & (data["High"][i] > buy_price):
#                 # Work out how many shares you have
#                 bet_per_pt = max_exposure * balance / buy_price # always leave some balance to pay financing for a few years
#                 hold = True
#         
#         if daily_balances:
#             # Add balance to the data frame
#             daily_balance_data = pd.concat([daily_balance_data,
#                                         pd.DataFrame({"Date":[str(data["Date"][i])],
#                                                       "High":[data["High"][i]],
#                                                       "Low" :[data["Low"][i]],
#                                                       "bet_per_pt": [bet_per_pt],
#                                                       "balance":[balance]})],
#                                        ignore_index = True)
#     if daily_balances:
#         
#         # Convert Date column to proper date
#         
#         return daily_balance_data
#     else:
#         return balance - initial_balance
# =============================================================================

# This function calculates the profits for vectors of buy prices and corresponding sell prices
def calculate_profit_vector(data, 
                             buy_prices, 
                             sell_prices, 
                             stop_losses,
                             max_exposure = 1, 
                             initial_balance = 1e4, 
                             end_loss = False,
                             daily_balances = False):
 
    # For debugging
# =============================================================================
#     data = train_data
#     buy_prices = results["Buy"]
#     sell_prices = results["Sell"]
#     stop_losses = results["Stop"]
#     max_exposure = 0.5
#     initial_balance = 10000
#     end_loss = False
#     daily_balances = False
# =============================================================================
    
    # To avoid errors, reset the index
    data = data.reset_index(drop = True)

    # Initial variables
    results_data = pl.DataFrame({"buy_price": np.array(buy_prices, dtype = float),
                                 "sell_price": np.array(sell_prices, dtype = float),
                                 "stop_loss": np.array(stop_losses, dtype = float),
                                 "bet_per_pt": np.array([0] * len(buy_prices), dtype = float),
                                 "balance": np.array([initial_balance] * len(buy_prices), dtype = float),
                                 "trades_won": np.array([0] * len(buy_prices), dtype = float),
                                 "trades_lost": np.array([0] * len(buy_prices), dtype = float)})
       
    results_data = results_data.with_column((pl.col("sell_price") - pl.col("buy_price") - 0.15).alias("buy_sell_diff")) # includes the spread paid deducted
    results_data = results_data.with_column((pl.col("stop_loss") - pl.col("buy_price") - 0.15).alias("stop_loss_diff"))
    
    daily_balance_data = pl.DataFrame({"Date": ["1900-01-01"],
                                       "High": [0.1],
                                       "Low": [0.1],
                                       "buy_price": np.array([0], dtype = float),
                                       "sell_price": np.array([0], dtype = float),
                                       "bet_per_pt":[0.1],
                                       "balance": np.array([0], dtype = float),
                                       "trades_won": np.array([0], dtype = float),
                                       "trades_lost": np.array([0], dtype = float)})
        
    
    # Loop through each day
    for i in range(1, (len(data.index) - 1)):
       
        # Calculate overnight costs
        results_data = results_data.with_column((pl.col("bet_per_pt") * 
                                                 pl.lit((data["Date_ft"][i] - data["Date_ft"][i - 1]).days) * 
                                                 pl.lit(data["overnight_cost_per_pt"][i])
                                                ).alias("overnight_costs"))
                                                
        # Update balances with costs
        results_data = results_data.with_column((pl.col("balance") - pl.col("overnight_costs")).alias("balance"))
        
        # Check if we hit a stop loss during the day
        results_data = results_data.with_column(((pl.col("stop_loss") >= pl.lit(data["Low"][i])) & 
                                                  (pl.col("bet_per_pt") > pl.lit(0))
                                                 ).alias("stopped_ind"))
        
        # Sell out holding for those where it's true
        results_data = results_data.with_column((pl.col("balance") + 
                                                 (pl.col("stop_loss_diff") *
                                                  pl.col("bet_per_pt") *
                                                  pl.col("stopped_ind"))
                                                 ).alias("balance"))
        
        # Check if we've hit a selling opportunity in the day
        results_data = results_data.with_column(((~pl.col("stopped_ind")) &
                                                 (pl.col("bet_per_pt") > pl.lit(0)) &
                                                 (pl.col("sell_price") < pl.lit(data["High"][i]))
                                                 ).alias("sell_ind"))
        
        # Sell out holding for those where it's true
        results_data = results_data.with_column((pl.col("balance") + 
                                                 pl.col("sell_ind") *
                                                 pl.col("bet_per_pt") *
                                                 pl.col("buy_sell_diff")
                                                 ).alias("balance"))
        
        # Calculate running tally of wins and losses
        results_data = results_data.with_column((pl.when((pl.col("balance") <= pl.lit(0)) & 
                                                         (pl.col("bet_per_pt") == pl.lit(0)))
                                                 .then(pl.col("trades_won"))
                                                 .otherwise(pl.col("trades_won") + pl.lit(1) * pl.col("sell_ind"))
                                                 ).alias("trades_won"))
        
        results_data = results_data.with_column((pl.when((pl.col("balance") <= pl.lit(0)) & 
                                                         (pl.col("bet_per_pt") == pl.lit(0)))
                                                 .then(pl.col("trades_lost"))
                                                 .otherwise(pl.col("trades_lost") + pl.lit(1) * pl.col("stopped_ind"))
                                                 ).alias("trades_lost"))
                        
        # Check if we've hit a day for buying before reseting bet (so we don't buy and sell on same day)
        results_data = results_data.with_column(((pl.col("bet_per_pt") == pl.lit(0)) &
                                                 (pl.col("buy_price") < pl.lit(data["High"][i])) &
                                                 (pl.col("buy_price") > pl.lit(data["Low"][i])))
                                                .alias("buy_ind"))
        
        # Set the bet to 0 if sell indicator flagged
        results_data = results_data.with_column((pl.when(pl.col("sell_ind") | pl.col("stopped_ind")).
                                                 then(pl.lit(0)).
                                                 otherwise(pl.col("bet_per_pt")))
                                                 .alias("bet_per_pt"))
        
        # Work out the size of bet available                
        # Now bet on the shares where appropriate
        results_data = results_data.with_column((pl.when(pl.col("buy_ind"))
                                                 .then(pl.col("balance") *
                                                       pl.lit(max_exposure) /
                                                       pl.col("buy_price"))
                                                 .otherwise(pl.col("bet_per_pt")))
                                                .alias("bet_per_pt"))
        
        # Check whether the trade gets stopped out at the end of the day
        # Check if we hit a stop loss during the day
        results_data = results_data.with_column(((pl.col("stop_loss") >= pl.lit(data["Close"][i])) & 
                                                  (pl.col("bet_per_pt") > pl.lit(0))
                                                 ).alias("stopped_ind"))
        
        # Sell out holding for those where it's true
        results_data = results_data.with_column((pl.col("balance") + 
                                                 ((pl.lit(data["Close"][i]) - pl.col("buy_price") - 0.15) *
                                                  pl.col("bet_per_pt") *
                                                  pl.col("stopped_ind"))
                                                 ).alias("balance"))

        # Set the bet to 0 if sell indicator flagged
        results_data = results_data.with_column((pl.when(pl.col("stopped_ind")).
                                                 then(pl.lit(0)).
                                                 otherwise(pl.col("bet_per_pt")))
                                                 .alias("bet_per_pt"))
        
        # Check whether balance has gone to 0 and reset variables if so
        results_data = results_data.with_column((pl.when(pl.col("balance") <= 0)
                                                 .then(pl.lit(0))
                                                 .otherwise(pl.col("balance"))
                                                 ).alias("balance"))
        
        results_data = results_data.with_column((pl.when(pl.col("balance") <= 0)
                                                 .then(pl.lit(0))
                                                 .otherwise(pl.col("bet_per_pt"))
                                                 ).alias("bet_per_pt"))
        
        
        if daily_balances:
            # Add balance to the data frame
            daily_balance_data = pl.concat([daily_balance_data,
                                            pl.concat([pl.DataFrame({"Date":[str(data["Date"][i])] * len(results_data),
                                                                     "High":[data["High"][i]] * len(results_data),
                                                                     "Low":[data["Low"][i]] * len(results_data)}),
                                                                   results_data.select(["buy_price", 
                                                                                        "sell_price", 
                                                                                        "bet_per_pt",
                                                                                        "balance",
                                                                                        "trades_won",
                                                                                        "trades_lost"])],
                                                      how = "horizontal")])
        
    # On the last day, sell out if necessary
    if end_loss:
        results_data = results_data.with_column((pl.col("balance") +
                                                 pl.col("bet_per_pt") *
                                                 (pl.lit(data["Open"][len(data.index) - 1]) - pl.col("buy_price") - pl.lit(0.15))
                                                 ).alias("balance"))
        
    # Calculate profits
    results_data = results_data.with_column((pl.col("balance") - initial_balance)
                                            .alias("profit"))
    
    if daily_balances:
        return daily_balance_data
    else:        
        return results_data[["profit", "trades_won", "trades_lost"]]



# This function calculates profits for single years only
def calculate_profit_yearly(data, 
                            buy_prices, # a list of values
                            sell_prices, 
                            stop_losses,
                            max_exposure, 
                            initial_balance, 
                            end_loss = False):

# =============================================================================
#     # For debugging
#     data = test_data
#     buy_prices = [20.1]
#     sell_prices = [30.1]
#     stop_losses = [15]
#     max_exposure = 0.5
#     initial_balance = 20000
#     end_loss = False
# =============================================================================

    daily_data = calculate_profit_vector(data, 
                                        pd.Series(buy_prices), # input as a Series
                                        pd.Series(sell_prices), 
                                        pd.Series(stop_losses),
                                        max_exposure = max_exposure, 
                                        initial_balance = initial_balance / len(buy_prices), 
                                        end_loss = end_loss,
                                        daily_balances = True).to_pandas()
    
    # Split the date column
    daily_data[["Year", "Month", "Day"]] = daily_data["Date"].str.split("-", expand = True)

    yearly_data = daily_data.groupby("Year", as_index = False).last()
    
    # Add yearly return
    yearly_data["prior_balance"] = yearly_data["balance"].shift(1)
    yearly_data["annual_return"] = 100 * (yearly_data["balance"] / yearly_data["prior_balance"] - 1)
    
    # Calculate CAGR
    cagr = round(((daily_data.loc[len(daily_data.index) - 1, "balance"] / daily_data.loc[1, "balance"]
            ) ** (1 / ((datetime.strptime(daily_data.loc[len(daily_data.index) - 1, "Date"],
                                       "%Y-%m-%d") -
                     datetime.strptime(daily_data.loc[1, "Date"],
                                       "%Y-%m-%d")).days / 365)) - 1) * 100, 1)
    
    print(f"CAGR rate is {cagr}%")
    
    return yearly_data.loc[1:, ["Year", "balance", "annual_return", "trades_won", "trades_lost"]]


   
    
def monte_carlo_test_runs(data,
                            n_iterations,
                            n_years,
                            buy_prices, 
                            sell_prices, 
                            stop_losses,
                            max_exposure = 1, 
                            initial_balance = 1e4, 
                            end_loss = True, 
                            overnight_rate = (0.065 / 365)):
     
    
# =============================================================================
# =============================================================================
#      # For debugging
#      data = raw_prices
#      n_iterations = 2
#      n_years = 1
#      buy_prices = [20.1] 
#      sell_prices = [30.1]
#      stop_losses = [20]
#      max_exposure = 0.5
#      initial_balance = 1e4 
#      end_loss = True
# =============================================================================
# =============================================================================
    
    # Set the minimum start year given the data we have
    min_start_year = min(data["Year"])
    max_start_year = max(data["Year"]) - n_years
    
    # Prepare stack of results
    results_stack = pd.DataFrame(columns = ["Buy", "Sell", "Stop", "Profit", "mc_run"])
    
    for iteration in range(1, n_iterations + 1):
    
        # We want at least n years of data, so choose a random start point
        start_year = r.randrange(min_start_year,
                                 max_start_year)
        # Choose a random start month
        start_month = r.randrange(1, 13)
        
        data_subset = data[(data.Year >= start_year) &
                           (data.Year <= (start_year + n_years))]
        
        data_subset = data_subset[~((data_subset.Year == start_year) &
                                    (data_subset.Month < start_month))]
        
        data_subset = data_subset[~((data_subset.Year == data_subset.Year.max()) &
                                    (data_subset.Month > start_month))]
           
        data_subset = data_subset.reset_index(drop = True)
        
        # Prepare inputs for the model
        results = pd.DataFrame(list(product(buy_prices, sell_prices, stop_losses)),
                               columns = ["Buy", "Sell", "Stop"])
        
        # Now work out the best buy and sell
        results["Profit"], results["trades_won"], results["trades_lost"]  = (
            calculate_profit_vector(data_subset,
                                    results["Buy"],
                                    results["Sell"],
                                    results["Stop"],
                                    max_exposure = max_exposure,
                                    initial_balance = (initial_balance / len(buy_prices)), # split the balance across the strategies being run
                                    end_loss = end_loss))
        
        # Now add the results to the stack
        results["mc_run"] = iteration
        
        results_stack = pd.concat([results_stack,
                                   results[["Buy", "Sell", "Stop", "Profit", "mc_run"]]])

        if iteration % 50 == 0:
            print(f"{iteration} runs complete")
    
        del results
        del data_subset
    
    results_stack["Percent_profit"] = results_stack["Profit"] / initial_balance * 100
    
    return results_stack


# This function returns stats associated with losers from results of monte carlo
def loser_info(data):
    
# =============================================================================
#     data = one_year_monte_carlo
# =============================================================================

    prob_of_losing = round(len(data[data.Profit < 0].index) / len(data.index) * 100, 1)
    average_loss = round(data[data.Profit < 0]["Percent_profit"].mean(),1)
    max_loss = round(data[data.Profit < 0]["Percent_profit"].min(),1)
    average_gain = round(data[data.Profit > 0]["Percent_profit"].mean(),1)
    total_loss_probability = round(
        100 * (len(data[data.Percent_profit < -90].index) /
               len(data.index)),
        1)

    return print(f"{prob_of_losing}% chance of losing. Average loss {average_loss}% and max loss {max_loss}%.\
                 \nProbability of >90% loss is {total_loss_probability}%\
                     \nAverage gain of {average_gain}%")