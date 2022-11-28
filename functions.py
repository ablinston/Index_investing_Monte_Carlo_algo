import numpy as np
import pandas as pd
import random as r
import statistics as st
from datetime import datetime
import polars as pl
import pdb


# This function calculates the profits for vectors of buy prices and corresponding sell prices
def calculate_profit_vector(data,
                            buy_triggers, # a vector of proportions of changes in the index from a previous high
                            sell_triggers, # a vector of proportion increases in the index from a buy point,
                            assumed_annual_dividend, # a proportion yield assumed for holding
                            initial_balance = 1e4,
                            daily_values = False):
 
# =============================================================================
#     # For debugging
#     data = raw_prices
#     buy_triggers = pd.Series([0])
#     sell_triggers = pd.Series([1e6])
#     assumed_annual_dividend = 0.03
#     initial_balance = 10000
#     daily_values = True
# =============================================================================
    
    # Sort by date
    data = data.sort_values("Date_ft")

    # To avoid errors, reset the index
    data = data.reset_index(drop = True)
    
# =============================================================================
#     # Set up a control run which is buy and hold
#     buy_triggers = pd.concat([pd.Series([1]), buy_triggers])
#     sell_triggers = pd.concat([pd.Series([1e6]), sell_triggers])
#     
# =============================================================================
    # Initial variables
    results_data = pl.DataFrame({"buy_trigger": np.array(buy_triggers, dtype = float),
                                 "sell_trigger": np.array(sell_triggers, dtype = float),
                                 "shares_held": np.array([0] * len(buy_triggers), dtype = float),
                                 "buy_price": np.array([data["Open"][1]] * len(buy_triggers), dtype = float),
                                 "sell_price": np.array([1e6] * len(buy_triggers), dtype = float),
                                 "cash_balance": np.array([initial_balance] * len(buy_triggers), dtype = float),
                                 "total_value": np.array([0] * len(buy_triggers), dtype = float),
                                 "trades": np.array([0] * len(buy_triggers), dtype = float)})
       
    daily_balance_data = pl.DataFrame({"Date": ["1900-01-01"],
                                       "High": [0.1],
                                       "Low": [0.1],
                                       "buy_price": np.array([0], dtype = float),
                                       "sell_price": np.array([0], dtype = float),
                                       "shares_held": np.array([0], dtype = float),
                                       "trades": np.array([0], dtype = float),
                                       "cash_balance": np.array([0], dtype = float),
                                       "total_value": np.array([0], dtype = float)})
        
# =============================================================================
#     # Manually set the control to buy on the first day
#     results_data[0, "buy_price"] = data["High"][1]
# =============================================================================
    
    # Loop through each day
    for i in range(1, (len(data.index) - 1)):
       
        # Calculate overnight reinvested dividends as an increase to holding
        results_data = results_data.with_column((pl.col("shares_held") * pl.lit((1 + assumed_annual_dividend) ** (1/365))
                                                ).alias("shares_held"))
        
        # Check if we've hit a selling opportunity in the day
        results_data = results_data.with_column(((pl.col("shares_held") > pl.lit(0)) &
                                                 (pl.col("sell_price") <= pl.lit(data["High"][i]))
                                                 ).alias("sell_ind"))
        
        # Sell out holding for those where it's true
        results_data = results_data.with_column((pl.col("cash_balance") + 
                                                 pl.col("sell_ind") *
                                                 pl.col("shares_held") *
                                                 pl.col("sell_price")
                                                 ).alias("cash_balance"))
        
        # See if we set a new high and set buy price accordingly
        results_data = results_data.with_columns(((pl.lit(1) + pl.col("buy_trigger")) *
                                                  pl.lit(data["ATH"][i])
                                                  ).alias("buy_price"))
                                
        # Check if we've hit a day for buying before reseting bet (so we don't buy and sell on same day)
        results_data = results_data.with_column(((pl.col("shares_held") == pl.lit(0)) &
                                                 (pl.col("buy_price") >= pl.lit(data["Low"][i])))
                                                .alias("buy_ind"))
        
        # Now sell the holding if necessary
        results_data = results_data.with_columns((pl.col("shares_held") *
                                                  (~pl.col("sell_ind"))
                                                  ).alias("shares_held"))
                
        # Now execute the buy where appropriate
        results_data = results_data.with_column((pl.when(pl.col("buy_ind"))
                                                 .then((pl.col("cash_balance") - pl.lit(25)) / # trading cost
                                                       pl.col("buy_price"))
                                                 .otherwise(pl.col("shares_held")))
                                                .alias("shares_held"))
        
        # Update the cash balance
        results_data = results_data.with_column((pl.when(pl.col("buy_ind"))
                                                 .then(pl.lit(0))
                                                 .otherwise(pl.col("cash_balance")))
                                                .alias("cash_balance"))
        
        # Count the trade
        results_data = results_data.with_column((pl.when(pl.col("buy_ind"))
                                                 .then(pl.col("trades") + pl.lit(1))
                                                 .otherwise(pl.col("trades")))
                                                .alias("trades"))
        
        # Set the sell price
        results_data = results_data.with_column((pl.when(pl.col("buy_ind"))
                                                 .then((pl.lit(1) + pl.col("sell_trigger")) *
                                                       pl.col("buy_price"))
                                                 .otherwise(pl.col("sell_price"))
                                                 ).alias("sell_price"))
        
        # Calculate the total valuation today
        results_data = results_data.with_column((pl.col("cash_balance") +
                                                 pl.col("shares_held") * pl.lit(data["Close"][i])
                                                 ).alias("total_value"))
        
        if daily_values:
            # Add balance to the data frame
            daily_balance_data = pl.concat([daily_balance_data,
                                            pl.concat([pl.DataFrame({"Date": [datetime.strftime(data["Date_ft"][i], "%Y-%m-%d")] * len(results_data),
                                                                     "High":[data["High"][i]] * len(results_data),
                                                                     "Low":[data["Low"][i]] * len(results_data)}),
                                                                   results_data.select(["buy_price", 
                                                                                        "sell_price", 
                                                                                        "shares_held",
                                                                                        "trades",
                                                                                        "cash_balance",
                                                                                        "total_value"])],
                                                      how = "horizontal")])
        
    # Calculate profits
    results_data = results_data.with_column((pl.col("total_value") - initial_balance)
                                            .alias("profit"))
    
    if daily_values:
        return daily_balance_data
    else:        
        return results_data[["profit", "trades"]]



# This function calculates profits for single years only
def calculate_profit_yearly(data, 
                            buy_triggers, # a list of values
                            sell_triggers,
                            assumed_annual_dividend,
                            initial_balance, 
                            end_loss = False):

# =============================================================================
#     # For debugging
#     data = test_data
#     buy_triggers = [20.1]
#     sell_triggers = [30.1]
#     stop_losses = [15]
#     max_exposure = 0.5
#     initial_balance = 20000
#     end_loss = False
# =============================================================================

    daily_data = calculate_profit_vector(data, 
                                        pd.Series(buy_triggers), # input as a Series
                                        pd.Series(sell_triggers),
                                        assumed_annual_dividend = assumed_annual_dividend,
                                        initial_balance = initial_balance / len(buy_triggers),
                                        daily_values = True).to_pandas()
    
    # Split the date column
    daily_data[["Year", "Month", "Day"]] = daily_data["Date"].str.split("-", expand = True)

    yearly_data = daily_data.groupby("Year", as_index = False).last()
    
    # Add yearly return
    yearly_data["prior_total_value"] = yearly_data["total_value"].shift(1)
    yearly_data["annual_return"] = 100 * (yearly_data["total_value"] / yearly_data["prior_total_value"] - 1)
    
    # Calculate CAGR
    cagr = round(((daily_data.loc[len(daily_data.index) - 1, "total_value"] / daily_data.loc[1, "total_value"]
            ) ** (1 / ((datetime.strptime(daily_data.loc[len(daily_data.index) - 1, "Date"],
                                       "%Y-%m-%d") -
                     datetime.strptime(daily_data.loc[1, "Date"],
                                       "%Y-%m-%d")).days / 365)) - 1) * 100, 1)
    
    print(f"CAGR rate is {cagr}%")
    
    return yearly_data.loc[1:, ["Year", "total_value", "annual_return", "trades"]]


   
    
def monte_carlo_test_runs(data,
                            n_iterations,
                            n_years,
                            buy_triggers, 
                            sell_triggers, 
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
#      buy_triggers = [20.1] 
#      sell_triggers = [30.1]
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
        results = pd.DataFrame(list(product(buy_triggers, sell_triggers, stop_losses)),
                               columns = ["Buy", "Sell", "Stop"])
        
        # Now work out the best buy and sell
        results["Profit"], results["trades_won"], results["trades_lost"]  = (
            calculate_profit_vector(data_subset,
                                    results["Buy"],
                                    results["Sell"],
                                    results["Stop"],
                                    max_exposure = max_exposure,
                                    initial_balance = (initial_balance / len(buy_triggers)), # split the balance across the strategies being run
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
