import numpy as np
import pandas as pd
import random as r
import statistics as st
from datetime import datetime
import polars as pl
import pdb


# This function calculates the alphas for vectors of buy prices and corresponding sell prices
def calculate_alpha_vector(data,
                            buy_triggers, # a vector of proportions of changes in the index from a previous high
                            sell_triggers, # a vector of proportion increases in the index from a buy point,
                            assumed_annual_dividend, # a proportion yield assumed for holding
                            initial_balance = 1e4,
                            daily_values = False):
 
# =============================================================================
#     # For debugging
#     data = raw_prices
#     buy_triggers = pd.Series([-0.2])
#     sell_triggers = pd.Series([1])
#     assumed_annual_dividend = 0.03
#     initial_balance = 10000
#     daily_values = True
# =============================================================================
    
    # Sort by date
    data = data.sort_values("Date_ft")

    # To avoid errors, reset the index
    data = data.reset_index(drop = True)
    
    # Set up a control run which is buy and hold
    buy_triggers = pd.concat([pd.Series([0]), buy_triggers])
    sell_triggers = pd.concat([pd.Series([1e6]), sell_triggers])
    
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
        
    # Manually set the control to buy on the first day
    results_data[0, "buy_price"] = data["High"][1]
    
    # Loop through each day
    for i in range(1, (len(data.index) - 1)):
       
        # Calculate overnight reinvested dividends as an increase to holding
        results_data = results_data.with_column((pl.col("shares_held") * pl.lit((1 + assumed_annual_dividend) ** (1/260)
                                                ).alias("shares_held")))
        
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
        
    # Calculate alphas
    benchmark = results_data[0, "total_value"]
    results_data = results_data.with_column((pl.col("total_value") - benchmark)
                                            .alias("alpha"))
    
    if daily_values:
        return daily_balance_data
    else:        
        return results_data[1:, ["alpha", "trades"]]



# This function calculates alphas for single years only
def calculate_alpha_yearly(data, 
                            buy_triggers, # a list of values
                            sell_triggers,
                            assumed_annual_dividend,
                            initial_balance, 
                            end_loss = False):

    # For debugging
# =============================================================================
#     data = test_data
#     buy_triggers = [20.1]
#     sell_triggers = [30.1]
#     stop_losses = [15]
#     max_exposure = 0.5
#     initial_balance = 20000
#     end_loss = False
# =============================================================================

    daily_data = calculate_alpha_vector(data, 
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
                            assumed_annual_dividend,
                            initial_balance = 1e4,
                            years_hard_limit = False): #whether we always want that number of years exactly
     
    
# =============================================================================
# =============================================================================
#      # For debugging
#      data = raw_prices
#      n_iterations = 2
#      n_years = 100
#      buy_triggers = [20.1] 
#      sell_triggers = [30.1]
#      stop_losses = [20]
#      max_exposure = 0.5
#      initial_balance = 1e4 
#      end_loss = True
#      years_hard_limit = False
# =============================================================================
# =============================================================================
    
    # Set the minimum start year given the data we have
    min_start_year = min(data["Year"])
    
    if years_hard_limit:
        max_start_year = max(data["Year"]) - n_years
    else:
        max_start_year = max(data["Year"]) - 2 # get at least 2 years
    
    # Prepare stack of results
    results_stack = pd.DataFrame(columns = ["Buy", "Sell", "alpha", "mc_run"])
    
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
        results = pd.DataFrame(list(product(buy_triggers, sell_triggers)),
                               columns = ["Buy", "Sell"])
        
        # Now work out the best buy and sell
        results["alpha"], results["trades"]  = (
            calculate_alpha_vector(data_subset,
                                    results["Buy"],
                                    results["Sell"],
                                    assumed_annual_dividend = assumed_annual_dividend,
                                    initial_balance = (initial_balance / len(buy_triggers)) # split the balance across the strategies being run
                                    ))
        
        # Now add the results to the stack
        results["mc_run"] = iteration
        
        results_stack = pd.concat([results_stack,
                                   results[["Buy", "Sell", "alpha", "mc_run"]]])

        if iteration % 50 == 0:
            print(f"{iteration} runs complete")
    
        del results
        del data_subset
    
    results_stack["Percent_alpha"] = results_stack["alpha"] / initial_balance * 100
    
    return results_stack


def fast_monte_carlo_test_runs(data,
                                n_iterations,
                                min_years,
                                buy_trigger, 
                                sell_trigger,
                                assumed_annual_dividend,
                                initial_balance = 1e4,
                                to_end = True): #whether we always want that number of years exactly
     
    
# =============================================================================
# =============================================================================
#     # For debugging
#     data = raw_prices
#     n_iterations = 5
#     min_years = 5
#     buy_trigger = -0.1
#     sell_trigger = 2.1
#     initial_balance = 1e4 
#     years_hard_limit = False
# =============================================================================
# =============================================================================

    # Set the minimum start year given the data we have
    min_start_year = min(data["Year"])
    max_start_year = max(data["Year"]) - min_years
    
    # Prepare stack of results
    results_stack = pl.DataFrame({"mc_run": np.array(list(range(1, n_iterations + 1)), dtype = float),
                                  "shares_held": np.array([0] * n_iterations, dtype = float),
                                  "buy_price": np.array([0] * n_iterations, dtype = float),
                                  "sell_price": np.array([1e6] * n_iterations, dtype = float),
                                  "cash_balance": np.array([initial_balance] * n_iterations, dtype = float),
                                  "total_value": np.array([0] * n_iterations, dtype = float),
                                  "trades": np.array([0] * n_iterations, dtype = float)})
    
    # Add empty input frame ready to stack
    input_stack = pl.DataFrame(data.loc[1:0])
    input_stack = input_stack.with_columns((pl.lit(0.0)).alias("mc_run"))
    input_stack = input_stack.with_columns((pl.lit(0)).alias("index"))
    input_stack = input_stack.drop("Date") # causing stacking issues
    
    max_days = 0
    data = data.sort_values("Date_ft")
        
    # Create stack of input data
    for iteration in range(1, n_iterations + 1):
    
        # We want at least n years of data, so choose a random start point
        start_year = r.randrange(min_start_year,
                                 max_start_year)
        # Choose a random start month
        start_month = r.randrange(1, 13)
        
        data_subset = data[(data.Year >= start_year)]
        
        data_subset = data_subset[~((data_subset.Year == start_year) &
                                    (data_subset.Month < start_month))]
        
        #data_subset = data_subset.reset_index(drop = True)
        data_subset["mc_run"] = float(iteration)
        data_subset["index"] = range(1, len(data_subset) + 1)
        
        # Save how many days there are to find the max
        max_days = max(max_days, len(data_subset.mc_run))
        
        input_stack = pl.concat([input_stack,
                                 pl.DataFrame(data_subset[["Open", "High",
                                                           "Low", "Close", "Date_ft", "Year", 
                                                           "Month", "ATH", "discount", "mc_run", "index"]])],
                                how = "vertical")
        
    #input_stack = input_stack.reset_index(drop = True)
    
    # Add rows to the results stack to give the comparable buy and hold strategy
    results_stack = results_stack.with_columns([
        (pl.lit(buy_trigger)).alias("buy_trigger"),
        (pl.lit(sell_trigger)).alias("sell_trigger"),
        (pl.lit(buy_trigger)).alias("original_buy_trigger"),
        (pl.lit(sell_trigger)).alias("original_sell_trigger"),
        (pl.lit(False)).alias("benchmark")
        ])
    
    results_stack_control = results_stack.with_columns([
        (pl.lit(0.0)).alias("buy_trigger"),
        (pl.lit(1000000.0)).alias("sell_trigger"),
        (pl.lit(True)).alias("benchmark")
        ])
    
    results_stack = pl.concat([results_stack,
                               results_stack_control],
                              how = "vertical")
    
    # Now that the input data is ready, loop through all the days    
    for day in range(1, max_days + 1):
        
        todays_data = input_stack.filter(pl.col("index") == day).drop("index")
          
        # Merge onto the results stack
        results_stack = results_stack.join(
            todays_data,
            on = "mc_run",
            how = "left")
        
        # For anywhere that has run out of data, sell out and finalise the value
        results_stack = results_stack.with_columns([
            pl.when((pl.col("Open").is_null()))
            .then(pl.col("total_value"))
            .otherwise(pl.col("cash_balance"))
            .alias("cash_balance"),
            pl.when((pl.col("Open").is_null()))
            .then(pl.lit(0))
            .otherwise(pl.col("shares_held"))
            .alias("shares_held"),
            pl.when((pl.col("Open").is_null()))
            .then(pl.lit(1e6)) #  set buy trigger very high so it never happens
            .otherwise(pl.col("buy_trigger"))
            .alias("buy_trigger")
            ])
        
        # Calculate overnight reinvested dividends as an increase to holding
        results_stack = results_stack.with_column(
            (pl.col("shares_held") * pl.lit((1 + assumed_annual_dividend) ** (1/260)) #  used working days so don't have to worry about weekends
             ).alias("shares_held"))
        
        # Check if we've hit a selling opportunity in the day
        results_stack = results_stack.with_column(
            ((pl.col("shares_held") > pl.lit(0)) &
             (pl.col("sell_price") <= pl.col("High"))
             ).alias("sell_ind"))
        
        # Sell out holding for those where it's true
        results_stack = results_stack.with_column(
            (pl.col("cash_balance") + 
             pl.col("sell_ind") *
             pl.col("shares_held") *
             pl.col("sell_price")
             ).alias("cash_balance"))
        
        # See if we set a new high and set buy price accordingly
        results_stack = results_stack.with_columns(
            ((pl.lit(1) + pl.col("buy_trigger")) *
             pl.col("ATH")
            ).alias("buy_price"))
                                
        # Check if we've hit a day for buying before reseting bet (so we don't buy and sell on same day)
        results_stack = results_stack.with_column(
            ((pl.col("shares_held") == pl.lit(0)) &
             (pl.col("buy_price") >= pl.col("Low"))
             ).alias("buy_ind"))
        
        # Now sell the holding if necessary
        results_stack = results_stack.with_columns(
            (pl.col("shares_held") *
             (~pl.col("sell_ind"))
             ).alias("shares_held"))
                
        # Now execute the buy where appropriate
        results_stack = results_stack.with_column(
            (pl.when(pl.col("buy_ind"))
             .then((pl.col("cash_balance") - pl.lit(25)) / # trading cost
                   pl.col("buy_price"))
             .otherwise(pl.col("shares_held")))
            .alias("shares_held"))
        
        # Update the cash balance
        results_stack = results_stack.with_column(
            (pl.when(pl.col("buy_ind"))
             .then(pl.lit(0))
             .otherwise(pl.col("cash_balance")))
            .alias("cash_balance"))
        
        # Count the trade
        results_stack = results_stack.with_column(
            (pl.when(pl.col("buy_ind"))
             .then(pl.col("trades") + pl.lit(1))
             .otherwise(pl.col("trades")))
            .alias("trades"))
        
        # Set the sell price
        results_stack = results_stack.with_column(
            (pl.when(pl.col("buy_ind"))
             .then((pl.lit(1) + pl.col("sell_trigger")) *
                   pl.col("buy_price"))
             .otherwise(pl.col("sell_price"))
             ).alias("sell_price"))
   
        # Calculate the total valuation today
        results_stack = results_stack.with_column(
            pl.when(pl.col("Close").is_null())
            .then(pl.col("cash_balance"))
            .otherwise((pl.col("cash_balance") +
                        pl.col("shares_held") * pl.col("Close")))
            .alias("total_value"))
        
        # Drop columns with today's data ready for next day
        results_stack = results_stack.drop(["Open", "High",
                                            "Low", "Close", "Date_ft", "Year", 
                                            "Month", "ATH"])
        
        #######################################
        
        
    # Calculate the returns for the benchmark
    
    benchmark = results_stack.filter(pl.col("benchmark") == True).select(["mc_run", "total_value"])
    benchmark.columns = ["mc_run", "benchmark_value"]
    
    # Work out how many years each period lasted for
    day_count = input_stack.groupby("mc_run").count()
    day_count.columns = ["mc_run", "days"]
    
    results_stack = results_stack.join(benchmark,
                                       on = "mc_run",
                                       how = "left").filter(pl.col("benchmark") == False)
    
    results_stack = results_stack.join(day_count,
                                       on = "mc_run",
                                       how = "left")

    # Now work out the best buy and sell
    results_stack = results_stack.with_column(
        (pl.col("total_value") - pl.col("benchmark_value"))
        .alias("alpha")
        )
        
    # Calculate alpha as a percentage
    results_stack = results_stack.with_column(
        (pl.col("alpha") / pl.col("benchmark_value") * pl.lit(100))
        .alias("Percent_alpha")
        )
    
    # Calculate the compound annual growth rates
    results_stack = results_stack.with_columns([
        (((pl.col("total_value") / pl.lit(initial_balance)) ** 
         (pl.lit(1) / (pl.col("days") / pl.lit(260))) - pl.lit(1)) * pl.lit(100))
        .alias("CAGR"),
        (((pl.col("benchmark_value") / pl.lit(initial_balance)) ** 
         (pl.lit(1) / (pl.col("days") / pl.lit(260))) - pl.lit(1)) * pl.lit(100))
        .alias("Benchmark_CAGR")
        ])
    
    results_stack = results_stack.with_column(
        (pl.col("CAGR") - pl.col("Benchmark_CAGR"))
        .alias("CAGR_alpha")
        )
    
    return results_stack.select(["mc_run", 
                                 "original_buy_trigger", 
                                 "original_sell_trigger",
                                 "trades",
                                 "total_value",
                                 "benchmark",
                                 "alpha",
                                 "Percent_alpha",
                                 "CAGR",
                                 "Benchmark_CAGR",
                                 "CAGR_alpha"]).to_pandas()

# This function returns stats associated with losers from results of monte carlo
def loser_info(data):
    
# =============================================================================
#     data = one_year_monte_carlo
# =============================================================================

    prob_of_losing_to_market = round(len(data[data.alpha < 0].index) / len(data.index) * 100, 1)
    average_underperformance = round(data[data.alpha < 0]["Percent_alpha"].mean(),1)
    max_underperformance = round(data[data.alpha < 0]["Percent_alpha"].min(),1)
    average_outperformance = round(data[data.alpha > 0]["Percent_alpha"].mean(),1)

    return print(f"{prob_of_losing_to_market}% chance of not beating market. \nAverage under-performance {average_underperformance}% and max underperformance is {max_underperformance}%.\
                     \nAverage out-performance of {average_outperformance}%")
