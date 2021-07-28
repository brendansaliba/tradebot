from datetime import datetime
from datetime import timedelta

from functions import setup_func
from cashflow.classes.indicators import Indicators_Isaac

symbol = "NIO"
trading_options = True

# Sets up the robot class, robot's portfolio, and the TDSession object
trading_robot = setup_func()

# Grab the historical prices for the symbol we're trading.
end_date = datetime.today()
start_date = end_date - timedelta(minutes=200)  # previously seconds=5 ???

historical_prices = trading_robot.grab_historical_prices(
    start=end_date,
    end=start_date,
    bar_size=1,
    bar_type='minute',
    symbols=[symbol]
)

# Convert data to a Data StockFrame
stock_frame = trading_robot.stock_frame

# Get positions on startup
trading_robot.get_positions_for_symbol(symbol=symbol)

# Get current date and create the excel sheet name
now = datetime.now().strftime("%Y_%m_%d-%I%M_%p")
filename = "{}_run_{}".format(symbol, now)
json_path = trading_robot.json_path
full_path = json_path + r"/" + filename + ".xlsx"

# Give data to indicator.
trading_robot.indicator.price_data_frame = stock_frame

# Add indicators to the client
trading_robot.indicator.per_of_change()
trading_robot.indicator.sma(period=9)
trading_robot.indicator.sma(period=50)
trading_robot.indicator.sma9_crossed_sma50()
trading_robot.indicator.sma(period=200)
trading_robot.indicator.sma_volume(period=9)
trading_robot.indicator.sma_volume(period=50)
trading_robot.indicator.sma_volume(period=200)
trading_robot.indicator.abs_9_minus_50_slope()
trading_robot.indicator.max_option_chain(symbol)
trading_robot.indicator.buy_condition(symbol)

# Define initial refresh time so we know when to refresh the TDClient
refresh_time = datetime.now() + timedelta(minutes=21)

# Do the loop
while True:
    # Update token after 21 minutes
    if datetime.now() > refresh_time:
        trading_robot.session.login()
        print('refresh')
        refresh_time = datetime.now() + timedelta(minutes=21)

    # Get the stock DF from indicators
    stock_df = trading_robot.indicator.stock_data
    # print(stock_df.tail())

    # Send signals, puts, and calls to the bot client
    # TODO Fix this dumb shit lmao
    trading_robot.signals = trading_robot.indicator.indicator_signal_list
    trading_robot.call_options = trading_robot.indicator.calls_options
    trading_robot.put_options = trading_robot.indicator.puts_options

    # Set the StockFrame in the robot to the same as the indicator one
    trading_robot.stock_frame = stock_df
    order, order_response = trading_robot.execute_orders_2(symbol=symbol, trading_options=trading_options)

    # Add order info to the dataframe
    stock_info_df = trading_robot.indicator.populate_order_data_2(order=order)

    # Save an excel sheet with the data
    # stock_info_df[269:].to_excel(full_path)

    # Grab the latest bar.
    latest_bars = trading_robot.get_latest_bar(symbol=symbol)

    # Add to the Stock Frame.
    stock_frame.add_rows(data=latest_bars)

    # Update the stock frame in the robot and indicator client again
    trading_robot.stock_frame = stock_frame.frame
    trading_robot.indicator.stock_data = stock_frame.frame

    # Refresh the Indicators.
    trading_robot.indicator.refresh()

    # Grab the last bar.
    last_bar_timestamp = trading_robot.stock_frame.tail(n=1).index.get_level_values(1)

    # Wait till the next bar.
    trading_robot.wait_till_next_bar(last_bar_timestamp=last_bar_timestamp)
