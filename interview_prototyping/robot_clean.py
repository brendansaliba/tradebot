from datetime import datetime
from datetime import timedelta

from interview_prototyping.functions import setup_func
from interview_prototyping.indicators_isaac import Indicators_Isaac

symbol = "WEN"

# Sets up the robot class, robot's portfolio, and the TDSession object
trading_robot, _, TDSession = setup_func()

# Grab the historical prices for the symbol we're trading.
end_date = datetime.today()
start_date = end_date - timedelta(minutes=2)  # previously seconds=5 ???

historical_prices = trading_robot.grab_historical_prices(
    TDClient=TDSession,
    start=end_date,
    end=start_date,
    bar_size=1,
    bar_type='minute',
    symbols=[symbol]
)

# Convert data to a Data Frame.
stock_frame = trading_robot.create_stock_frame(
    data=historical_prices['aggregated']
)

# We can also add the stock frame to the Portfolio object.
trading_robot.portfolio.stock_frame = stock_frame

# Additionally the historical prices can be set as well.
trading_robot.portfolio.historical_prices = historical_prices

# Create an indicator Object.
indicator_client = Indicators_Isaac(price_data_frame=stock_frame)

# Indicators
indicator_client.per_of_change()
indicator_client.sma(period=9)
indicator_client.sma(period=50)
indicator_client.sma9_crossed_sma50()
indicator_client.sma(period=200)
indicator_client.sma_volume(period=9)
indicator_client.sma_volume(period=50)
indicator_client.sma_volume(period=200)
indicator_client.abs_9_minus_50_slope()
indicator_client.max_option_chain(TDSession, symbol)
indicator_client.buy_condition(TDSession, symbol)

# Define refresh time so we know when to refresh the TDClient
refresh_time = datetime.now() + timedelta(minutes=21)

while True:
    # update token after 21 minutes
    if datetime.now() > refresh_time:
        TDSession.login()
        print('refresh')
        refresh_time = datetime.now() + timedelta(minutes=21)

    # Trade Now
    # sma_df, signal_list = indicator_client.buy_condition(TDSession, symbol)
    # sma_df = indicator_client.populate_order_data(TDSession, symbol) # test

    stock_df = indicator_client.stock_data
    signal_list = indicator_client.indicator_signal_list

    print(stock_df.tail())

    # Set the StockFrame in the trading robot to the one that is spit out from the indicator client because shitty
    # spaghetti code
    trading_robot.stock_frame = stock_df
    trading_robot.execute_orders_2(TDSession=TDSession, symbol=symbol, signal_list=signal_list)

    # rownum = 0
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    # for jk in sma_df:print(jk)
    # for row in sma_df['open']:
    #     # print(row)
    #     if rownum == len(sma_df['open']):print(sma_df['buy_condition'][rownum],sma_df['new york time'][rownum])
    #     if rownum == len(sma_df['open'])-1: print(sma_df['buy_condition'][rownum], sma_df['new york time'][rownum])
    #     if rownum == len(sma_df['open'])-2: print(sma_df['buy_condition'][rownum], sma_df['new york time'][rownum])
    #     if rownum == len(sma_df['open'])-3: print(sma_df['buy_condition'][rownum], sma_df['new york time'][rownum])
    #     rownum += 1
    # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    # Grab the latest bar.
    latest_bars = trading_robot.get_latest_bar(TDSession=TDSession, symbol=symbol)

    # Add to the Stock Frame.
    stock_frame.add_rows(data=latest_bars)

    # Update the stock frame in the robot and indicator client again
    trading_robot.stock_frame = stock_frame.frame
    indicator_client.stock_data = stock_frame.frame

    # Refresh the Indicators.
    indicator_client.refresh()

    # Check for signals.
    signals = indicator_client.check_signals()

    # Grab the last bar.
    last_bar_timestamp = trading_robot.stock_frame.tail(
        n=1
    ).index.get_level_values(1)

    # Wait till the next bar.
    trading_robot.wait_till_next_bar(last_bar_timestamp=last_bar_timestamp)