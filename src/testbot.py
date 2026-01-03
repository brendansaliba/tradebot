# Filename: testbot.py
# Created: 9/21/2021
# Author: Brendan Saliba
# Copyright 2021, Brendan Saliba
# Description:

# SETUP
import pprint
import operator

from datetime import datetime
from datetime import timedelta
from configparser import ConfigParser

from pyrobot.classes.robot import PyRobot
from pyrobot.classes.indicators import Indicators

from functions_dev import get_current_positions

# Grab configuration values.
config = ConfigParser()
config.read(r'E:\Projects\tradebot\config\config.ini')

CLIENT_ID = config.get('main', 'CLIENT_ID')
ACCOUNT_ID = config.get('main', 'ACCOUNT_ID')
REDIRECT_URI = config.get('main', 'REDIRECT_URI')
CREDENTIALS_PATH = config.get('main', 'CREDENTIALS_PATH_WIN')
ACCOUNT_NUMBER = config.get('main', 'ACCOUNT_ID')

# Initalize the robot.
trading_robot = PyRobot(
    client_id=CLIENT_ID,
    account_id=ACCOUNT_ID,
    redirect_uri=REDIRECT_URI,
    credentials_path=CREDENTIALS_PATH,
)



TRADING_SYMBOL = "SPY"

# Grab historical prices, first define the start date and end date.
start_date = datetime.today()
end_date = start_date - timedelta(days=10)

# Grab the historical prices.
historical_prices = trading_robot.grab_historical_prices(
    start=end_date,
    end=start_date,
    bar_size=1,
    bar_type='minute',
    symbols=[TRADING_SYMBOL]
)

# Convert data to a Data Frame.
stock_frame = trading_robot.create_stock_frame(
    data=historical_prices['aggregated']
)

print(stock_frame.frame)

# We can also add the stock frame to the Portfolio object.
trading_robot.create_portfolio()
trading_robot.portfolio.stock_frame = stock_frame
trading_robot.portfolio.historical_prices = historical_prices
trading_robot.get_current_positions()
trading_robot.create_indicator_client()

# ASDF
trading_robot.indicator.percent_change()
trading_robot.indicator.sma(period=9)
trading_robot.indicator.sma(period=50)
trading_robot.indicator.sma(period=200)
trading_robot.indicator.volume_avg(period=9)
trading_robot.indicator.volume_avg(period=50)
trading_robot.indicator.volume_avg(period=200)
trading_robot.indicator.sma9_crossed_sma50()
trading_robot.indicator.abs_9_minus_50_slope()
trading_robot.indicator.max_option_chain(symbol=TRADING_SYMBOL)


# Define initial refresh time so we know when to refresh the TDClient
refresh_time = datetime.now() + timedelta(minutes=21)

while True:
    # Update token after 21 minutes
    if datetime.now() > refresh_time:
        trading_robot.session.login()
        print('refresh')
        refresh_time = datetime.now() + timedelta(minutes=21)

    # Grab the latest bar.
    latest_bars = trading_robot.get_latest_bar()

    # Add to the Stock Frame.
    stock_frame.add_rows(data=latest_bars)
    trading_robot.stock_frame = stock_frame

    # Refresh the Indicators.
    trading_robot.indicator.refresh()

    print("="*50)
    print("Current StockFrame")
    print("-"*50)
    print(stock_frame.frame.tail())
    print("-"*50)
    print("")

    # Check for signals.
    # signals = indicator_client.check_signals()

    # Grab the last bar.
    last_bar_timestamp = trading_robot.stock_frame.frame.tail(n=1).index.get_level_values(1)

    # Wait till the next bar.
    trading_robot.wait_till_next_bar(last_bar_timestamp=last_bar_timestamp)
