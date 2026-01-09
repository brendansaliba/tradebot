# Filename: bot.py
# Created: 9/21/2021
# Author: Brendan Saliba
# Copyright 2021, Brendan Saliba
# Description:

# SETUP
import pprint
import operator
import os
from dotenv import load_dotenv

from datetime import datetime
from datetime import timedelta
from configparser import ConfigParser

from pyrobot.robot import Robot
from pyrobot.indicators import Indicators

from functions_dev import get_current_positions

load_dotenv()

# Grab configuration values.
# dirname = os.path.dirname(__file__)
# config_path = os.path.join(dirname, 'config/config.ini')
# config = ConfigParser()
# config.read(config_path)

# SYMBOL = config.get('main', 'symbol')

app_key = os.getenv("SCHWAB_APP_KEY")
app_secret = os.getenv("SCHWAB_APP_SECRET")
refresh_token = os.getenv("REFRESH_TOKEN")

bot = Robot(
    app_key=app_key,
    app_secret=app_secret,
    refresh_token=refresh_token,
    credentials_path=None,
)

bot.client.get_schwab_equity_session_hours()
bot.client.get_accounts()

# Grab historical prices, first define the start date and end date.
# end = datetime.today()
# start = end - timedelta(days=10)

# Grab the historical prices.
# historical_prices = bot.grab_historical_prices(
#     start=start,
#     end=end,
#     bar_size=1,
#     bar_type='minute',
#     symbols=[SYMBOL]
# )

# Convert data to a Data Frame.
# stock_frame = bot.create_stock_frame(
#     data=historical_prices['aggregated']
# )


# We can also add the stock frame to the Portfolio object.
# bot.create_portfolio()
# bot.portfolio.stock_frame = stock_frame
# bot.portfolio.historical_prices = historical_prices
# bot.get_current_positions()
# bot.create_indicator_client()

# Add indicators
# bot.indicator.percent_change()
# bot.indicator.sma(period=9)
# bot.indicator.sma(period=50)
# bot.indicator.sma(period=200)
# bot.indicator.volume_avg(period=9)
# bot.indicator.volume_avg(period=50)
# bot.indicator.volume_avg(period=200)
# bot.indicator.sma9_crossed_sma50()
# bot.indicator.abs_9_minus_50_slope()
# bot.indicator.max_option_chain(symbol=SYMBOL)


# Define initial refresh time so we know when to refresh the TDClient
# refresh_time = datetime.now() + timedelta(minutes=21)

# while True:
    # Update token after 21 minutes
    # if datetime.now() > refresh_time:
    #     bot.session.login()
    #     print('refresh')
    #     refresh_time = datetime.now() + timedelta(minutes=21)

    # Grab the latest bar.
    # latest_bars = bot.get_latest_bar()

    # Add to the Stock Frame.
    # stock_frame.add_rows(data=latest_bars)
    # bot.stock_frame = stock_frame

    # Refresh the Indicators.
    # bot.indicator.refresh()

    # print("="*50)
    # print("Current StockFrame")
    # print("-"*50)
    # print(stock_frame.frame.tail())
    # print("-"*50)
    # print("")

    # Check for signals.
    # signals = indicator_client.check_signals()

    # Grab the last bar.
    # last_bar_timestamp = bot.stock_frame.frame.tail(n=1).index.get_level_values(1)

    # Wait till the next bar.
    # bot.wait_till_next_bar(last_bar_timestamp=last_bar_timestamp)
