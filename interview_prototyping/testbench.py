import pprint
import operator
import pandas as pd

from datetime import datetime
from datetime import timedelta

from interview_prototyping.functions import setup_func
from interview_prototyping.indicators_isaac import Indicators_Isaac

from pyrobot.robot import PyRobot
from pyrobot.indicators import Indicators

symbol = "TSLA"

# Sets up the robot class, robot's portfolio, and the TDSession object
trading_robot, _, TDSession = setup_func()

# Grab the historical prices for the symbol we're trading.
start_date = datetime.today()
end_date = start_date - timedelta(minutes=2)  # previously seconds=5 ???

# print(start_date, ' ', end_date)
#
# historical_prices = trading_robot.grab_historical_prices(
#     TDClient=TDSession,
#     start=end_date,
#     end=start_date,
#     bar_size=1,
#     bar_type='minute',
#     symbols=[symbol]
# )
#
# print(historical_prices)

data = TDSession.get_accounts()
accountId = data[0]['securitiesAccount']['accountId']
print(accountId)


print(type(str(datetime.now())))
