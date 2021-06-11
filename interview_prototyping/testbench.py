import pprint
import operator
import pandas as pd
import pickle

from datetime import datetime
from datetime import timedelta

from interview_prototyping.functions import setup_func
from interview_prototyping.indicators_isaac import Indicators_Isaac

from pyrobot.robot import PyRobot
from pyrobot.indicators import Indicators

symbol = "TSLA"

# Sets up the robot class, robot's portfolio, and the TDSession object
# trading_robot, _, TDSession = setup_func()
#
# accounts = TDSession.get_accounts()
#
# account_id = trading_robot.account_id
#
# transactions_info = TDSession.get_orders(account_id)
#
# print(accounts[0]['securitiesAccount']['type'], accounts[0]['securitiesAccount']['accountId'])
# print(accounts[1]['securitiesAccount']['type'], accounts[1]['securitiesAccount']['accountId'])

# print(trading_robot.portfolio.positions)

testdict = {}

testdict[symbol] = {}
print(testdict)
