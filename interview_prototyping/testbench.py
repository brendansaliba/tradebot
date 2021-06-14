import pprint
import operator
import pandas as pd
import pickle
import sys
import platform
import os
import random
from datetime import datetime

possible_choices = ['chicken pot pie', 'quiche', 'chicken soup dumplings',
                    'sauteed spinach w/ meat', 'pasta frozen from trader joes with pink sauce']

random_num = random.randrange(0, len(possible_choices)-1, 1)
print('you should eat', possible_choices[random_num])

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

now = datetime.now().strftime("%Y_%m_%d-%I%M_%p")
print(now)

filename = "{}_run_{}".format(symbol, now)

print(filename)