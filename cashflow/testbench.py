import pprint
import operator
import pandas as pd
import pickle
import sys
import platform
import os
import random
from datetime import datetime
from functions import setup_func

symbol = "TSLA"

trading_robot, TDClient = setup_func()

accounts_info = TDClient.get_accounts(account='all', fields=['orders', 'positions'])

all_symbols = []

for account in accounts_info:
    account_info = account['securitiesAccount']
    print(account_info['accountId'])
    if 'positions' in account_info:
        positions = account_info['positions']
        print(positions)
    else:
        print('No positions')
