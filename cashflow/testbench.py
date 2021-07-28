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

# for account in accounts_info:
#     account_info = account['securitiesAccount']
#     print(account_info)