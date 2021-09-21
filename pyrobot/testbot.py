# The Stock App
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

# Grab configuration values.
config = ConfigParser()
config.read(r'E:\Projects\tradebot\config\config.ini')

CLIENT_ID = config.get('main', 'CLIENT_ID')
REDIRECT_URI = config.get('main', 'REDIRECT_URI')
CREDENTIALS_PATH = config.get('main', 'CREDENTIALS_PATH_WIN')
ACCOUNT_NUMBER = config.get('main', 'ACCOUNT_ID')

# Initalize the robot.
trading_robot = PyRobot(
    client_id=CLIENT_ID,
    redirect_uri=REDIRECT_URI,
    credentials_path=CREDENTIALS_PATH,
    paper_trading=True
)

print(trading_robot.session.get_accounts())