import pprint
import threading
import time

from datetime import datetime
from datetime import timedelta
from configparser import ConfigParser

from pyrobot.robot import PyRobot
from pyrobot.indicators import Indicators
from td.client import TDClient
from functions import main
from indicators import Indicators
from robot import PyRobot
from portfolio import Portfolio


def main():
    # Initalize the robot.
    trading_robot = PyRobot(
        client_id="XTDX2KUZV4EY2JIWX8TRTUVT9WGYOABN",
        redirect_uri="https://localhost",
        credentials_path=r'C:\Users\Isaac\Desktop\DESKTOP\Stocktraderclass.com\AlexReedGitHub\td-ameritrade-python-api-master\token.txt',
        trading_account='865852744',
        paper_trading=True
    )

    td_client = trading_robot._create_session()

    # Create a Portfolio
    trading_robot_portfolio = trading_robot.create_portfolio()

    return trading_robot, trading_robot_portfolio