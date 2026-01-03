import platform

from configparser import ConfigParser
from cashflow.classes.robot import PyRobot


def setup_func():
    """
    Imports credentials from config and initializes a robot object. Then, the function initializes a session with the TD
    client. A portfolio object within the robot object is also automatically created. Function returns the robot object.
    """
    print('=' * 80)
    print("Beginning setup...")

    CLIENT_ID, REDIRECT_URI, CREDENTIALS_PATH, ACCOUNT_NUMBER, ACCOUNT_ID, JSON_PATH = import_credentials()

    # Initialize the robot object with credentials.
    trading_robot = PyRobot(
        client_id=CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        credentials_path=CREDENTIALS_PATH,
        trading_account=ACCOUNT_NUMBER,
        account_id=ACCOUNT_ID,
        json_path=JSON_PATH
    )

    # Extract TD Session from the robot object
    td_client = trading_robot.session

    # Create a Portfolio
    bot_portfolio = trading_robot.portfolio
    print("Setup complete.")
    print('='*80)

    return trading_robot

def import_credentials():
    """
    Imports credentials from /config/config.ini based on whichever system the script is being run on.
    """

    system = platform.system()
    config = ConfigParser()

    if system == 'Darwin':
        config.read(r'/Users/brendansaliba/Projects/TradeBot/tradebot_prototype/config/config.ini')
        CLIENT_ID = config.get('main', 'CLIENT_ID')
        REDIRECT_URI = config.get('main', 'REDIRECT_URI')
        ACCOUNT_NUMBER = config.get('main', 'ACCOUNT_NUMBER')
        ACCOUNT_ID = config.get('main', 'ACCOUNT_ID')
        CREDENTIALS_PATH = config.get('main', 'CREDENTIALS_PATH_MAC')
        JSON_PATH = config.get('main', 'JSON_PATH_MAC')

    elif system == 'Windows':
        config.read(r'E:\Projects\tradebot\config\config.ini')
        CLIENT_ID = config.get('main', 'CLIENT_ID')
        REDIRECT_URI = config.get('main', 'REDIRECT_URI')
        ACCOUNT_NUMBER = config.get('main', 'ACCOUNT_NUMBER')
        ACCOUNT_ID = config.get('main', 'ACCOUNT_ID')
        CREDENTIALS_PATH = config.get('main', 'CREDENTIALS_PATH_WIN')
        JSON_PATH = config.get('main', 'JSON_PATH_WIN')

    else:
        config.read(r'E:\Projects\tradebot\config\config.ini')
        CLIENT_ID = config.get('main', 'CLIENT_ID')
        REDIRECT_URI = config.get('main', 'REDIRECT_URI')
        ACCOUNT_NUMBER = config.get('main', 'ACCOUNT_NUMBER')
        ACCOUNT_ID = config.get('main', 'ACCOUNT_ID')
        CREDENTIALS_PATH = config.get('main', 'CREDENTIALS_PATH_WIN')
        JSON_PATH = config.get('main', 'JSON_PATH_WIN')

    return CLIENT_ID, REDIRECT_URI, CREDENTIALS_PATH, ACCOUNT_NUMBER, ACCOUNT_ID, JSON_PATH
