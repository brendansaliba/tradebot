import platform

from configparser import ConfigParser
from interview_prototyping.robot import PyRobot


def setup_func():
    # Get credentials
    CLIENT_ID, REDIRECT_URI, CREDENTIALS_PATH, ACCOUNT_NUMBER, ACCOUNT_ID, JSON_PATH = import_credentials()

    # Initalize the robot with my credentials.
    trading_robot = PyRobot(
        client_id=CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        credentials_path=CREDENTIALS_PATH,
        trading_account=ACCOUNT_NUMBER,
        account_id=ACCOUNT_ID,
        json_path=JSON_PATH
    )
    print("Bot created.")

    # Create TDSession
    # td_client = trading_robot._create_session()
    td_client = trading_robot.session
    print("Session created.")

    # Create a Portfolio
    trading_robot_portfolio = trading_robot.create_portfolio()
    print("Portfolio created.")
    print('Trading with account:', ACCOUNT_ID)
    print('='*80)

    return trading_robot, td_client


def import_credentials():
    system = platform.system()
    config = ConfigParser()

    if system == 'Darwin':
        # Grab configuration values.
        config.read(r'/Users/brendansaliba/Projects/TradeBot/tradebot_prototype/config/config.ini')
        CLIENT_ID = config.get('main', 'CLIENT_ID')
        REDIRECT_URI = config.get('main', 'REDIRECT_URI')
        ACCOUNT_NUMBER = config.get('main', 'ACCOUNT_NUMBER')
        ACCOUNT_ID = config.get('main', 'ACCOUNT_ID')
        CREDENTIALS_PATH = config.get('main', 'CREDENTIALS_PATH_MAC')
        JSON_PATH = config.get('main', 'JSON_PATH_MAC')

    elif system == 'Windows':
        config.read(r'E:\Projects\TradeBot\python-trading-robot\config\config.ini')
        CLIENT_ID = config.get('main', 'CLIENT_ID')
        REDIRECT_URI = config.get('main', 'REDIRECT_URI')
        ACCOUNT_NUMBER = config.get('main', 'ACCOUNT_NUMBER')
        ACCOUNT_ID = config.get('main', 'ACCOUNT_ID')
        CREDENTIALS_PATH = config.get('main', 'CREDENTIALS_PATH_WIN')
        JSON_PATH = config.get('main', 'JSON_PATH_WIN')

    else:
        config.read(r'E:\Projects\TradeBot\python-trading-robot\config\config.ini')
        CLIENT_ID = config.get('main', 'CLIENT_ID')
        REDIRECT_URI = config.get('main', 'REDIRECT_URI')
        ACCOUNT_NUMBER = config.get('main', 'ACCOUNT_NUMBER')
        ACCOUNT_ID = config.get('main', 'ACCOUNT_ID')
        CREDENTIALS_PATH = config.get('main', 'CREDENTIALS_PATH_WIN')
        JSON_PATH = config.get('main', 'JSON_PATH_WIN')

    return CLIENT_ID, REDIRECT_URI, CREDENTIALS_PATH, ACCOUNT_NUMBER, ACCOUNT_ID, JSON_PATH
