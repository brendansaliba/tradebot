from configparser import ConfigParser
from interview_prototyping.robot import PyRobot


def setup_func():
    # Get credentials
    CLIENT_ID, REDIRECT_URI, CREDENTIALS_PATH, ACCOUNT_NUMBER, ACCOUNT_ID = import_credentials()

    # Initalize the robot with my credentials.
    trading_robot = PyRobot(
        client_id=CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        credentials_path=CREDENTIALS_PATH,
        trading_account=ACCOUNT_NUMBER,
        account_id=ACCOUNT_ID
        # paper_trading=True
    )
    print("Bot created.")

    # Isaac's stuff
    # trading_robot = PyRobot(
    #     client_id="XTDX2KUZV4EY2JIWX8TRTUVT9WGYOABN",
    #     redirect_uri="https://localhost",
    #     credentials_path=r'C:\Users\Isaac\Desktop\DESKTOP\Stocktraderclass.com\AlexReedGitHub\td-ameritrade-python-api-master\token.txt',
    #     trading_account='865852744',
    #     paper_trading=True
    # )

    # Create TDSession
    td_client = trading_robot._create_session()
    print("Session created.")

    # Create a Portfolio
    trading_robot_portfolio = trading_robot.create_portfolio()
    print("Portfolio created.")

    return trading_robot, trading_robot_portfolio, td_client


def import_credentials():
    # Grab configuration values.
    config = ConfigParser()
    config.read(r'/Users/brendansaliba/Projects/TradeBot/tradebot_prototype/config/config.ini')

    CLIENT_ID = config.get('main', 'CLIENT_ID')
    REDIRECT_URI = config.get('main', 'REDIRECT_URI')
    CREDENTIALS_PATH = config.get('main', 'CREDENTIALS_PATH')
    ACCOUNT_NUMBER = config.get('main', 'ACCOUNT_NUMBER')
    ACCOUNT_ID = config.get('main', 'ACCOUNT_ID')

    return CLIENT_ID, REDIRECT_URI, CREDENTIALS_PATH, ACCOUNT_NUMBER, ACCOUNT_ID
