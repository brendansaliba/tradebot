# python 3.x
from configparser import ConfigParser

config = ConfigParser()

# General
# Your own credential are to be placed in here, replacing the "FAKE_CREDENTIAL"
config.add_section('main')
config.set('main', 'CLIENT_ID', 'IQGJLVUPDVWGRLVIAMNPNW3VAHQBLYVZ')
config.set('main', 'REDIRECT_URI', 'https://localhost')
config.set('main', 'ACCOUNT_NUMBER', '71611620')
config.set('main', 'ACCOUNT_ID', '238666280')

# Mac
config.set('main', 'CREDENTIALS_PATH_MAC', r'/Users/brendansaliba/Projects/TradeBot/tradebot_prototype/docs/tokens/token')
config.set('main', 'JSON_PATH_MAC', r'/Users/brendansaliba/Projects/TradeBot/tradebot_prototype/data/DataDump')

# Windows
config.set('main', 'CREDENTIALS_PATH_WIN', r'E:\Projects\tradebot\docs\tokens\token.txt')
config.set('main', 'JSON_PATH_WIN', r'E:\Projects\tradebot\data\DataDump')

with open(file='config.ini', mode='w') as f:
    config.write(f)