# python 3.x
import os
from configparser import ConfigParser

config = ConfigParser()

# Your own credential are to be placed in here, replacing the "FAKE_CREDENTIAL"
config.add_section('main')
config.set('main', 'CLIENT_ID', 'IQGJLVUPDVWGRLVIAMNPNW3VAHQBLYVZ')
config.set('main', 'REDIRECT_URI', 'https://localhost')
config.set('main', 'JSON_PATH', r'/Users/brendansaliba/Projects/TradeBot/tradebot_prototype/data/DataDump')
config.set('main', 'ACCOUNT_NUMBER', '71611620')
config.set('main', 'CREDENTIALS_PATH', r'/Users/brendansaliba/Projects/TradeBot/tradebot_prototype/docs/tokens/refreshtoken_Jun6_2021.txt')
config.set('main', 'ACCOUNT_ID', '426805001')

with open(file='config.ini', mode='w') as f:
    config.write(f)