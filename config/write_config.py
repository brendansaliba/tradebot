# python 3.x
import os
from configparser import ConfigParser

config = ConfigParser()

# Your own credential are to be placed in here, replacing the "FAKE_CREDENTIAL"
config.add_section('main')
config.set('main', 'CLIENT_ID', 'IQGJLVUPDVWGRLVIAMNPNW3VAHQBLYVZ')
config.set('main', 'REDIRECT_URI', 'http://127.0.0.1')
config.set('main', 'JSON_PATH', r'E:\Projects\Data Dump\tradebot\test.json')
config.set('main', 'ACCOUNT_NUMBER', '71611620')

with open(file='config.ini', mode='w') as f:
    config.write(f)