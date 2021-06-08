# python 3.x
import os
from configparser import ConfigParser

config = ConfigParser()

# Your own credential are to be placed in here, replacing the "FAKE_CREDENTIAL"
config.add_section('main')
config.set('main', 'CLIENT_ID', 'IQGJLVUPDVWGRLVIAMNPNW3VAHQBLYVZ')
config.set('main', 'REDIRECT_URI', 'https://localhost')
config.set('main', 'JSON_PATH', r'E:\Projects\Data Dump\tradebot\token.json')
config.set('main', 'ACCOUNT_NUMBER', '71611620')
config.set('main', 'CREDENTIALS_PATH', r'E:\Projects\TradeBot\python-trading-robot\docs\notes\refreshtoken_Jun6_2021.txt')
config.set('main', 'REFRESH_TOKEN', 'BRj85dK6yK7ogIivpJdHxWIfld2DXuqKUdrEuDG048XHkzGcKLDLeHNHiK/jOBM8cR7LciBA1IsVNVvlqbKxOBtiYEGhLWcfbWbNpaoaaHymAN3fv5Q5vIPBgVHIgBJwR7cxJM8dq9GjcPru5xFBVGSW46q/SZsnsR/64gd6iVVxMJA6/vC4/vu2Zo5xQL1IDeFU+j68/bhgsvmhmL/B8WEMC29byYi9a1wVNHLHvZDN70FgxtZK2rHQ67v+nU0qH2w6LGmujVmFFsNLtnyd9N415Ovxxh93BBkCxxmtIcDabWDnpzN1Xub5fencz0StVDeDKlzZJdk+N1AoLCzsbW8+2pM2TAWJDg/a50DuinsuqALht/EJVGAKV9J7rGZw4BzBZ88aq8Xpo5vYpki0QpJp1+dXmFhnAcTVLcFwnxHOp96QvNyJP5GKnrV100MQuG4LYrgoVi/JHHvlKce3KRp1RieYUXi1vquA5uqcUpxmr49mWwoULAGhHdPY9Ws/66RK0JZ+q77bG59vBut1hXHH85L0Ot22LLfBo1A4EZFbsSIygSTFolpIiLq4121VTKsdmEQfDm7TvU+STAPQI2v33S2XFlwa2VLzqdIEqR1luS9OK0QdATFOFujSLruKtaQu02/EMExecCMJnp1pyJq3MDuvXHAJJQ8dbrA+z6ZbXahHmILA7iyl70WAH5kxU7aU9id9eM+KwhUfKluzM01kGd7uSZm0gcFoXsR4HwlNHImPSdUjwNW8FUgCETOzbEL6AtxkanPlNTMUwZfaXzVUWUhKB25/piW/k9GyKgQwABxSgx6Kr6rwylH6m/m3TG8gveYSMws1a4gBW8wS4lSzvKa0AQbRTpAzdp8U4A+IggWxfx2CxytoY7Jjac/r1IXwmNODjXc=212FD3x19z9sWBHDJACbC00B75E')
config.set('main', 'ACCOUNT_ID', '426805001')

with open(file='config.ini', mode='w') as f:
    config.write(f)