# Filename: functions_dev.py
# Created: 9/21/2021
# Author: Brendan Saliba
# Copyright 2021, Brendan Saliba
# Description:

# SETUP

def get_current_positions(session, portfolio, ACCOUNT_ID):
    account_infos = session.get_accounts(fields=['orders', 'positions'])

    for account_info in account_infos:
        account = account_info['securitiesAccount']
        account_id = account['accountId']

        if account_id == ACCOUNT_ID:
            if 'positions' in account:
                positions = account['positions']

                for position in positions:
                    instrument = position['instrument']
                    average_price = position['averagePrice']
                    quantity = position['longQuantity']
                    asset_type = instrument['assetType']
                    position_symbol = instrument['symbol']

                    portfolio.add_position(symbol=position_symbol,
                                           asset_type=asset_type,
                                           quantity=quantity,
                                           purchase_price=average_price)

            else:
                print('Not currently holding any positions.')

    return portfolio
