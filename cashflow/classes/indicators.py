import numpy as np
import pandas as pd
import operator
import math
import json
import datetime

from typing import Any
from typing import Dict
from typing import Union
from typing import Optional
from typing import Tuple
from cashflow.classes.stock_frame import StockFrame
from cashflow.classes.portfolio import Portfolio

from td.client import TDClient
from td.utils import TDUtilities

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)


class Indicators():
    """
    Represents an Indicator Object which can be used
    to easily add technical indicators to a StockFrame.
    """

    def __init__(self, price_data_frame: StockFrame) -> None:
        """Initalizes the Indicator Client.

        Arguments:
        ----
        price_data_frame {pyrobot.StockFrame} -- The price data frame which is used to add indicators to.
            At a minimum this data frame must have the following columns: `['timestamp','close','open','high','low']`.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.price_data_frame
        """

        self._stock_frame: StockFrame = price_data_frame
        # change this price_groups = self._stock_frame.symbol_groups
        self._price_groups = price_data_frame.symbol_groups
        self._current_indicators = {}
        self._indicator_signals = {}

        self.session: TDClient = None
        self.stock_data = None
        self.indicator_signal_list = []
        self.calls_options = []
        self.puts_options = []

        # TODO: use Alex's add_rows() function instead of updating whole dataframe
        self._frame = self._stock_frame.frame

        # add by nikhil for storing buy count
        self.buy_count = 0

        self._indicators_comp_key = []
        self._indicators_key = []

        if self.is_multi_index:
            True

    def get_indicator_signal(self, indicator: str = None) -> Dict:
        """Return the raw Pandas Dataframe Object.

        Arguments:
        ----
        indicator {Optional[str]} -- The indicator key, for example `ema` or `sma`.

        Returns:
        ----
        {dict} -- Either all of the indicators or the specified indicator.
        """

        if indicator and indicator in self._indicator_signals:
            return self._indicator_signals[indicator]
        else:
            return self._indicator_signals

    def set_indicator_signal(self, indicator: str, buy: float, sell: float, condition_buy: Any, condition_sell: Any,
                             buy_max: float = None, sell_max: float = None, condition_buy_max: Any = None,
                             condition_sell_max: Any = None) -> None:
        """Used to set an indicator where one indicator crosses above or below a certain numerical threshold.

        Arguments:
        ----
        indicator {str} -- The indicator key, for example `ema` or `sma`.

        buy {float} -- The buy signal threshold for the indicator.

        sell {float} -- The sell signal threshold for the indicator.

        condition_buy {str} -- The operator which is used to evaluate the `buy` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`.

        condition_sell {str} -- The operator which is used to evaluate the `sell` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`.

        buy_max {float} -- If the buy threshold has a maximum value that needs to be set, then set the `buy_max` threshold.
            This means if the signal exceeds this amount it WILL NOT PURCHASE THE INSTRUMENT. (defaults to None).

        sell_max {float} -- If the sell threshold has a maximum value that needs to be set, then set the `buy_max` threshold.
            This means if the signal exceeds this amount it WILL NOT SELL THE INSTRUMENT. (defaults to None).

        condition_buy_max {str} -- The operator which is used to evaluate the `buy_max` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`. (defaults to None).

        condition_sell_max {str} -- The operator which is used to evaluate the `sell_max` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`. (defaults to None).
        """

        # Add the key if it doesn't exist.
        if indicator not in self._indicator_signals:
            self._indicator_signals[indicator] = {}
            self._indicators_key.append(indicator)

            # Add the signals.
        self._indicator_signals[indicator]['buy'] = buy
        self._indicator_signals[indicator]['sell'] = sell
        self._indicator_signals[indicator]['buy_operator'] = condition_buy
        self._indicator_signals[indicator]['sell_operator'] = condition_sell

        # Add the max signals
        self._indicator_signals[indicator]['buy_max'] = buy_max
        self._indicator_signals[indicator]['sell_max'] = sell_max
        self._indicator_signals[indicator]['buy_operator_max'] = condition_buy_max
        self._indicator_signals[indicator]['sell_operator_max'] = condition_sell_max

    def set_indicator_signal_compare(self, indicator_1: str, indicator_2: str, condition_buy: Any,
                                     condition_sell: Any) -> None:
        """Used to set an indicator where one indicator is compared to another indicator.

        Overview:
        ----
        Some trading strategies depend on comparing one indicator to another indicator.
        For example, the Simple Moving Average crossing above or below the Exponential
        Moving Average. This will be used to help build those strategies that depend
        on this type of structure.

        Arguments:
        ----
        indicator_1 {str} -- The first indicator key, for example `ema` or `sma`.

        indicator_2 {str} -- The second indicator key, this is the indicator we will compare to. For example,
            is the `sma` greater than the `ema`.

        condition_buy {str} -- The operator which is used to evaluate the `buy` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`.

        condition_sell {str} -- The operator which is used to evaluate the `sell` condition. For example, `">"` would
            represent greater than or from the `operator` module it would represent `operator.gt`.
        """

        # Define the key.
        key = "{ind_1}_comp_{ind_2}".format(
            ind_1=indicator_1,
            ind_2=indicator_2
        )

        # Add the key if it doesn't exist.
        if key not in self._indicator_signals:
            self._indicator_signals[key] = {}
            self._indicators_comp_key.append(key)

            # Grab the dictionary.
        indicator_dict = self._indicator_signals[key]

        # Add the signals.
        indicator_dict['type'] = 'comparison'
        indicator_dict['indicator_1'] = indicator_1
        indicator_dict['indicator_2'] = indicator_2
        indicator_dict['buy_operator'] = condition_buy
        indicator_dict['sell_operator'] = condition_sell

    @property
    def price_data_frame(self) -> pd.DataFrame:
        """Return the raw Pandas Dataframe Object.

        Returns:
        ----
        {pd.DataFrame} -- A multi-index data frame.
        """

        return self._frame

    @price_data_frame.setter
    def price_data_frame(self, price_data_frame: pd.DataFrame) -> None:
        """Sets the price data frame.

        Arguments:
        ----
        price_data_frame {pd.DataFrame} -- A multi-index data frame.
        """

        self._frame = price_data_frame

    @property
    def is_multi_index(self) -> bool:
        """Specifies whether the data frame is a multi-index dataframe.

        Returns:
        ----
        {bool} -- `True` if the data frame is a `pd.MultiIndex` object. `False` otherwise.
        """

        if isinstance(self._frame.index, pd.MultiIndex):
            return True
        else:
            return False

    def change_in_price(self, column_name: str = 'change_in_price') -> pd.DataFrame:
        """Calculates the Change in Price.

        Returns:
        ----
        {pd.DataFrame} -- A data frame with the Change in Price included.
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.change_in_price

        self._frame[column_name] = self._price_groups['close'].transform(
            lambda x: x.diff()
        )

        return self._frame

    def rsi(self, period: int, method: str = 'wilders', column_name: str = 'rsi') -> pd.DataFrame:
        """Calculates the Relative Strength Index (RSI).

        Arguments:
        ----
        period {int} -- The number of periods to use to calculate the RSI.

        Keyword Arguments:
        ----
        method {str} -- The calculation methodology. (default: {'wilders'})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the RSI indicator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.rsi(period=14)
            >>> price_data_frame = inidcator_client.price_data_frame
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.rsi

        # First calculate the Change in Price.
        if 'change_in_price' not in self._frame.columns:
            self.change_in_price()

        # Define the up days.
        self._frame['up_day'] = self._price_groups['change_in_price'].transform(
            lambda x: np.where(x >= 0, x, 0)
        )

        # Define the down days.
        self._frame['down_day'] = self._price_groups['change_in_price'].transform(
            lambda x: np.where(x < 0, x.abs(), 0)
        )

        # Calculate the EWMA for the Up days.
        self._frame['ewma_up'] = self._price_groups['up_day'].transform(
            lambda x: x.ewm(span=period).mean()
        )

        # Calculate the EWMA for the Down days.
        self._frame['ewma_down'] = self._price_groups['down_day'].transform(
            lambda x: x.ewm(span=period).mean()
        )

        # Calculate the Relative Strength
        relative_strength = self._frame['ewma_up'] / self._frame['ewma_down']

        # Calculate the Relative Strength Index
        relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))

        # Add the info to the data frame.
        self._frame['rsi'] = np.where(relative_strength_index == 0, 100, 100 - (100 / (1 + relative_strength_index)))

        # Clean up before sending back.
        self._frame.drop(
            labels=['ewma_up', 'ewma_down', 'down_day', 'up_day', 'change_in_price'],
            axis=1,
            inplace=True
        )

        return self._frame

    def abs_9_minus_50_slope(self, column_name: str = 'abs_9_minus_50_slope'):
        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.abs_9_minus_50_slope

        # grab sma_9 and sma_50
        sma_9 = self._frame["sma_9"]
        sma_50 = self._frame["sma_50"]

        # calculate abs_diff
        temp_sma_9_minus_50 = []
        for val_9, val_50 in zip(sma_9, sma_50):
            if math.isnan(val_9) or math.isnan(val_50):
                temp_sma_9_minus_50.append(0)
            else:
                temp_sma_9_minus_50.append(round(abs(val_9 - val_50) * 0.0174533 * 1000, 4)) # do these dudes know basic math what

        self._frame[column_name] = pd.Series(temp_sma_9_minus_50).values

        return self._frame

    def sma(self, period: int, column_name: str = 'sma') -> pd.DataFrame:
        """Calculates the Simple Moving Average (SMA).

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating the SMA.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the SMA indicator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.sma(period=100)
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name + '_' + str(period)] = {}
        self._current_indicators[column_name + '_' + str(period)]['args'] = locals_data
        self._current_indicators[column_name + '_' + str(period)]['func'] = self.sma

        # print(self.price_data_frame)
        # Add the SMA
        self._frame[column_name + '_' + str(period)] = self._price_groups['close'].transform(
            lambda x: x.rolling(window=period).mean()
        )

        # nikhil modified
        # adding logic for sma 9 slope
        index_count = 1
        prev = 0
        next = 0
        temp_list = []
        for index, row in self._frame.iterrows():

            if index_count == 1:
                prev = next = row[column_name + '_' + str(period)]
                temp_list.append(0)

            else:
                prev = next
                next = row[column_name + '_' + str(period)]

                if math.isnan(prev) or math.isnan(next):
                    temp_list.append(0)
                else:
                    temp_list.append(round(((next - prev) * 0.0174533) * 10000, 4))

            # print(prev, next)
            index_count += 1
        self._frame[column_name + '_' + str(period) + '_slope'] = pd.Series(temp_list).values
        # print(row[column_name])

        # print(self._frame)
        return self._frame

    def sma_volume(self, period: int, column_name: str = 'sma_volume') -> pd.DataFrame:
        """Calculates the Simple Moving Average (SMA).

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating the SMA.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the SMA indicator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.sma(period=100)
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name + '_' + str(period)] = {}
        self._current_indicators[column_name + '_' + str(period)]['args'] = locals_data
        self._current_indicators[column_name + '_' + str(period)]['func'] = self.sma_volume

        # print(self.price_data_frame)
        # Add the SMA
        self._frame[column_name + '_' + str(period)] = self._price_groups['volume'].transform(
            lambda x: x.rolling(window=period).mean()
        )

        # nikhil modified
        # adding logic for sma 9 slope
        index_count = 1
        prev = 0
        next = 0
        temp_list = []
        for index, row in self._frame.iterrows():

            if index_count == 1:
                prev = next = row[column_name + '_' + str(period)]
                temp_list.append(0)

            else:
                prev = next
                next = row[column_name + '_' + str(period)]

                if math.isnan(prev) or math.isnan(next):
                    temp_list.append(0)
                else:
                    temp_list.append(round(((next - prev) * 0.00174533), 4))

            # print(prev, next)
            index_count += 1

        self._frame[column_name + '_' + str(period) + '_slope'] = pd.Series(temp_list).values
        # print(row[column_name])

        # print(self._frame)
        return self._frame

    def per_of_change(self) -> pd.DataFrame:
        locals_data = locals()
        del locals_data['self']

        self._current_indicators['per_of_change'] = {}
        self._current_indicators['per_of_change']['args'] = locals_data
        self._current_indicators['per_of_change']['func'] = self.per_of_change

        # calculate per of change
        per_of_change = []
        count = 0

        while count < len(self._frame):
            if count == 0:
                per_of_change.append(0)
                prev = next = self._frame["close"][count]
            else:
                prev = next
                next = self._frame["close"][count]
                per_of_change.append(round(((next - prev) / prev) * 100, 3))
            count += 1

        self._frame['per_of_change'] = pd.Series(per_of_change).values

        return self._frame

    def vwap(self, column_name: str = 'vwap') -> pd.DataFrame:
        """Calculates the Volume Weighted Adjusted Price (VWAP).

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the vwap indicator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.vwap()
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.vwap

        # rint(self.price_data_frame)
        loop_count = int(len(self._frame) / 2) - 1

        self._frame['typical_mult_volume'] = ((self._frame["low"] + self._frame["close"] + self._frame["high"]) / 3) * \
                                             self._frame["volume"]

        # print(self._frame)
        mantain_index = 0
        temp_list = []
        # print(self._frame["typical_mult_volume"][mantain_index:mantain_index+2].values.mean())
        # print(self._frame["volume"][mantain_index:mantain_index+2].values.mean())
        for i in range(len(self._frame) - 1):
            # print(self._frame["typical_mult_volume"][mantain_index:mantain_index + 2].values)
            # print(self._frame["volume"][mantain_index:mantain_index + 2].values)
            temp_list.append(
                round(self._frame["typical_mult_volume"][mantain_index:mantain_index + 2].values.mean(), 3) /
                round(self._frame["volume"][mantain_index:mantain_index + 2].values.mean(), 3))
            mantain_index = mantain_index + 1

        # adding zero on last remaing cell
        len_new_list = len(temp_list)
        while len_new_list != len(self._frame):
            temp_list.append(0)
            len_new_list += 1

        # print(len(temp_list), len(self._frame))
        self._frame[column_name] = pd.Series(temp_list).values

        # nikhil modified
        # adding logic for sma 9 slope
        index_count = 1
        prev = 0
        next = 0
        temp_list = []
        for index, row in self._frame.iterrows():

            if index_count == 1:
                prev = next = row[column_name]
                temp_list.append(0)

            else:
                prev = next
                next = row[column_name]

                if math.isnan(prev) or math.isnan(next):
                    temp_list.append(0)
                else:
                    temp_list.append(round(((next - prev) * 0.174533) * 100, 4))

            # print(prev, next)
            index_count += 1
        self._frame[column_name + '_' + '_slope'] = pd.Series(temp_list).values

        return self._frame

    def reformat_symbol(self, symbol):
        """Returns the contract identifier from the symbol"""
        symbol_new_format_list = symbol.split('_')
        symbol_new_format = '.' + symbol_new_format_list[0] + symbol_new_format_list[1][4:6] + \
                            symbol_new_format_list[1][0:2] + symbol_new_format_list[1][2:4] + \
                            symbol_new_format_list[1][6:]
        return symbol_new_format

    def max_option_chain(self, symbol):
        """Returns near the money (NTM) options on the call and put side that are highest volume.

          Arguments:
          ----

          Returns:
          ----

          Usage:
          ----
              >>> historical_prices_df = trading_robot.grab_historical_prices(
                  start=start_date,
                  end=end_date,
                  bar_size=1,
                  bar_type='minute'
              )
              >>> price_data_frame = pd.DataFrame(data=historical_prices)
              >>> indicator_client = Indicators(price_data_frame=price_data_frame)
              >>> indicator_client.sma(period=100)
          """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators['calls_option'] = {}
        self._current_indicators['calls_option']['args'] = locals_data
        self._current_indicators['calls_option']['func'] = self.max_option_chain

        params = {
            "symbol": symbol,
            "range": "NTM"
        }

        watchlist_info = self.session.get_options_chain(option_chain=params)

        options_list_calls = []
        options_list_calls_total_volume = []
        temp_symbol_list = []

        # Level One - Option. .WMT210312C131 to convert it WMT_031221C131
        # Loop through the watchlist_info (which is actually the options chain information) and create a temp dictionary
        for call_key, call_value in watchlist_info.get("callExpDateMap", {}).items():
            for candle_key, candle_value in watchlist_info.get("callExpDateMap", {}).get(call_key).items():
                temp_d = {
                    "symbol": watchlist_info.get("callExpDateMap", {}).get(call_key).get(candle_key, [{}])[0].get(
                        "symbol"),
                    "total_volume": watchlist_info.get("callExpDateMap", {}).get(call_key).get(candle_key, [{}])[0].get(
                        "totalVolume"),
                    "percent_change":
                        watchlist_info.get("callExpDateMap", {}).get(call_key).get(candle_key, [{}])[0].get(
                            "percentChange")
                }
                options_list_calls.append(temp_d)

        # Get the calls with max volume
        max_volume_calls_index = max(range(len(options_list_calls)),
                                     key=lambda index: options_list_calls[index]['total_volume'])

        # Reformat the options
        symbol_new_format = options_list_calls[max_volume_calls_index].get("symbol")
        symbol_new_format_list = symbol_new_format.split('_')
        symbol_new_format = '.' + symbol_new_format_list[0] + symbol_new_format_list[1][4:6] + \
                            symbol_new_format_list[1][0:2] + symbol_new_format_list[1][2:4] + \
                            symbol_new_format_list[1][6:]

        for i in range(len(self._frame)):
            temp_symbol_list.append(options_list_calls[max_volume_calls_index].get("symbol"))
        self._frame['calls_option'] = pd.Series(temp_symbol_list).values
        self.calls_options = temp_symbol_list

        temp_symbol_list = []
        for i in range(len(self._frame)):
            temp_symbol_list.append(symbol_new_format)
        self._frame['calls_option_format'] = pd.Series(temp_symbol_list).values

        temp_symbol_list = []
        for i in range(len(self._frame)):
            temp_symbol_list.append(options_list_calls[max_volume_calls_index].get("percent_change"))
        self._frame['percent_change_calls'] = pd.Series(temp_symbol_list).values

        temp_symbol_list = []
        for i in range(len(self._frame)):
            temp_symbol_list.append(options_list_calls[max_volume_calls_index].get("total_volume"))
        self._frame['total_volume_calls'] = pd.Series(temp_symbol_list).values

        # make put columns
        options_list_puts = []
        options_list_puts_total_volume = []
        for call_key, call_value in watchlist_info.get("putExpDateMap", {}).items():
            for candle_key, candle_value in watchlist_info.get("putExpDateMap", {}).get(call_key).items():
                temp_d = {
                    "symbol": watchlist_info.get("putExpDateMap", {}).get(call_key).get(candle_key, [{}])[0].get(
                        "symbol"),
                    "total_volume": watchlist_info.get("putExpDateMap", {}).get(call_key).get(candle_key, [{}])[0].get(
                        "totalVolume"),
                    "percent_change": watchlist_info.get("putExpDateMap", {}).get(call_key).get(candle_key, [{}])[
                        0].get(
                        "percentChange"),
                }
                options_list_puts.append(temp_d)
                options_list_puts_total_volume.append(
                    watchlist_info.get("putExpDateMap", {}).get(call_key).get(candle_key, [{}])[0].get("totalVolume"))

        max_volume_puts_index = max(range(len(options_list_puts)),
                                    key=lambda index: options_list_puts[index]['total_volume'])
        symbol_new_format = options_list_puts[max_volume_puts_index].get("symbol")
        symbol_new_format_list = symbol_new_format.split('_')
        symbol_new_format = '.' + symbol_new_format_list[0] + symbol_new_format_list[1][4:6] + \
                            symbol_new_format_list[1][0:2] + symbol_new_format_list[1][2:4] + \
                            symbol_new_format_list[1][6:]

        temp_symbol_list = []
        for i in range(len(self._frame)):
            temp_symbol_list.append(options_list_puts[max_volume_puts_index].get("symbol"))
        self._frame['puts_option'] = pd.Series(temp_symbol_list).values
        self.puts_options = temp_symbol_list

        temp_symbol_list = []
        for i in range(len(self._frame)):
            temp_symbol_list.append(symbol_new_format)
        self._frame['puts_option_format'] = pd.Series(temp_symbol_list).values

        temp_symbol_list = []
        for i in range(len(self._frame)):
            temp_symbol_list.append(options_list_puts[max_volume_puts_index].get("percent_change"))
        self._frame['percent_change_puts'] = pd.Series(temp_symbol_list).values

        temp_symbol_list = []
        for i in range(len(self._frame)):
            temp_symbol_list.append(options_list_puts[max_volume_puts_index].get("total_volume"))
        self._frame['total_volume_puts'] = pd.Series(temp_symbol_list).values

        return self._frame

    def populate_order_data(self, symbol, account_id):
        """Populates order data columns in dataframe"""

        # Get orders (Which return list of order did in past)
        transactions_info = self.session.get_orders(account=account_id)

        for order in transactions_info:
            print(order)

            entered_time = order['enteredTime']
            status = order['status']
            order_id = order['orderId']
            order_leg = order['orderLegCollection'][0]
            contract_symbol = order_leg['instrument']['symbol']
            price = "N/A"
            quantity = order['quantity']
            filled_quantity = order['filledQuantity']
            remaining_quantity = order['remainingQuantity']
            converted_date = pd.to_datetime(entered_time).strftime("%Y-%m-%d %H:%M:%S")
            datetime = pd.to_datetime(entered_time).strftime("%Y-%m-%d %H:%M:%S")  # convert to Timestamp object

            try:
                order_activity = order['orderActivityCollection'][0]
                execution_leg = order_activity['executionLegs'][0]
                price = execution_leg['price']

            except:
                print("No Activity Collection in order info.")

            df_dict = {
                'symbol': [symbol],
                'datetime': [datetime],
                'enteredTime': [entered_time],
                'status': [status],
                'orderId': [order_id],
                'contract_symbol': [contract_symbol],
                'price': [price],
                'quantity': [quantity],
                'filledQuantity': [filled_quantity],
                'remainingQuantity': [remaining_quantity],
                'enteredTime_IC': [converted_date]
            }

            if symbol in contract_symbol:  # only add to file if symbol matches
                df_temp = pd.DataFrame.from_dict(df_dict)
                df_temp = df_temp.set_index(keys=['symbol', 'datetime'])
                self._frame = pd.concat([self._frame, df_temp])
                self._frame.sort_index(inplace=True)
        return self._frame

    def populate_order_data_2(self, order):
        """Populates order data columns in dataframe"""

        # Get orders (Which return list of order did in past)

        ''' ORDER TEMPLATE FILLED
            {
                'orderType': 'MARKET',
                'session': 'NORMAL',
                'duration': 'DAY',
                'orderStrategyType': 'SINGLE',
                'orderLegCollection': [{
                    'instruction': 'BUY_TO_OPEN',
                    'quantity': 1,
                    'instrument': {
                        'symbol': 'NIO_061121C45',
                        'assetType': 'OPTION'
                    }
                }]
            }
        '''
        if order:
            order_type = order['orderType']
            session = order['']
            duration = order['']
            instruction = order['orderLegCollection'][0]['instruction']
            quantity = order['orderLegCollection'][0]['quantity']
            symbol = order['orderLegCollection'][0]['instrument']['symbol']
            asset_type = order['orderLegCollection'][0]['instrument']['assetType']
        else:
            order_type = ''
            session = ''
            duration = ''
            instruction = ''
            quantity = ''
            symbol = ''
            asset_type = ''

        if 'order_type' not in self._frame:
            self._frame['order_type'] = ''
        if 'instruction' not in self._frame:
            self._frame['instruction'] = ''
        if 'quantity' not in self._frame:
            self._frame['quantity'] = ''
        if 'option_symbol' not in self._frame:
            self._frame['option_symbol'] = ''
        if 'asset_type' not in self._frame:
            self._frame['asset_type'] = ''

        self._frame.iloc[-1, self._frame.columns.get_loc('order_type')] = order_type
        self._frame.iloc[-1, self._frame.columns.get_loc('instruction')] = instruction
        self._frame.iloc[-1, self._frame.columns.get_loc('quantity')] = quantity
        self._frame.iloc[-1, self._frame.columns.get_loc('option_symbol')] = symbol
        self._frame.iloc[-1, self._frame.columns.get_loc('asset_type')] = asset_type

        return self._frame

    def buy_stock(self, symbol, instruction, account_id):
        # Define the Order.

        params = {
            "symbol": symbol,
            "range": "NTM",
            # "fromDate":"2021-02-24",
            # "toDate": "2021-02-28"
        }

        order_template = buy_limit_enter = {
            # "orderType": "LIMIT",
            "orderType": "MARKET",
            "session": "NORMAL",
            "duration": "DAY",
            # "price": .01,
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": instruction,
                    "quantity": 1,
                    "instrument": {
                        "symbol": symbol,
                        # "assetType": "EQUITY"
                        "assetType": "OPTION"
                    }
                }
            ]
        }

        # Place the Order.
        try:
            order_response = self.session.place_order(
                account=account_id,
                order=order_template
            )
            print("successful ", instruction, " for ", symbol)
            return order_response

        except Exception as e:
            print("Error comes while trying ", instruction)
            print(str(e))

    def current_positions(self):
        pos = self.session.get_accounts(account='all', fields=['orders', 'positions'])
        x = 0
        allsym = []
        for i in pos:
            if pos[x]['securitiesAccount']['accountId'] == '71611620':
                for sym in pos[x]['securitiesAccount']['positions']:
                    allsym.append(sym['instrument']['symbol'])
                # print(pos[x]['securitiesAccount']['positions'])

            x += 1
        return allsym

    def confirm_order(self, order_id, account_id):
        """Returns order confirmation info from Ameritrade"""
        order = self.session.get_orders(
            account=account_id,
            order_id=order_id
        )
        return order['status']

    def query_orders(self, symbol, account_id):
        """Returns order confirmed, quantity filled, and quantity remaining"""

        # Get orders (Which return list of order did in past)
        transactions_info = self.session.get_orders(
            account=account_id
        )

        # search for FILLED transactions
        filled_orders = []
        cumulative_calls_quantity = 0
        cumulative_puts_quantity = 0
        remaining_quantity = 0
        for order in transactions_info:
            if order['status'] == 'FILLED':
                filled_orders.append(order)
                order_leg = order['orderLegCollection'][0]
                contract_symbol = order_leg['instrument']['symbol']
                asset_type = order_leg['instrument']['assetType']
                desc = order_leg['instrument']['description']

                # check for matching symbol
                if symbol in contract_symbol:

                    # update buy
                    if order_leg['instruction'] == 'BUY_TO_OPEN' and order_leg['positionEffect'] == 'OPENING' \
                            and asset_type == 'OPTION':

                        print("Description: %s" % desc)
                        if 'Call' in desc:
                            cumulative_calls_quantity += order['filledQuantity']
                        elif 'Put' in desc:
                            cumulative_puts_quantity += order['filledQuantity']
                        remaining_quantity += order['remainingQuantity']

                    # update sell
                    if order_leg['instruction'] == 'SELL_TO_CLOSE' and order_leg['positionEffect'] == 'CLOSING' \
                            and asset_type == 'OPTION':
                        if 'Call' in desc:
                            cumulative_calls_quantity -= order['filledQuantity']
                        elif 'Put' in desc:
                            cumulative_puts_quantity -= order['filledQuantity']
                        remaining_quantity += order['remainingQuantity']

                print("Order ID: %s, Instruction: %s" % (order['orderId'], order_leg['instruction']))
                print("Calls Quantity: %d, Puts Quantity: %d" % (cumulative_calls_quantity, cumulative_puts_quantity))
        return filled_orders, cumulative_calls_quantity, cumulative_puts_quantity, remaining_quantity

    def buy_condition(self, symbol):
        locals_data = locals()
        del locals_data['self']

        self._current_indicators['buy_condition'] = {}
        self._current_indicators['buy_condition']['args'] = locals_data
        self._current_indicators['buy_condition']['func'] = self.buy_condition

        signal_list = []

        global buy_and_sell_count
        buy_calls_count = 0
        sell_calls_count = 0
        buy_puts_count = 0
        sell_puts_count = 0
        no_action_calls_count = 0
        no_action_puts_count = 0

        # Generates signals column called buy_condition
        for i in range(len(self._frame)):
            # Buy CALLS condition
            if self._frame["sma_9_slope"][i] > 0 and \
                    self._frame["sma_volume_50_slope"][i] > 0 and \
                    self._frame['sma9_crossed_sma50'][i] == "9maAbove50ma":
                no_action_calls_count = 0
                buy_calls_count += 1
                signal_list.append('Buy Calls ' + str(buy_calls_count) + ' ' + symbol)

            # Buy PUTS condition
            elif self._frame["sma_9_slope"][i] < 0 and \
                    self._frame["sma_volume_50_slope"][i] < 0 and \
                    self._frame['sma9_crossed_sma50'][i] == "9maBelow50ma" and \
                    self._frame['sma_200'][i] > self._frame['sma_50'][i]:
                no_action_puts_count = 0
                buy_puts_count += 1
                signal_list.append('Buy Puts ' + str(buy_puts_count) + ' ' + symbol)

            # NO ACTION condition
            else:
                signal_list.append('No action')

        # Sets a column in the dataframe containing the  signals: ['Buy Calls 1 PETS', 'Buy Puts 1 PETS', 'No action1']
        self._frame["buy_condition"] = pd.Series(signal_list).values
        self.stock_data = self._frame
        self.indicator_signal_list = signal_list

        return self._frame, signal_list

    def sma9_crossed_sma50(self):

        # def calculate(row):
        #     if row["sma_9"].item() == 0 or row["sma_50"].item() == 0:
        #         val = 0
        #     elif self._frame["sma_9"].item() > self._frame["sma_50"].item():
        #         value = '9maAbove50ma'
        #     else:
        #         value = '9maBelow50ma'
        #     return  value

        locals_data = locals()
        del locals_data['self']

        self._current_indicators['sma9_crossed_sma50'] = {}
        self._current_indicators['sma9_crossed_sma50']['args'] = locals_data
        self._current_indicators['sma9_crossed_sma50']['func'] = self.sma9_crossed_sma50

        temp_list = []
        crossed_above = False
        crossed_below = False
        for i in range(len(self._frame)):
            if self._frame["sma_9"][i] == 0 or self._frame["sma_50"][i] == 0 or self._frame["sma_50"][i] == \
                    self._frame["sma_9"][i]:
                temp_list.append(0)
            elif self._frame["sma_9"][i] > self._frame["sma_50"][i]:
                if crossed_above == False:
                    crossed_above = True
                    crossed_below = False
                    temp_list.append('9maCrossedAbove50ma')
                else:
                    temp_list.append('9maAbove50ma')
            else:
                if crossed_below == False:
                    crossed_below = True
                    crossed_above = False
                    temp_list.append('9maCrossedBelow50ma')
                else:
                    temp_list.append('9maBelow50ma')

        self._frame["sma9_crossed_sma50"] = pd.Series(temp_list).values

        # calculate colum for 9 mov cross to 50 mov

        # self._frame['sma9_crossed_sma50'] = np.where(self._frame["sma_9"] > self._frame["sma_50"], '9maAbove50ma', "9maBelow50ma")

        return self._frame

    def ema(self, period: int, alpha: float = 0.0, column_name='ema') -> pd.DataFrame:
        """Calculates the Exponential Moving Average (EMA).

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating the EMA.

        alpha {float} -- The alpha weight used in the calculation. (default: {0.0})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the EMA indicator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.ema(period=50, alpha=1/50)
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.ema

        # Add the EMA
        self._frame[column_name] = self._price_groups['close'].transform(
            lambda x: x.ewm(span=period).mean()
        )

        return self._frame

    def rate_of_change(self, period: int = 1, column_name: str = 'rate_of_change') -> pd.DataFrame:
        """Calculates the Rate of Change (ROC).

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the ROC. (default: {1})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the ROC indicator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.rate_of_change()
        """
        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.rate_of_change

        # Add the Momentum Price indicator.
        # print(self._price_groups)
        # for key, item in self._price_groups:
        #     print(self._price_groups.get_group(key), "\n\n")

        self._frame[column_name] = self._price_groups['close'].transform(
            lambda x: x.pct_change(periods=period)
        )

        return self._frame

    def rate_of_change_volume(self, period: int = 1, column_name: str = 'rate_of_change_volume') -> pd.DataFrame:
        """Calculates the Rate of Change (ROC) based on valume.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the ROC. (default: {1})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the ROC indicator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.rate_of_change_volume()
        """
        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.rate_of_change_volume

        # Add the Momentum Volume indicator.
        # print(self._price_groups)
        # for key, item in self._price_groups:
        #     print(self._price_groups.get_group(key), "\n\n")

        self._frame[column_name] = self._price_groups['volume'].transform(
            lambda x: x.pct_change(periods=period)
        )

        return self._frame

    def bollinger_bands(self, period: int = 20, column_name: str = 'bollinger_bands') -> pd.DataFrame:
        """Calculates the Bollinger Bands.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the Bollinger Bands. (default: {20})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Lower and Upper band
            indicator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.bollinger_bands()
        """
        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.bollinger_bands

        # Define the Moving Avg.
        self._frame['moving_avg'] = self._price_groups['close'].transform(
            lambda x: x.rolling(window=period).mean()
        )

        # Define Moving Std.
        self._frame['moving_std'] = self._price_groups['close'].transform(
            lambda x: x.rolling(window=period).std()
        )

        # Define the Upper Band.
        self._frame['band_upper'] = 4 * (self._frame['moving_std'] / self._frame['moving_avg'])

        # Define the lower band
        self._frame['band_lower'] = (
                (self._frame['close'] - self._frame['moving_avg']) +
                (2 * self._frame['moving_std']) /
                (4 * self._frame['moving_std'])
        )

        # Clean up before sending back.
        self._frame.drop(
            labels=['moving_avg', 'moving_std'],
            axis=1,
            inplace=True
        )

        return self._frame

    def average_true_range(self, period: int = 14, column_name: str = 'average_true_range') -> pd.DataFrame:
        """Calculates the Average True Range (ATR).

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the ATR. (default: {14})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the ATR included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.average_true_range()
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.average_true_range

        # Calculate the different parts of True Range.
        self._frame['true_range_0'] = abs(self._frame['high'] - self._frame['low'])
        self._frame['true_range_1'] = abs(self._frame['high'] - self._frame['close'].shift())
        self._frame['true_range_2'] = abs(self._frame['low'] - self._frame['close'].shift())

        # Grab the Max.
        self._frame['true_range'] = self._frame[['true_range_0', 'true_range_1', 'true_range_2']].max(axis=1)

        # Calculate the Average True Range.
        self._frame['average_true_range'] = self._frame['true_range'].transform(
            lambda x: x.ewm(span=period, min_periods=period).mean()
        )

        # Clean up before sending back.
        self._frame.drop(
            labels=['true_range_0', 'true_range_1', 'true_range_2', 'true_range'],
            axis=1,
            inplace=True
        )

        return self._frame

    def stochastic_oscillator(self, column_name: str = 'stochastic_oscillator') -> pd.DataFrame:
        """Calculates the Stochastic Oscillator.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Stochastic Oscillator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.stochastic_oscillator()
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.stochastic_oscillator

        # Calculate the stochastic_oscillator.
        self._frame['stochastic_oscillator'] = (
                self._frame['close'] - self._frame['low'] /
                self._frame['high'] - self._frame['low']
        )

        return self._frame

    def macd(self, fast_period: int = 12, slow_period: int = 26, column_name: str = 'macd') -> pd.DataFrame:
        """Calculates the Moving Average Convergence Divergence (MACD).

        Arguments:
        ----
        fast_period {int} -- The number of periods to use when calculating
            the fast moving MACD. (default: {12})

        slow_period {int} -- The number of periods to use when calculating
            the slow moving MACD. (default: {26})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the MACD included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.macd(fast_period=12, slow_period=26)
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.macd

        # Calculate the Fast Moving MACD.
        self._frame['macd_fast'] = self._frame['close'].transform(
            lambda x: x.ewm(span=fast_period, min_periods=fast_period).mean()
        )

        # Calculate the Slow Moving MACD.
        self._frame['macd_slow'] = self._frame['close'].transform(
            lambda x: x.ewm(span=slow_period, min_periods=slow_period).mean()
        )

        # Calculate the difference between the fast and the slow.
        self._frame['macd_diff'] = self._frame['macd_fast'] - self._frame['macd_slow']

        # Calculate the Exponential moving average of the fast.
        self._frame['macd'] = self._frame['macd_diff'].transform(
            lambda x: x.ewm(span=9, min_periods=8).mean()
        )

        return self._frame

    def mass_index(self, period: int = 9, column_name: str = 'mass_index') -> pd.DataFrame:
        """Calculates the Mass Index indicator.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the mass index. (default: {9})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Mass Index included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.mass_index(period=9)
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.mass_index

        # Calculate the Diff.
        self._frame['diff'] = self._frame['high'] - self._frame['low']

        # Calculate Mass Index 1
        self._frame['mass_index_1'] = self._frame['diff'].transform(
            lambda x: x.ewm(span=period, min_periods=period - 1).mean()
        )

        # Calculate Mass Index 2
        self._frame['mass_index_2'] = self._frame['mass_index_1'].transform(
            lambda x: x.ewm(span=period, min_periods=period - 1).mean()
        )

        # Grab the raw index.
        self._frame['mass_index_raw'] = self._frame['mass_index_1'] / self._frame['mass_index_2']

        # Calculate the Mass Index.
        self._frame['mass_index'] = self._frame['mass_index_raw'].transform(
            lambda x: x.rolling(window=25).sum()
        )

        # Clean up before sending back.
        self._frame.drop(
            labels=['diff', 'mass_index_1', 'mass_index_2', 'mass_index_raw'],
            axis=1,
            inplace=True
        )

        return self._frame

    def force_index(self, period: int, column_name: str = 'force_index') -> pd.DataFrame:
        """Calculates the Force Index.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the force index.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the force index included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.force_index(period=9)
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.force_index

        # Calculate the Force Index.
        self._frame[column_name] = self._frame['close'].diff(period) * self._frame['volume'].diff(period)

        return self._frame

    def ease_of_movement(self, period: int, column_name: str = 'ease_of_movement') -> pd.DataFrame:
        """Calculates the Ease of Movement.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the Ease of Movement.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Ease of Movement included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.ease_of_movement(period=9)
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.ease_of_movement

        # Calculate the ease of movement.
        high_plus_low = (self._frame['high'].diff(1) + self._frame['low'].diff(1))
        diff_divi_vol = (self._frame['high'] - self._frame['low']) / (2 * self._frame['volume'])
        self._frame['ease_of_movement_raw'] = high_plus_low * diff_divi_vol

        # Calculate the Rolling Average of the Ease of Movement.
        self._frame['ease_of_movement'] = self._frame['ease_of_movement_raw'].transform(
            lambda x: x.rolling(window=period).mean()
        )

        # Clean up before sending back.
        self._frame.drop(
            labels=['ease_of_movement_raw'],
            axis=1,
            inplace=True
        )

        return self._frame

    def commodity_channel_index(self, period: int, column_name: str = 'commodity_channel_index') -> pd.DataFrame:
        """Calculates the Commodity Channel Index.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the Commodity Channel Index.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Commodity Channel Index included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.commodity_channel_index(period=9)
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.commodity_channel_index

        # Calculate the Typical Price.
        self._frame['typical_price'] = (self._frame['high'] + self._frame['low'] + self._frame['close']) / 3

        # Calculate the Rolling Average of the Typical Price.
        self._frame['typical_price_mean'] = self._frame['pp'].transform(
            lambda x: x.rolling(window=period).mean()
        )

        # Calculate the Rolling Standard Deviation of the Typical Price.
        self._frame['typical_price_std'] = self._frame['pp'].transform(
            lambda x: x.rolling(window=period).std()
        )

        # Calculate the Commodity Channel Index.
        self._frame[column_name] = self._frame['typical_price_mean'] / self._frame['typical_price_std']

        # Clean up before sending back.
        self._frame.drop(
            labels=['typical_price', 'typical_price_mean', 'typical_price_std'],
            axis=1,
            inplace=True
        )

        return self._frame

    def standard_deviation(self, period: int, column_name: str = 'standard_deviation') -> pd.DataFrame:
        """Calculates the Standard Deviation.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the standard deviation.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Standard Deviation included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.standard_deviation(period=9)
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.standard_deviation

        # Calculate the Standard Deviation.
        self._frame[column_name] = self._frame['close'].transform(
            lambda x: x.ewm(span=period).std()
        )

        return self._frame

    def chaikin_oscillator(self, period: int, column_name: str = 'chaikin_oscillator') -> pd.DataFrame:
        """Calculates the Chaikin Oscillator.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the Chaikin Oscillator.

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Chaikin Oscillator included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.chaikin_oscillator(period=9)
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.chaikin_oscillator

        # Calculate the Money Flow Multiplier.
        money_flow_multiplier_top = 2 * (self._frame['close'] - self._frame['high'] - self._frame['low'])
        money_flow_multiplier_bot = (self._frame['high'] - self._frame['low'])

        # Calculate Money Flow Volume
        self._frame['money_flow_volume'] = (money_flow_multiplier_top / money_flow_multiplier_bot) * self._frame[
            'volume']

        # Calculate the 3-Day moving average of the Money Flow Volume.
        self._frame['money_flow_volume_3'] = self._frame['money_flow_volume'].transform(
            lambda x: x.ewm(span=3, min_periods=2).mean()
        )

        # Calculate the 10-Day moving average of the Money Flow Volume.
        self._frame['money_flow_volume_10'] = self._frame['money_flow_volume'].transform(
            lambda x: x.ewm(span=10, min_periods=9).mean()
        )

        # Calculate the Chaikin Oscillator.
        self._frame[column_name] = self._frame['money_flow_volume_3'] - self._frame['money_flow_volume_10']

        # Clean up before sending back.
        self._frame.drop(
            labels=['money_flow_volume_3', 'money_flow_volume_10', 'money_flow_volume'],
            axis=1,
            inplace=True
        )

        return self._frame

    def kst_oscillator(self, r1: int, r2: int, r3: int, r4: int, n1: int, n2: int, n3: int, n4: int,
                       column_name: str = 'kst_oscillator') -> pd.DataFrame:
        """Calculates the Mass Index indicator.

        Arguments:
        ----
        period {int} -- The number of periods to use when calculating
            the mass index. (default: {9})

        Returns:
        ----
        {pd.DataFrame} -- A Pandas data frame with the Mass Index included.

        Usage:
        ----
            >>> historical_prices_df = trading_robot.grab_historical_prices(
                start=start_date,
                end=end_date,
                bar_size=1,
                bar_type='minute'
            )
            >>> price_data_frame = pd.DataFrame(data=historical_prices)
            >>> indicator_client = Indicators(price_data_frame=price_data_frame)
            >>> indicator_client.mass_index(period=9)
        """

        locals_data = locals()
        del locals_data['self']

        self._current_indicators[column_name] = {}
        self._current_indicators[column_name]['args'] = locals_data
        self._current_indicators[column_name]['func'] = self.kst_oscillator

        # Calculate the ROC 1.
        self._frame['roc_1'] = self._frame['close'].diff(r1 - 1) / self._frame['close'].shift(r1 - 1)

        # Calculate the ROC 2.
        self._frame['roc_2'] = self._frame['close'].diff(r2 - 1) / self._frame['close'].shift(r2 - 1)

        # Calculate the ROC 3.
        self._frame['roc_3'] = self._frame['close'].diff(r3 - 1) / self._frame['close'].shift(r3 - 1)

        # Calculate the ROC 4.
        self._frame['roc_4'] = self._frame['close'].diff(r4 - 1) / self._frame['close'].shift(r4 - 1)

        # Calculate the Mass Index.
        self._frame['roc_1_n'] = self._frame['roc_1'].transform(
            lambda x: x.rolling(window=n1).sum()
        )

        # Calculate the Mass Index.
        self._frame['roc_2_n'] = self._frame['roc_2'].transform(
            lambda x: x.rolling(window=n2).sum()
        )

        # Calculate the Mass Index.
        self._frame['roc_3_n'] = self._frame['roc_3'].transform(
            lambda x: x.rolling(window=n3).sum()
        )

        # Calculate the Mass Index.
        self._frame['roc_4_n'] = self._frame['roc_4'].transform(
            lambda x: x.rolling(window=n4).sum()
        )

        self._frame[column_name] = 100 * (
                    self._frame['roc_1_n'] + 2 * self._frame['roc_2_n'] + 3 * self._frame['roc_3_n'] + 4 * self._frame[
                'roc_4_n'])
        self._frame[column_name + "_signal"] = self._frame['column_name'].transform(
            lambda x: x.rolling().mean()
        )

        # Clean up before sending back.
        self._frame.drop(
            labels=['roc_1', 'roc_2', 'roc_3', 'roc_4', 'roc_1_n', 'roc_2_n', 'roc_3_n', 'roc_4_n'],
            axis=1,
            inplace=True
        )

        return self._frame

    def refresh(self):
        """Updates the Indicator columns after adding the new rows."""

        # First update the groups since, we have new rows.
        self._price_groups = self._stock_frame.symbol_groups

        # Grab all the details of the indicators so far.
        for indicator in self._current_indicators:
            # Grab the function.
            indicator_argument = self._current_indicators[indicator]['args']

            # Grab the arguments.
            indicator_function = self._current_indicators[indicator]['func']

            # Update the function.
            indicator_function(**indicator_argument)

    def check_signals(self) -> Union[pd.DataFrame, None]:
        """Checks to see if any signals have been generated.

        Returns:
        ----
        {Union[pd.DataFrame, None]} -- If signals are generated then a pandas.DataFrame
            is returned otherwise nothing is returned.
        """

        signals_df = self._stock_frame._check_signals(
            indicators=self._indicator_signals,
            indciators_comp_key=self._indicators_comp_key,
            indicators_key=self._indicators_key
        )

        return signals_df
