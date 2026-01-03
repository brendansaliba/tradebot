import json
import time as time_true
import pathlib

import pandas
import pandas as pd

from datetime import datetime
from datetime import timezone
from datetime import timedelta

from typing import List
from typing import Dict
from typing import Union

from cashflow.classes.trades import Trade
from cashflow.classes.portfolio import Portfolio
from cashflow.classes.stock_frame import StockFrame
from cashflow.classes.indicators import Indicators

from td.client import TDClient
from td.utils import TDUtilities

# We are going to be doing some timestamp conversions.
milliseconds_since_epoch = TDUtilities().milliseconds_since_epoch


class PyRobot():

    def __init__(self, client_id: str, redirect_uri: str, paper_trading: bool = True, credentials_path: str = None, json_path: str = None, trading_account: str = None, account_id: str = None, trading_symbol: str = None) -> None:
        """Initalizes a new instance of the robot and logs into the API platform specified.

        Arguments:
        ----
        client_id {str} -- The Consumer ID assigned to you during the App registration.
            This can be found at the app registration portal.

        redirect_uri {str} -- This is the redirect URL that you specified when you created your
            TD Ameritrade Application.

        Keyword Arguments:
        ----
        credentials_path {str} -- The path to the session state file used to prevent a full
            OAuth workflow. (default: {None})

        trading_account {str} -- Your TD Ameritrade account number. (default: {None})

        """

        # Set the attirbutes
        print("Bot created...")
        self.trading_account = trading_account
        self.account_id = account_id
        self.client_id = client_id
        self.redirect_uri = redirect_uri
        self.credentials_path = credentials_path
        self.json_path = json_path
        self.trades = {}
        self.jank_portfolio = {}
        self.old_responses = []
        self.historical_prices = None

        # Classes
        self.session: TDClient = self._create_session()
        self.portfolio: Portfolio = self._create_portfolio()
        self.stock_frame: StockFrame = self._create_stock_frame()
        self.indicator: Indicators = self._create_indicator_client()

        # Trading stuff
        self.signals = []
        self.call_options = []
        self.put_options = []
        self.trading_symbol = trading_symbol

        self.paper_trading = paper_trading

        self._bar_size = None
        self._bar_type = None

    def _create_session(self) -> TDClient:
        """Start a new session.

        Creates a new session with the TD Ameritrade API and logs the user into
        the new session.

        Returns:
        ----
        TDClient -- A TDClient object with an authenticated sessions.

        """

        # Create a new instance of the client
        td_client = TDClient(
            client_id=self.client_id,
            redirect_uri=self.redirect_uri,
            credentials_path=self.credentials_path
        )

        # log the client into the new session
        td_client.login()
        print("TDClient session created and logged in...")
        print('Trading with account:', self.account_id)

        return td_client

    def _create_portfolio(self) -> Portfolio:
        """Create a new portfolio.

        Creates a Portfolio Object to help store and organize positions
        as they are added and removed during trading.

        Returns:
        ----
        Portfolio -- A pyrobot.Portfolio object with no positions.
        """

        # Initalize the portfolio.
        self.portfolio = Portfolio(account_number=self.trading_account)
        print("Portfolio created...")

        # Assign the Client
        self.portfolio.td_client = self.session

        return self.portfolio

    def _create_indicator_client(self) -> Indicators:
        """Create a new indicator client.

        Returns:
        ----
        Indicator -- A pyrobot.Indicator object with no indicators.
        """

        self.indicator = Indicators(price_data_frame=self.stock_frame)
        print("Indicator client created...")

        # Assign the Client
        self.indicator.td_client = self.session

        return self.indicator

    def _get_historical_prices(self) -> pandas.DataFrame:
        """
        Yes, things. Stuff also.
        """
        end_date = datetime.today()
        start_date = end_date - timedelta(minutes=200)  # previously seconds=5 ???

        historical_prices = self.grab_historical_prices(
            start=end_date,
            end=start_date,
            bar_size=1,
            bar_type='minute',
            symbols=[self.trading_symbol]
        )

        self.historical_prices = historical_prices['aggregated']

        return self.historical_prices

    def _create_stock_frame(self) -> StockFrame:
        """Generates a new StockFrame Object.

        Arguments:
        ----
        data {List[dict]} -- The data to add to the StockFrame object.

        Returns:
        ----
        StockFrame -- A multi-index pandas data frame built for trading.
        """

        data = self._get_historical_prices()

        # Create the Frame.
        self.stock_frame = StockFrame(data=data)

        return self.stock_frame

    @property
    def pre_market_open(self) -> bool:
        """Checks if pre-market is open.

        Uses the datetime module to create US Pre-Market Equity hours in
        UTC time.

        Usage:
        ----
            >>> trading_robot = PyRobot(
            client_id=CLIENT_ID,
            redirect_uri=REDIRECT_URI,
            credentials_path=CREDENTIALS_PATH
            )
            >>> pre_market_open_flag = trading_robot.pre_market_open
            >>> pre_market_open_flag
            True

        Returns:
        ----
        bool -- True if pre-market is open, False otherwise.

        """

        pre_market_start_time = datetime.utcnow().replace(
            hour=8,
            minute=00,
            second=00
        ).timestamp()

        market_start_time = datetime.utcnow().replace(
            hour=13,
            minute=30,
            second=00
        ).timestamp()

        right_now = datetime.utcnow().timestamp()

        if market_start_time >= right_now >= pre_market_start_time:
            return True
        else:
            return False

    @property
    def post_market_open(self):
        """Checks if post-market is open.

        Uses the datetime module to create US Post-Market Equity hours in
        UTC time.

        Usage:
        ----
            >>> trading_robot = PyRobot(
            client_id=CLIENT_ID,
            redirect_uri=REDIRECT_URI,
            credentials_path=CREDENTIALS_PATH
            )
            >>> post_market_open_flag = trading_robot.post_market_open
            >>> post_market_open_flag
            True

        Returns:
        ----
        bool -- True if post-market is open, False otherwise.

        """

        post_market_end_time = datetime.utcnow().replace(
            hour=00,
            minute=00,
            second=00
        ).timestamp()

        market_end_time = datetime.utcnow().replace(
            hour=20,
            minute=00,
            second=00
        ).timestamp()

        right_now = datetime.utcnow().timestamp()

        if post_market_end_time >= right_now >= market_end_time:
            return True
        else:
            return False

    @property
    def regular_market_open(self):
        """Checks if regular market is open.

        Uses the datetime module to create US Regular Market Equity hours in
        UTC time.

        Usage:
        ----
            >>> trading_robot = PyRobot(
            client_id=CLIENT_ID,
            redirect_uri=REDIRECT_URI,
            credentials_path=CREDENTIALS_PATH
            )
            >>> market_open_flag = trading_robot.market_open
            >>> market_open_flag
            True

        Returns:
        ----
        bool -- True if post-market is open, False otherwise.

        """

        market_start_time = datetime.utcnow().replace(
            hour=13,
            minute=30,
            second=00
        ).timestamp()

        market_end_time = datetime.utcnow().replace(
            hour=20,
            minute=00,
            second=00
        ).timestamp()

        right_now = datetime.utcnow().timestamp()

        if market_end_time >= right_now >= market_start_time:
            return True
        else:
            return False

    def create_trade(self, trade_id: str, enter_or_exit: str, long_or_short: str, order_type: str = 'mkt', price: float = 0.0, stop_limit_price=0.0) -> Trade:
        """Initalizes a new instance of a Trade Object.

        This helps simplify the process of building an order by using pre-built templates that can be
        easily modified to incorporate more complex strategies.

        Arguments:
        ----
        trade_id {str} -- The ID associated with the trade, this can then be used to access the trade during runtime.

        enter_or_exit {str} -- Defines whether this trade will be used to enter or exit a position.
            If used to enter, specify `enter`. If used to exit, speicfy `exit`.

        long_or_short {str} -- Defines whether this trade will be used to go long or short a position.
            If used to go long, specify `long`. If used to go short, speicfy `short`.

        Keyword Arguments:
        ----
        order_type {str} -- Defines the type of order to initalize. Possible values
            are `'mkt', 'lmt', 'stop', 'stop-lmt', 'trailign-stop'` (default: {'mkt'})

        price {float} -- The Price to be associate with the order. If the order type is `stop` or `stop-lmt` then
            it is the stop price, if it is a `lmt` order then it is the limit price, and `mkt` is the market
            price.(default: {0.0})

        stop_limit_price {float} -- Only used if the order is a `stop-lmt` and represents the limit price of
            the `stop-lmt` order. (default: {0.0})

        Usage:
        ----
            >>> trading_robot = PyRobot(
                client_id=CLIENT_ID,
                redirect_uri=REDIRECT_URI,
                credentials_path=CREDENTIALS_PATH
            )
            >>> new_trade = trading_robot_portfolio.create_trade(
                trade_id='long_1',
                enter_or_exit='enter',
                long_or_short='long',
                order_type='mkt'
            )
            >>> new_trade

            >>> new_market_trade = trading_robot_portfolio.create_trade(
                trade_id='long_2',
                enter_or_exit='enter',
                long_or_short='long',
                order_type='mkt',
                price=12.00
            )
            >>> new_market_trade

            >>> new_stop_trade = trading_robot_portfolio.create_trade(
                trade_id='long_3',
                enter_or_exit='enter',
                long_or_short='long',
                order_type='stop',
                price=2.00
            )
            >>> new_stop_trade

            >>> new_stop_limit_trade = trading_robot_portfolio.create_trade(
                trade_id='long_4',
                enter_or_exit='enter',
                long_or_short='long',
                order_type='stop-lmt',
                price=2.00,
                stop_limit_price=1.90
            )
            >>> new_stop_limit_trade

        Returns:
        ----
        Trade -- A pyrobot.Trade object with the specified template.
        """

        # Initalize a new trade object.
        trade = Trade()

        # Create a new trade.
        trade.new_trade(
            trade_id=trade_id,
            order_type=order_type,
            side=long_or_short,
            enter_or_exit=enter_or_exit,
            price=price,
            stop_limit_price=stop_limit_price
        )

        # Set the Client.
        trade.account = self.trading_account
        trade._td_client = self.session

        self.trades[trade_id] = trade

        return trade

    def delete_trade(self, index: int) -> None:
        """Deletes an exisiting trade from the `trades` collection.

        Arguments:
        ----
        index {int} -- The index of the order.

        Usage:
        ----
            >>> trading_robot = PyRobot(
                client_id=CLIENT_ID,
                redirect_uri=REDIRECT_URI,
                credentials_path=CREDENTIALS_PATH
            )
            >>> new_trade = trading_robot_portfolio.create_trade(
                enter_or_exit='enter',
                long_or_short='long',
                order_type='mkt'
            )
            >>> trading_robot.delete_trade(index=1)
        """

        if index in self.trades:
            del self.trades[index]

    def grab_current_quotes(self) -> dict:
        """Grabs the current quotes for all positions in the portfolio.

        Makes a call to the TD Ameritrade Get Quotes endpoint with all
        the positions in the portfolio. If only one position exist it will
        return a single dicitionary, otherwise a nested dictionary.

        Usage:
        ----
            >>> trading_robot = PyRobot(
                client_id=CLIENT_ID,
                redirect_uri=REDIRECT_URI,
                credentials_path=CREDENTIALS_PATH
            )
            >>> trading_robot_portfolio.add_position(
            symbol='MSFT',
            asset_type='equity'
            )
            >>> current_quote = trading_robot.grab_current_quotes()
            >>> current_quote
            {
                "MSFT": {
                    "assetType": "EQUITY",
                    "assetMainType": "EQUITY",
                    "cusip": "594918104",
                    ...
                    "regularMarketPercentChangeInDouble": 0,
                    "delayed": true
                }
            }

            >>> trading_robot = PyRobot(
            client_id=CLIENT_ID,
            redirect_uri=REDIRECT_URI,
            credentials_path=CREDENTIALS_PATH
            )
            >>> trading_robot_portfolio.add_position(
            symbol='MSFT',
            asset_type='equity'
            )
            >>> trading_robot_portfolio.add_position(
            symbol='AAPL',
            asset_type='equity'
            )
            >>> current_quote = trading_robot.grab_current_quotes()
            >>> current_quote

            {
                "MSFT": {
                    "assetType": "EQUITY",
                    "assetMainType": "EQUITY",
                    "cusip": "594918104",
                    ...
                    "regularMarketPercentChangeInDouble": 0,
                    "delayed": False
                },
                "AAPL": {
                    "assetType": "EQUITY",
                    "assetMainType": "EQUITY",
                    "cusip": "037833100",
                    ...
                    "regularMarketPercentChangeInDouble": 0,
                    "delayed": False
                }
            }

        Returns:
        ----
        dict -- A dictionary containing all the quotes for each position.

        """

        # First grab all the symbols.
        symbols = self.portfolio.positions.keys()

        # Grab the quotes.
        quotes = self.session.get_quotes(instruments=list(symbols))

        return quotes

    def grab_historical_prices(self, start: datetime, end: datetime, bar_size: int = 1,
                               bar_type: str = 'minute', symbols: List[str] = None) -> List[dict]:
        """Grabs the historical prices for all the postions in a portfolio.

        Overview:
        ----
        Any of the historical price data returned will include extended hours
        price data by default.

        Arguments:
        ----
        start {datetime} -- Defines the start date for the historical prices.

        end {datetime} -- Defines the end date for the historical prices.

        Keyword Arguments:
        ----
        bar_size {int} -- Defines the size of each bar. (default: {1})

        bar_type {str} -- Defines the bar type, can be one of the following:
            `['minute', 'week', 'month', 'year']` (default: {'minute'})

        symbols {List[str]} -- A list of ticker symbols to pull. (default: None)

        Returns:
        ----
        {List[Dict]} -- The historical price candles.

        Usage:
        ----
            >>> trading_robot = PyRobot(
                client_id=CLIENT_ID,
                redirect_uri=REDIRECT_URI,
                credentials_path=CREDENTIALS_PATH
                )
            >>> start_date = datetime.today()
            >>> end_date = start_date - timedelta(days=30)
            >>> historical_prices = trading_robot.grab_historical_prices(
                    start=end_date,
                    end=start_date,
                    bar_size=1,
                    bar_type='minute'
                )
        """

        self._bar_size = bar_size
        self._bar_type = bar_type

        # Convert to ms since epoch since that's what TD wants
        start = str(milliseconds_since_epoch(dt_object=start))
        end = str(milliseconds_since_epoch(dt_object=end))

        new_prices = []

        if not symbols:
            symbols = self.portfolio.positions

        for symbol in symbols:

            # Call to TD API
            historical_prices_response = self.session.get_price_history(
                symbol=symbol,
                period_type='day',
                start_date=start,
                end_date=end,
                frequency_type=bar_type,
                frequency=str(bar_size)
            )

            self.historical_prices[symbol] = {}
            self.historical_prices[symbol]['candles'] = historical_prices_response['candles']

            for candle in historical_prices_response['candles']:

                new_price_mini_dict = {}
                new_price_mini_dict['symbol'] = symbol
                new_price_mini_dict['open'] = candle['open']
                new_price_mini_dict['close'] = candle['close']
                new_price_mini_dict['high'] = candle['high']
                new_price_mini_dict['low'] = candle['low']
                new_price_mini_dict['volume'] = candle['volume']
                new_price_mini_dict['datetime'] = candle['datetime']
                new_prices.append(new_price_mini_dict)

        self.historical_prices['aggregated'] = new_prices

        return self.historical_prices

    def get_latest_bar(self, symbol) -> List[dict]:
        """Returns the latest bar for each symbol in the portfolio.

        Returns:
        ---
        {List[dict]} -- A simplified quote list.

        Usage:
        ----
            >>> trading_robot = PyRobot(
                client_id=CLIENT_ID,
                redirect_uri=REDIRECT_URI,
                credentials_path=CREDENTIALS_PATH
            )
            >>> latest_bars = trading_robot.get_latest_bar()
            >>> latest_bars
        """

        # Grab the info from the last quest.
        bar_size = self._bar_size
        bar_type = self._bar_type

        # Define the start and end date.
        end_date = datetime.today()
        start_date = end_date - timedelta(days=1)
        start = str(milliseconds_since_epoch(dt_object=start_date))
        end = str(milliseconds_since_epoch(dt_object=end_date))

        latest_prices = []
        symbols = [symbol]

        # Loop through each symbol.
        # previously looping through self.portfolio.positions
        for symbol in symbols:

            try:

                # Grab the request.
                historical_prices_response = self.session.get_price_history(
                    symbol=symbol,
                    period_type='day',
                    start_date=start,
                    end_date=end,
                    frequency_type=bar_type,
                    frequency=bar_size,
                    extended_hours=True
                )

            except:

                time_true.sleep(2)

                # Grab the request.
                historical_prices_response = self.session.get_price_history(
                    symbol=symbol,
                    period_type='day',
                    start_date=start,
                    end_date=end,
                    frequency_type=bar_type,
                    frequency=bar_size,
                    extended_hours=True
                )

            # parse the candles.
            for candle in historical_prices_response['candles'][-1:]:

                new_price_mini_dict = {}
                new_price_mini_dict['symbol'] = symbol
                new_price_mini_dict['open'] = candle['open']
                new_price_mini_dict['close'] = candle['close']
                new_price_mini_dict['high'] = candle['high']
                new_price_mini_dict['low'] = candle['low']
                new_price_mini_dict['volume'] = candle['volume']
                new_price_mini_dict['datetime'] = candle['datetime']
                latest_prices.append(new_price_mini_dict)

        return latest_prices

    def wait_till_next_bar(self, last_bar_timestamp: pd.DatetimeIndex) -> None:
        """Waits the number of seconds till the next bar is released.

        Arguments:
        ----
        last_bar_timestamp {pd.DatetimeIndex} -- The last bar's timestamp.
        """

        last_bar_time = last_bar_timestamp.to_pydatetime()[0].replace(tzinfo=timezone.utc)
        next_bar_time = last_bar_time + timedelta(seconds=60)
        curr_bar_time = datetime.now(tz=timezone.utc)

        last_bar_timestamp = int(last_bar_time.timestamp())
        next_bar_timestamp = int(next_bar_time.timestamp())
        curr_bar_timestamp = int(curr_bar_time.timestamp())

        time_to_wait_now = next_bar_timestamp - curr_bar_timestamp

        if time_to_wait_now < 0:
            time_to_wait_now = 0

        print("=" * 80)
        print("Pausing for the next bar")
        print("-" * 80)
        print("Curr Time: {time_curr}".format(
            time_curr=curr_bar_time.strftime("%Y-%m-%d %H:%M:%S")
        )
        )
        print("Next Time: {time_next}".format(
            time_next=next_bar_time.strftime("%Y-%m-%d %H:%M:%S")
        )
        )
        print("Sleep Time: {seconds}".format(seconds=time_to_wait_now))
        print("-" * 80)
        print('')

        time_true.sleep(time_to_wait_now)

    def execute_signals(self, signals: List[pd.Series], trades_to_execute: dict) -> List[dict]:
        """Executes the specified trades for each signal.

        Arguments:
        ----
        signals {list} -- A pandas.Series object representing the buy signals and sell signals.
            Will check if series is empty before making any trades.

        Trades:
        ----
        trades_to_execute {dict} -- the trades you want to execute if signals are found.

        Returns:
        ----
        {List[dict]} -- Returns all order responses.

        Usage:
        ----
            >>> trades_dict = {
                    'MSFT': {
                        'trade_func': trading_robot.trades['long_msft'],
                        'trade_id': trading_robot.trades['long_msft'].trade_id
                    }
                }
            >>> signals = indicator_client.check_signals()
            >>> trading_robot.execute_signals(
                    signals=signals,
                    trades_to_execute=trades_dict
                )
        """
        
        # Define the Buy and sells.
        buys: pd.Series = signals['buys']
        sells: pd.Series = signals['sells']

        order_responses = []

        # If we have buys or sells continue.
        if not buys.empty:

            # Grab the buy Symbols.
            symbols_list = buys.index.get_level_values(0).to_list()

            # Loop through each symbol.
            for symbol in symbols_list:

                # Check to see if there is a Trade object.
                if symbol in trades_to_execute:

                    if self.portfolio.in_portfolio(symbol=symbol):
                        self.portfolio.set_ownership_status(
                            symbol=symbol,
                            ownership=True
                        )

                    # Set the Execution Flag.
                    trades_to_execute[symbol]['has_executed'] = True
                    trade_obj: Trade = trades_to_execute[symbol]['buy']['trade_func']

                    if not self.paper_trading:

                        # Execute the order.
                        order_response = self.execute_orders(
                            trade_obj=trade_obj
                        )

                        order_response = {
                            'order_id': order_response['order_id'],
                            'request_body': order_response['request_body'],
                            'timestamp': datetime.now().isoformat()
                        }

                        order_responses.append(order_response)

                    else:

                        order_response = {
                            'order_id': trade_obj._generate_order_id(),
                            'request_body': trade_obj.order,
                            'timestamp': datetime.now().isoformat()
                        }

                        order_responses.append(order_response)

        elif not sells.empty:

            # Grab the buy Symbols.
            symbols_list = sells.index.get_level_values(0).to_list()

            # Loop through each symbol.
            for symbol in symbols_list:

                # Check to see if there is a Trade object.
                if symbol in trades_to_execute:

                    # Set the Execution Flag.
                    trades_to_execute[symbol]['has_executed'] = True

                    if self.portfolio.in_portfolio(symbol=symbol):
                        self.portfolio.set_ownership_status(
                            symbol=symbol,
                            ownership=False
                        )

                    trade_obj: Trade = trades_to_execute[symbol]['sell']['trade_func']

                    if not self.paper_trading:

                        # Execute the order.
                        order_response = self.execute_orders(
                            trade_obj=trade_obj
                        )

                        order_response = {
                            'order_id': order_response['order_id'],
                            'request_body': order_response['request_body'],
                            'timestamp': datetime.now().isoformat()
                        }

                        order_responses.append(order_response)

                    else:

                        order_response = {
                            'order_id': trade_obj._generate_order_id(),
                            'request_body': trade_obj.order,
                            'timestamp': datetime.now().isoformat()
                        }

                        order_responses.append(order_response)

        # Save the response.
        self.save_orders(order_response_dict=order_responses)

        return order_responses

    def execute_orders(self, trade_obj: Trade) -> dict:
        """Executes a Trade Object.

        Overview:
        ----
        The `execute_orders` method will execute trades as they're signaled. When executed,
        the `Trade` object will have the order response saved to it, and the order response will
        be saved to a JSON file for further analysis.

        Arguments:
        ----
        trade_obj {Trade} -- A trade object with the `order` property filled out.

        Returns:
        ----
        {dict} -- An order response dicitonary.
        """

        # Execute the order.
        order_dict = self.session.place_order(
            account=self.trading_account,
            order=trade_obj.order
        )

        # Store the order.
        trade_obj._order_response = order_dict

        # Process the order response.
        trade_obj._process_order_response()

        return order_dict

    def execute_orders_2(self, symbol, trading_options):
        """Executes a orders from the prototype Bot.

        Overview:
        ----
        The `execute_orders_2` method will execute orders according to signals in the 'buy_condition' column that
        was created by the prototype bot. The 'buy_condition' column exists in the self.stock_frame dataframe.

        Returns:
        ----
        {dict} -- An order response dicitonary.

        Portfolio position format
            {
                'asset_type': 'equity',
                'quantity': 2,
                'purchase_price': 4.00,
                'symbol': 'MSFT',
                'purchase_date': '2020-01-31'
            }

        """

        # Required variables
        buy_and_sell_count = 0
        buy_calls_count = 0
        buy_puts_count = 0
        stock_data = self.stock_frame
        order = {}
        order_response = {}

        # Query portfolio for existing orders
        filled_orders, calls_quantity, puts_quantity, remaining_quantity = self.query_orders(symbol)

        # Options
        call_symbol = self.call_options[-1]  # last element in the calls_option column
        put_symbol = self.put_options[-1]
        signal = self.signals[-1]
        if "Buy" in signal or "Sell" in signal:
            buy_n = int(signal.split()[-2])
        else:
            buy_n = 1

        print(signal)

        # Buys and sells CALL options ===============================================================
        # Buy CALLS logic
        if signal.startswith("Buy Calls"):

            # buy condition met and no position held in ameritrade
            if buy_n >= 2 and calls_quantity < 1:
                instruction = "BUY_TO_OPEN"

                # if we don't own any of the current option symbol, otherwise, don't buy
                if not self.portfolio.in_portfolio(call_symbol):
                    print("Buying CALL option for {} at time: ".format(symbol), datetime.now().time())
                    # BUY THE CALLS
                    # TODO UNCOMMENT TO ACTUALLY BUY
                    # order, order_response = self.buy_stock(symbol=symbol,
                    #                                        option_symbol=call_symbol,
                    #                                        instruction=instruction)
                elif self.portfolio.in_portfolio(call_symbol):
                    print("Already have option {}.".format(call_symbol))
                    # TODO Implement a function to check for other options for the same underlying that may be sold
                else:
                    print("Something went wrong.")

                buy_and_sell_count = 1
                buy_calls_count += 1
                print("Buy and sell count:", buy_and_sell_count)

        # Sell CALLS logic
        elif signal.startswith("No action"):
            print("Sell CALLS if we have them.")

            # sell condition met and we have CALLS in the portfolio, abs value between 9-50 ma is decreasing
            if stock_data["abs_9_minus_50_slope"][-1] < stock_data["abs_9_minus_50_slope"][-2] and \
                    stock_data['sma9_crossed_sma50'][-1] == '9maAbove50ma' and calls_quantity >= 1:
                instruction = "SELL_TO_CLOSE"

                # if we own a call, sell it, otherwise, don't sell
                if self.portfolio.in_portfolio(call_symbol):
                    print("Selling CALL options.")
                    # SELL THE CALLS
                    # TODO UNCOMMENT TO ACTUALLY SELL
                    # order, order_response = self.sell_stock(symbol=symbol,
                    #                                         option_symbol=call_symbol,
                    #                                         instruction=instruction)
                elif not self.portfolio.in_portfolio(call_symbol):
                    print("Do not have option {}.".format(call_symbol))
                    # TODO Implement a function to check for other options for the same underlying that may be sold
                else:
                    print("Something went wrong.")

                buy_and_sell_count = 0
            else:
                print("Do not own any CALLS.")

        # Buys and sells PUTS options ===============================================================
        # Buy PUTS logic
        if signal.startswith("Buy Puts"):

            # buy puts condition met and no position held in TD
            if buy_n >= 2 and puts_quantity < 1:
                instruction = "BUY_TO_OPEN"

                # if we don't own any of the current option symbol, otherwise, don't buy
                if not self.portfolio.in_portfolio(put_symbol):
                    print("Buying PUT option for {} at time: ".format(symbol), datetime.now().time())
                    # BUY THE PUTS
                    # TODO UNCOMMENT TO ACTUALLY BUY
                    # order, order_response = self.buy_stock(symbol=symbol,
                    #                                        option_symbol=put_symbol,
                    #                                        instruction=instruction)
                elif self.portfolio.in_portfolio(put_symbol):
                    print("Already have option {}.".format(put_symbol))
                    # TODO Implement a function to check for other options for the same underlying that may be sold
                else:
                    print("Something went wrong.")

                buy_and_sell_count = 1
                buy_puts_count += 1
                print("Buy and sell count:", buy_and_sell_count)

        elif signal.startswith("No action"):
            print("Sell PUTS if we have them.")

            # Sell PUTS logic
            if stock_data["abs_9_minus_50_slope"][-1] < stock_data["abs_9_minus_50_slope"][-2] and \
                    stock_data['sma9_crossed_sma50'][-1] == '9maBelow50ma' and puts_quantity >= 1:
                instruction = "SELL_TO_CLOSE"

                # if we own a put, sell it, otherwise, don't sell
                if put_symbol in self.portfolio.positions:
                    print("Selling PUT options.")
                    # SELL THE CALLS
                    # TODO UNCOMMENT TO ACTUALLY SELL
                    # order, order_response = self.sell_stock(symbol=symbol,
                    #                                         option_symbol=put_symbol,
                    #                                         instruction=instruction)
                elif not self.portfolio.in_portfolio(put_symbol):
                    print("Do not have option {}.".format(put_symbol))
                    # TODO Implement a function to check for other options for the same underlying that may be sold
                else:
                    print("Something went wrong.")

                stock_data["buy_count"] = -1
                buy_and_sell_count = 0
            else:
                print("Do not own any PUTS.")

        return order, order_response

    def buy_stock(self, symbol, option_symbol, instruction):
        """
        Format to add to the portfolio
        {
                'asset_type': 'equity',
                'quantity': 2,
                'purchase_price': 4.00,
                'symbol': 'MSFT',
                'purchase_date': '2020-01-31'
            }
        """

        # Define the Order.
        order_response = {}
        order_template = {
            "orderType": "MARKET",  # "LIMIT"
            "session": "NORMAL",
            "duration": "DAY",
            # "price": .01, # add for limit order
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": instruction,  # "BUY_TO_OPEN", "SELL_TO_CLOSE"
                    "quantity": 1,  # number of instruments
                    "instrument": {
                        "symbol": option_symbol,
                        "assetType": "OPTION"  # "EQUITY"
                    }
                }
            ]
        }

        # Place the Order.
        try:
            order_response = self.session.place_order(
                account=self.account_id,
                order=order_template
            )
            print(instruction, "order placed for", symbol)

            # Process the response
            order_leg = order_response['request_body']['orderLegCollection'][0]
            quantity = order_leg['quantity']
            average_price = 0
            asset_type = order_leg['instrument']['assetType']
            position_symbol = order_leg['instrument']['symbol']

            # If symbol isn't in portfolio, add it
            if not self.portfolio.in_portfolio(position_symbol):
                self.portfolio.add_position(symbol=position_symbol,
                                            asset_type=asset_type,
                                            quantity=quantity,
                                            purchase_price=average_price)
            else:
                # If already exists in portfolio, add to the quantity
                existing_quantity = self.portfolio.positions[position_symbol][quantity]
                self.portfolio.positions[position_symbol][quantity] = existing_quantity + quantity

            return order_template, order_response

        except Exception as e:
            print("Error trying to place", instruction, "for", symbol)
            print(str(e))
            return order_template, order_response

    def sell_stock(self, symbol, option_symbol, instruction: str):
        """
        Portfolio format
            {
                'asset_type': 'equity',
                'quantity': 2,
                'purchase_price': 4.00,
                'symbol': 'MSFT',
                'purchase_date': '2020-01-31'
            }
        """

        # Go away warnings
        order_response = {}

        # Check if we have position in our portfolio
        if self.portfolio.in_portfolio(option_symbol):
            quantity = self.portfolio.positions[option_symbol]['quantity']
        else:
            print("Do not have any {}".format(option_symbol))
            return

        # Define the Order.
        new_order_response = {}
        order_template = {
            "orderType": "MARKET",  # "LIMIT"
            "session": "NORMAL",
            "duration": "DAY",
            # "price": .01, # add for limit order
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [
                {
                    "instruction": instruction,  # "BUY_TO_OPEN", "SELL_TO_CLOSE"
                    "quantity": quantity,  # number of instruments
                    "instrument": {
                        "symbol": option_symbol,
                        "assetType": "OPTION"  # "EQUITY"
                    }
                }
            ]
        }

        # Place the Order.
        try:
            order_response = self.session.place_order(
                account=self.account_id,
                order=order_template
            )
            print(instruction, "order placed for", symbol)

            # Remove from the portfolio
            self.portfolio.remove_position(option_symbol)

            return order_template, order_response

        except Exception as e:
                print("Error trying to place ", instruction, " for ", symbol)
                print(str(e))
                return order_template, order_response

    def current_positions(self):
        accounts_info = self.session.get_accounts(account='all', fields=['orders', 'positions'])

        all_symbols = []
        for account in accounts_info:
            if account['securitiesAccount']['accountId'] == self.account_id:
                for sym in account['securitiesAccount']['positions']:
                    all_symbols.append(sym['instrument']['symbol'])

        return all_symbols

    def get_positions_for_symbol(self, symbol) -> dict:
        """ Position format in robot.portfolio
        {
            'asset_type': 'equity',
            'quantity': 2,
            'purchase_price': 4.00,
            'symbol': 'MSFT',
            'purchase_date': '2020-01-31'
        }
        """

        accounts_info = self.session.get_accounts(fields=['orders', 'positions'])

        for account in accounts_info:
            account_info = account['securitiesAccount']
            account_id = account_info['accountId']

            if account_id == self.account_id:
                if 'positions' in account_info:
                    positions = account_info['positions']

                    for position in positions:
                        instrument = position['instrument']
                        quantity = position['longQuantity']
                        average_price = position['averagePrice']
                        asset_type = instrument['assetType']
                        position_symbol = instrument['symbol']

                        if symbol in position_symbol:
                            new_position = self.portfolio.add_position(symbol=position_symbol,
                                                                       asset_type=asset_type,
                                                                       quantity=quantity,
                                                                       purchase_price=average_price)

                            print(new_position)
                else:
                    print('Not currently holding any positions.')

        return self.portfolio.positions

    def save_orders(self, order_response_dict: dict) -> bool:
        """Saves the order to a JSON file for further review.

        Arguments:
        ----
        order_response {dict} -- A single order response.

        Returns:
        ----
        {bool} -- `True` if the orders were successfully saved.
        """

        def default(obj):

            if isinstance(obj, bytes):
                return str(obj)

        # Define the folder.
        folder: pathlib.PurePath = pathlib.Path(
            __file__
        ).parents[1].joinpath("data")

        # See if it exist, if not create it.
        if not folder.exists():
            folder.mkdir()

        # Define the file path.
        file_path = folder.joinpath('orders.json')

        # First check if the file alread exists.
        if file_path.exists():
            with open('data/orders.json', 'r') as order_json:
                orders_list = json.load(order_json)
        else:
            orders_list = []

        # Combine both lists.
        orders_list = orders_list + order_response_dict

        # Write the new data back.
        with open(file='data/orders.json', mode='w+') as order_json:
            json.dump(obj=orders_list, fp=order_json, indent=4, default=default)

        return True

    def query_orders(self, symbol):
        """Returns order confirmed, quantity filled, and quantity remaining"""

        # Get orders (Which return list of order did in past)
        transactions_info = self.session.get_orders(account=self.account_id)

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

    def get_accounts(self, account_number: str = None, all_accounts: bool = False) -> dict:
        """Returns all the account balances for a specified account.

        Keyword Arguments:
        ----
        account_number {str} -- The account number you want to query. (default: {None})

        all_accounts {bool} -- Specifies whether you want to grab all accounts `True` or not
            `False`. (default: {False})

        Returns:
        ----
        Dict -- A dictionary containing all the information in your account.

        Usage:
        ----

            >>> trading_robot = PyRobot(
                client_id=CLIENT_ID,
                redirect_uri=REDIRECT_URI,
                credentials_path=CREDENTIALS_PATH
            )
            >>> trading_robot_accounts = trading_robot.session.get_accounts(
                account_number="<YOUR ACCOUNT NUMBER>"
            )
            >>> trading_robot_accounts
            [
                {
                    'account_number': 'ACCOUNT_ID',
                    'account_type': 'CASH',
                    'available_funds': 0.0,
                    'buying_power': 0.0,
                    'cash_available_for_trading': 0.0,
                    'cash_available_for_withdrawl': 0.0,
                    'cash_balance': 0.0,
                    'day_trading_buying_power': 0.0,
                    'long_market_value': 0.0,
                    'maintenance_call': 0.0,
                    'maintenance_requirement': 0.0,
                    'short_balance': 0.0,
                    'short_margin_value': 0.0,
                    'short_market_value': 0.0
                }
            ]
        """

        # Depending on how the client was initalized, either use the state account
        # or the one passed through the function.
        if all_accounts:
            account = 'all'
        elif self.trading_account:
            account = self.trading_account
        else:
            account = account_number

        # Grab the accounts.
        accounts = self.session.get_accounts(
            account=account
        )

        # Parse the account info.
        accounts_parsed = self._parse_account_balances(
            accounts_response=accounts
        )

        return accounts_parsed

    def _parse_account_balances(self, accounts_response: Union[Dict, List]) -> List[Dict]:
        """Parses an Account response into a more simplified dictionary.

        Arguments:
        ----
        accounts_response {Union[Dict, List]} -- A response from the `get_accounts` call.

        Returns:
        ----
        List[Dict] -- A list of simplified account dictionaries.
        """

        account_lists = []

        if isinstance(accounts_response, dict):

            account_dict = {}

            for account_type_key in accounts_response:

                account_info = accounts_response[account_type_key]

                account_id = account_info['accountId']
                account_type = account_info['type']
                account_current_balances = account_info['currentBalances']
                # account_inital_balances = account_info['initialBalances']

                account_dict['account_number'] = account_id
                account_dict['account_type'] = account_type
                account_dict['cash_balance'] = account_current_balances['cashBalance']
                account_dict['long_market_value'] = account_current_balances['longMarketValue']

                account_dict['cash_available_for_trading'] = account_current_balances.get(
                    'cashAvailableForTrading', 0.0
                )
                account_dict['cash_available_for_withdrawl'] = account_current_balances.get(
                    'cashAvailableForWithDrawal', 0.0
                )
                account_dict['available_funds'] = account_current_balances.get(
                    'availableFunds', 0.0
                )
                account_dict['buying_power'] = account_current_balances.get(
                    'buyingPower', 0.0
                )
                account_dict['day_trading_buying_power'] = account_current_balances.get(
                    'dayTradingBuyingPower', 0.0
                )
                account_dict['maintenance_call'] = account_current_balances.get(
                    'maintenanceCall', 0.0
                )
                account_dict['maintenance_requirement'] = account_current_balances.get(
                    'maintenanceRequirement', 0.0
                )

                account_dict['short_balance'] = account_current_balances.get(
                    'shortBalance', 0.0
                )
                account_dict['short_market_value'] = account_current_balances.get(
                    'shortMarketValue', 0.0
                )
                account_dict['short_margin_value'] = account_current_balances.get(
                    'shortMarginValue', 0.0
                )

                account_lists.append(account_dict)

        elif isinstance(accounts_response, list):

            for account in accounts_response:

                account_dict = {}

                for account_type_key in account:

                    account_info = account[account_type_key]

                    account_id = account_info['accountId']
                    account_type = account_info['type']
                    account_current_balances = account_info['currentBalances']
                    # account_inital_balances = account_info['initialBalances']

                    account_dict['account_number'] = account_id
                    account_dict['account_type'] = account_type
                    account_dict['cash_balance'] = account_current_balances['cashBalance']
                    account_dict['long_market_value'] = account_current_balances['longMarketValue']

                    account_dict['cash_available_for_trading'] = account_current_balances.get(
                        'cashAvailableForTrading', 0.0
                    )
                    account_dict['cash_available_for_withdrawl'] = account_current_balances.get(
                        'cashAvailableForWithDrawal', 0.0
                    )
                    account_dict['available_funds'] = account_current_balances.get(
                        'availableFunds', 0.0
                    )
                    account_dict['buying_power'] = account_current_balances.get(
                        'buyingPower', 0.0
                    )
                    account_dict['day_trading_buying_power'] = account_current_balances.get(
                        'dayTradingBuyingPower', 0.0
                    )
                    account_dict['maintenance_call'] = account_current_balances.get(
                        'maintenanceCall', 0.0
                    )
                    account_dict['maintenance_requirement'] = account_current_balances.get(
                        'maintenanceRequirement', 0.0
                    )
                    account_dict['short_balance'] = account_current_balances.get(
                        'shortBalance', 0.0
                    )
                    account_dict['short_market_value'] = account_current_balances.get(
                        'shortMarketValue', 0.0
                    )
                    account_dict['short_margin_value'] = account_current_balances.get(
                        'shortMarginValue', 0.0
                    )

                    account_lists.append(account_dict)

        return account_lists

    def get_positions(self, account_number: str = None, all_accounts: bool = False) -> List[Dict]:
        """Gets all the positions for a specified account number.

        Arguments:
        ----
        account_number (str, optional): The account number of the account you want
            to pull positions for. Defaults to None.

        all_accounts (bool, optional): If you want to return all the positions for every
            account then set to `True`. Defaults to False.

        Returns:
        ----
        List[Dict]: A list of Position objects.

        Usage:
        ----

            >>> trading_robot = PyRobot(
                client_id=CLIENT_ID,
                redirect_uri=REDIRECT_URI,
                credentials_path=CREDENTIALS_PATH
            )
            >>> trading_robot_positions = trading_robot.session.get_positions(
                account_number="<YOUR ACCOUNT NUMBER>"
            )
            >>> trading_robot_positions
            [
                {
                    'account_number': '111111111',
                    'asset_type': 'EQUITY',
                    'average_price': 0.00,
                    'current_day_profit_loss': -0.96,
                    'current_day_profit_loss_percentage': -5.64,
                    'cusip': '565849106',
                    'description': '',
                    'long_quantity': 3.0,
                    'market_value': 16.05,
                    'settled_long_quantity': 3.0,
                    'settled_short_quantity': 0.0,
                    'short_quantity': 0.0,
                    'sub_asset_type': '',
                    'symbol': 'MRO',
                    'type': ''
                },
                {
                    'account_number': '111111111',
                    'asset_type': 'EQUITY',
                    'average_price': 5.60667,
                    'current_day_profit_loss': -0.96,
                    'current_day_profit_loss_percentage': -5.64,
                    'cusip': '565849106',
                    'description': '',
                    'long_quantity': 3.0,
                    'market_value': 16.05,
                    'settled_long_quantity': 3.0,
                    'settled_short_quantity': 0.0,
                    'short_quantity': 0.0,
                    'sub_asset_type': '',
                    'symbol': 'MRO',
                    'type': ''
                }
            ]
        """

        if all_accounts:
            account = 'all'
        elif self.trading_account and account_number is None:
            account = self.trading_account
        else:
            account = account_number

        # Grab the positions.
        positions = self.session.get_accounts(
            account=account,
            fields=['positions']
        )

        # Parse the positions.
        positions_parsed = self._parse_account_positions(
            positions_response=positions
        )

        return positions_parsed

    def _parse_account_positions(self, positions_response: Union[List, Dict]) -> List[Dict]:
        """Parses the response from the `get_positions` into a more simplified list.

        Arguments:
        ----
        positions_response {Union[List, Dict]} -- Either a list or a dictionary that represents a position.

        Returns:
        ----
        List[Dict] -- A more simplified list of positions.
        """

        positions_lists = []

        if isinstance(positions_response, dict):

            for account_type_key in positions_response:

                account_info = positions_response[account_type_key]

                account_id = account_info['accountId']
                positions = account_info['positions']

                for position in positions:
                    position_dict = {}
                    position_dict['account_number'] = account_id
                    position_dict['average_price'] = position['averagePrice']
                    position_dict['market_value'] = position['marketValue']
                    position_dict['current_day_profit_loss_percentage'] = position['currentDayProfitLossPercentage']
                    position_dict['current_day_profit_loss'] = position['currentDayProfitLoss']
                    position_dict['long_quantity'] = position['longQuantity']
                    position_dict['short_quantity'] = position['shortQuantity']
                    position_dict['settled_long_quantity'] = position['settledLongQuantity']
                    position_dict['settled_short_quantity'] = position['settledShortQuantity']

                    position_dict['symbol'] = position['instrument']['symbol']
                    position_dict['cusip'] = position['instrument']['cusip']
                    position_dict['asset_type'] = position['instrument']['assetType']
                    position_dict['sub_asset_type'] = position['instrument'].get(
                        'subAssetType', ""
                    )
                    position_dict['description'] = position['instrument'].get(
                        'description', ""
                    )
                    position_dict['type'] = position['instrument'].get(
                        'type', ""
                    )

                    positions_lists.append(position_dict)

        elif isinstance(positions_response, list):

            for account in positions_response:

                for account_type_key in account:

                    account_info = account[account_type_key]

                    account_id = account_info['accountId']
                    positions = account_info['positions']

                    for position in positions:
                        position_dict = {}
                        position_dict['account_number'] = account_id
                        position_dict['average_price'] = position['averagePrice']
                        position_dict['market_value'] = position['marketValue']
                        position_dict['current_day_profit_loss_percentage'] = position['currentDayProfitLossPercentage']
                        position_dict['current_day_profit_loss'] = position['currentDayProfitLoss']
                        position_dict['long_quantity'] = position['longQuantity']
                        position_dict['short_quantity'] = position['shortQuantity']
                        position_dict['settled_long_quantity'] = position['settledLongQuantity']
                        position_dict['settled_short_quantity'] = position['settledShortQuantity']

                        position_dict['symbol'] = position['instrument']['symbol']
                        position_dict['cusip'] = position['instrument']['cusip']
                        position_dict['asset_type'] = position['instrument']['assetType']
                        position_dict['sub_asset_type'] = position['instrument'].get(
                            'subAssetType', ""
                        )
                        position_dict['description'] = position['instrument'].get(
                            'description', ""
                        )
                        position_dict['type'] = position['instrument'].get(
                            'type', ""
                        )

                        positions_lists.append(position_dict)

        return positions_lists
