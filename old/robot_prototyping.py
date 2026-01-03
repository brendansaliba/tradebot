from datetime import datetime
from datetime import timedelta

from functions import setup_func

symbol = "PETS"

# Sets up the robot class, robot's portfolio, and the TDSession object
trading_robot, trading_robot_portfolio, TDSession = setup_func()

# myoption_chart = mydb["tbl_options_charts_2021_Feb_24"]
# cursor = myoption_chart.find({})
# count = 0
# symbol = "COTY"
# for row in cursor:
# # Add a single position
#     if count == 0:
#         count +=1
#         dic_test = trading_robot_portfolio.add_position(
#             symbol=symbol,
#             # symbol="AAPL_022621C125",
#             quantity=1,
#             purchase_price=1,
#             asset_type='option',
#             purchase_date='2020-04-01',
#         )

# import csv
# watchlist_file = r"C:\Users\Isaac\Desktop\DESKTOP\Stocktraderclass.com\AlexReedGitHub\python-trading-robot-master\samples\3-9-21 Active Options.csv"
# csv_file_obj = open(watchlist_file)
# csv_reader = csv.reader(csv_file_obj)
# line_count = 0
# # mycol = mydb["tbl_watchlist_info"]
# watchlist_list = []
# for row in csv_reader:
#     watchlist_list.append(row[0])
#
# remove_watchlist = []
# watchlist_list = watchlist_list[0:5]
#
# watchlist_list= ["TAL", 'CAN', "AEE", "SPY", "VIPS"]
# for row in watchlist_list:
#     dic_test = trading_robot_portfolio.add_position(
#                 symbol=symbol,
#                 # symbol="AAPL_022621C125",
#                 quantity=1,
#                 purchase_price=1,
#                 asset_type='option',
#                 purchase_date='2020-04-01',
#             )

            #print(dic_test)
# total_volume = TDSession.get_quotes(instruments=["symbol"])
# print(total_volume["symbol"]["totalVolume"])
# print(total_volume)

buy_flag = 0
no_action_count = 0
buy_calls_count = 0
count = 8
next_sma_9_val = 0
prev_sma_9_val = 0
bc4_flag = False
refresh_time = datetime.now() + timedelta(minutes=21)

while True:
    # update token after 21 minitues
    if datetime.now() > refresh_time:
        TDSession.login()
        print('refresh')
        refresh_time = refresh_time = datetime.now() + timedelta(minutes=21)

    # Grab historical prices, first define the start date and end date.
    start_date = datetime.today()
    end_date = start_date - timedelta(seconds=5)

    # Grab the historical prices for all positions in the portfolio.
    historical_prices = trading_robot.grab_historical_prices(
        start=end_date,
        end=start_date,
        bar_size=1,
        bar_type='minute'
    )

    #print("Historical Prices Candles in json")
    #print(historical_prices)

    #print('*'*30+"\n\n")

    # Convert data to a Data Frame.
    stock_frame = trading_robot.create_stock_frame(
        data=historical_prices['aggregated']
    )

    # We can also add the stock frame to the Portfolio object.
    trading_robot.portfolio.stock_frame = stock_frame

    # Additionally the historical prices can be set as well.
    trading_robot.portfolio.historical_prices = historical_prices

    # Create an indicator Object.
    indicator_client = Indicators_Isaac(price_data_frame=stock_frame)

    print(indicator_client)

    # percentage of change
    indicator_client.per_of_change()

    # Add the 9 day simple moving average.

    sma_df = indicator_client.sma(period=9)
    sma_df = indicator_client.sma(period=50)
    sma_df = indicator_client.sma9_crossed_sma50()
    sma_df = indicator_client.sma(period=200)
    sma_df = indicator_client.sma_volume(period=9)
    sma_df = indicator_client.sma_volume(period=50)
    sma_df = indicator_client.sma_volume(period=200)

    sma_df = indicator_client.abs_9_minus_50_slope()

    # sma_df = indicator_client.buy_condition(TDSession, symbol)
    # sma_df = indicator_client.vwap()
    # sma_df = indicator_client.max_option_chain(TDSession, symbol)
    # sma_df = indicator_client.populate_order_data(TDSession, symbol) # test

    rownum=0
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    # for jk in sma_df:print(jk)
    for row in sma_df['open']:
        # print(row)
        if rownum==len(sma_df['open']):print(sma_df['buy_condition'][rownum],sma_df['new york time'][rownum])
        if rownum == len(sma_df['open'])-1: print(sma_df['buy_condition'][rownum], sma_df['new york time'][rownum])
        if rownum == len(sma_df['open'])-2: print(sma_df['buy_condition'][rownum], sma_df['new york time'][rownum])
        if rownum == len(sma_df['open'])-3: print(sma_df['buy_condition'][rownum], sma_df['new york time'][rownum])
        rownum += 1
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

    # print("50 value ", sma_df.iloc[-1]["sma"])
    # sma_50 = sma_df.iloc[-1]["sma"]
    #
    # print("The 50 day simple moving average in table format")
    # print(sma_df)
    # print('*' * 30 + "\n\n")

    # calculate slope
    # eg. prev value = 237.069567
    # eg next value = 237.069444
    # if prev_sma_9_val == 0 or next_sma_9_val == 0:
    #     prev_sma_9_val = next_sma_9_val = sma_9
    #     slope_9 = 0
    # else:
    #     prev_sma_9_val = next_sma_9_val
    #     next_sma_9_val = sma_9
    #     print('prev value = ',prev_sma_9_val, ' next value =', next_sma_9_val)
    #     slope_9 = (next_sma_9_val - prev_sma_9_val)/0.00001*0.0174533
    #
    # time.sleep(1)
    # print(slope_9)

    # Add the 200 day simple moving average.
    # sma_df = indicator_client.sma(period=200)
    # print("200 value ", sma_df.iloc[-1]["sma"])
    # print("The 200 day simple moving average in table format")
    # print(sma_df)
    # print('*' * 30 + "\n\n")
    # sma_df = indicator_client.buy()
    # print(sma_df.loc["TAL"])
    # sma_df_temp = sma_df

        # buy_condition = sma_df.iloc[-1]["buy_condition"]
        # #calls_option = sma_df.iloc[-1]["calls_option"]
        #
        # if buy_condition.startswith("Buy Calls"):
        #     buy_n = int(buy_condition.split("Buy Calls")[1])
        #
        #     if buy_n >= 1:
        #         buy_stock(calls_option, "BUY")
        #         bc4_flag = True
        #
        # elif buy_condition.startswith("No action"):
        #     no_ac_n = int(buy_condition.split("No acion")[1])
        #     if no_ac_n ==1 and bc4_flag == True:
        #         buy_stock(calls_option,"SELL")
        #         bc4_flag = False








       # print('no_action_count', no_action_count, ' buy call count ', buy_calls_count)



    # sma_200 = sma_df.iloc[-1]["sma"]
    #
    # if count==0:
    #     if (sma_9 > sma_50) and slope_9 > 3:
    #         if  buy_flag == 1:
    #             # buy stock
    #             buy_stock()
    #             buy_flag = 0
    #     else:
    #         buy_flag = 1
    # else:
    #     count-=1

    # # Add the 50 day exponentials moving average.
    # ema_df = indicator_client.ema(period=50)
    # print("The 50 day exponentials moving average in table format")
    # print(ema_df)
    # print('*' * 30 + "\n\n")

    # Add the Bollinger Bands.
    # bollinger_df = indicator_client.bollinger_bands(period=20)
    # print("the Bollinger Bands. in table format")
    # print(sma_df)
    # print('*' * 30 + "\n\n")


    # Add the Rate of Change.
    # rate_of_change= indicator_client.rate_of_change(period=1)
    # print("the Rate of Change in table format")
    # print(rate_of_change)
    # print('*' * 30 + "\n\n")
    #
    # Add the Rate of Change Volume.
    # rate_of_change_volume= indicator_client.rate_of_change_volume(period=1)
    # print("the Rate of Change Volume in table format")
    # print(rate_of_change_volume)
    # print('*' * 30 + "\n\n")

    # # Add the Average True Range.
    # average_true_range = indicator_client.average_true_range(period=14)
    # print("the Average True Range in table format")
    # print(average_true_range)
    # print('*' * 30 + "\n\n")

    # # Add the Stochastic Oscillator.
    # stochastic_oscillator= indicator_client.stochastic_oscillator()
    # print("the Stochastic Oscillator in table format")
    # print(stochastic_oscillator)
    # print('*' * 30 + "\n\n")

    # Add the MACD.
    # macd= indicator_client.macd(fast_period=12, slow_period=26)
    # print("the MACD in table format")
    # print(macd)
    # print('*' * 30 + "\n\n")

    # sma_df.reindex(["open", "close", "high", "low", "volume", "new york time", "per_of_change", "sma_9", "sma_9_slope",
    #             "sma_50", "sma_50_slope", "sma9_crossed_sma50", "buy_condition", "sma_200", 'sma_200_slope', "sma_volume_9", "sma_volume_9_slope",
    #                 "sma_volume_9_slope", "sma_volume_50", "sma_volume_50_slope", "sma_volume_200", "sma_volume_200_slope",
    #                 "typical_mult_volume", "vwap", "vwap__slope"
    #             ], axis=1)

        # save/update csv file

    print('file modifing')
    sma_df.to_csv(
        r"/Users/brendansaliba/Projects/Projects Data Dump/TradeBotDump/analysis_" + symbol + '_' + datetime.now().strftime(
            "%Y-%m-%d") + ".csv",
        header=True, mode='w')

    # Grab the latest bar.
    latest_bars = trading_robot.get_latest_bar()

    # Add to the Stock Frame.
    stock_frame.add_rows(data=latest_bars)

    # Refresh the Indicators.
    indicator_client.refresh()

    # print("="*50)
    # print("Current StockFrame")
    # print("-"*50)
    # print(stock_frame._data)
    # #print(stock_frame.symbol_groups.tail().c)
    # print("-"*50)
    # print("")
    #
    # Check for signals.
    signals = indicator_client.check_signals()

    # Grab the last bar.
    last_bar_timestamp = trading_robot.stock_frame.frame.tail(
        n=1
    ).index.get_level_values(1)

    #Wait till the next bar.
    trading_robot.wait_till_next_bar(last_bar_timestamp=last_bar_timestamp)

symbol = "PETS"
params = {
        "symbol": symbol,
        "range":"NTM",
        # "fromDate":"2021-02-24",
        # "toDate": "2021-02-28"`
    }
watchlist_info = TDSession.get_options_chain(option_chain=params)
# print(watchlist_info)


options_list_calls = []
# Level One - Option. .WMT210312C131 to convert it WMT_031221C131
for call_key, call_value in watchlist_info.get("callExpDateMap",{}).items():
    for candle_key, candle_value in watchlist_info.get("callExpDateMap",{}).get(call_key).items():
        temp_d = {
            "symbol": watchlist_info.get("callExpDateMap",{}).get(call_key).get(candle_key, [{}])[0].get("symbol"),
            "total_volume_calls": watchlist_info.get("callExpDateMap",{}).get(call_key).get(candle_key, [{}])[0].get("totalVolume"),
            "percentage_of_change_calls": watchlist_info.get("callExpDateMap",{}).get(call_key).get(candle_key, [{}])[0].get("percentChange"),
        }
        options_list_calls.append(temp_d)

print(options_list_calls)

# puts
options_list_puts = []
options_list_puts_total_volume = []
for put_key, put_value in watchlist_info.get("putExpDateMap",{}).items():
    for candle_key, candle_value in watchlist_info.get("putExpDateMap",{}).get(put_key).items():
        temp_d = {
            "symbol": watchlist_info.get("putExpDateMap", {}).get(put_key).get(candle_key, [{}])[0].get("symbol"),
            "total_volume_puts": watchlist_info.get("putExpDateMap", {}).get(put_key).get(candle_key, [{}])[0].get(
                "totalVolume"),
            "percentage_of_change_puts": watchlist_info.get("putExpDateMap", {}).get(put_key).get(candle_key, [{}])[0].get(
                "percentChange"),
        }
        options_list_puts.append(temp_d)
        options_list_puts_total_volume.append(watchlist_info.get("putExpDateMap",{}).get(put_key).get(candle_key, [{}])[0].get("totalVolume"))

# max_volume_calls = max(options_list_calls, key=lambda x:x["total_valumn"])
# max_volume_calls_index = max(range(len(options_list_calls)), key=lambda index: options_list_calls[index]['total_volume_calls'])
# print(options_list_calls[max_volume_calls_index].get("symbol"))
print(options_list_puts)
