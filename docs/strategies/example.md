# Example Trading Strategy

## Overview

This example strategy uses simple and easy to understand indicators that define a trend in the stock price.
It is primarily an options trading strategy.

## Indicators

1. 9-Day SMA (Simple Moving Average), represents the short-term trend.
2. 50-Day SMA (Simple Moving Average), represents the medium-term trend.
3. 200-Day SMA (Simple Moving Average), represents the long-term trend.
4. 50-Day Volume Average

## Buy Signal & Sell Signal

**BUY CALLS CONDITION**

1. 9-Day SMA Increasing (Slope > 0)
2. 50-Day Average Volume Increasing (Slope > 0)
3. 9-Day SMA **Above** 50-Day SMA

**BUY PUTS CONDITION**

1. 9-Day SMA Decreasing (Slope < 0)
2. 50-Day Average Volume Decreasing (Slope < 0)
3. 9-Day SMA **BELOW** 50-Day SMA
4. 50-Day SMA **BELOW** 200-Day SMA

**SELL CONDITION**

When neither of these conditions exist, we should sell all options that we are holding.

## How to use The Strategy

Code it in a bot.

## Python Code

```python
# Create an indicator Object.
indicator_client = Indicators(price_data_frame=stock_frame)

# Add the 200-Day simple moving average.
indicator_client.sma(period=200)

# Add the 50-Day simple moving average.
indicator_client.sma(period=200)

# Add a signal to check for.
indicator_client.set_indicator_signal_compare(
    indicator_1='sma',
    indicator_2='sma',
    condition_buy=operator.ge,
    condition_sell=operator.le,
)
```
