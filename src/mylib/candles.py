import pandas as pd
from datetime import timedelta, datetime, timezone
from tinkoff.invest import Client, CandleInterval, InstrumentStatus, InstrumentRequest
from tinkoff.invest.retrying.settings import RetryClientSettings
from tinkoff.invest.retrying.sync.client import RetryingClient
from tinkoff.invest.utils import now
from tinkoff.invest.services import InstrumentsService, MarketDataService

def get_price_from_q(q):
    return q.units + q.nano * 0.000000001

def get_figi(ticker, TOKEN):
    with Client(TOKEN) as cl:
        instruments: InstrumentsService = cl.instruments
        market_data: MarketDataService = cl.market_data

        l = []
        for method in ['shares', 'bonds', 'etfs']:
            for item in getattr(instruments, method)().instruments:
                l.append({
                    'ticker': item.ticker,
                    'figi': item.figi,
                    'type': method,
                    'name': item.name,
                })

        df = pd.DataFrame(l)

        df = df[df['ticker'] == ticker]
        if df.empty:
            return None

        return df['figi'].iloc[0]

def get_candles_data(ticker, tf, date_from, date_to, TOKEN, settings):
    timeframe_config = {
        '1MIN': CandleInterval.CANDLE_INTERVAL_1_MIN,
        '5MIN': CandleInterval.CANDLE_INTERVAL_5_MIN,
        '15MIN': CandleInterval.CANDLE_INTERVAL_15_MIN,
        '1HOUR': CandleInterval.CANDLE_INTERVAL_HOUR,
        '1DAY': CandleInterval.CANDLE_INTERVAL_DAY,
    }

    timeframe = timeframe_config[tf]

    figi = get_figi(ticker, TOKEN)

    data = {
        'open': [],
        'high': [],
        'low': [],
        'close': [],
        'volume': [],
        'time': [],
    }
    
    retry_settings = RetryClientSettings(use_retry=True, max_retry_attempt=2)
    
    with RetryingClient(TOKEN, settings=retry_settings) as client:
        for candle in client.get_all_candles(
                figi=figi,
                from_=date_from,
                to=date_to,
                interval=timeframe,
        ):
            data['open'].append(get_price_from_q(candle.open))
            data['high'].append(get_price_from_q(candle.high))
            data['low'].append(get_price_from_q(candle.low))
            data['close'].append(get_price_from_q(candle.close))
            data['volume'].append(candle.volume)
            data['time'].append(candle.time)

    df = pd.DataFrame(data)

    for row in df.itertuples():
        df.loc[row.Index, 'time'] = row.time + timedelta(hours=3)

    return df
