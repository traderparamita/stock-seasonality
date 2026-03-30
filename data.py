"""공용 데이터 수집 모듈"""

import pandas as pd
from pykrx import stock


def fetch_ohlcv(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """KRX OHLCV 데이터를 가져온다. start_date/end_date: YYYYMMDD 형식."""
    df = stock.get_market_ohlcv(start_date, end_date, ticker)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def get_ticker_name(ticker: str) -> str:
    """종목코드 → 종목명. 유효하지 않으면 빈 문자열."""
    return stock.get_market_ticker_name(ticker)
