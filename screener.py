"""
계절성 스크리너 — 특정 기간(분기/월)에 수익률이 좋은 종목을 탐색
사용법: python3.12 screener.py
"""

import pandas as pd
import numpy as np
from pykrx import stock
import FinanceDataReader as fdr
from datetime import datetime
import time
import os

# ── 설정 ──────────────────────────────────────────────────
TARGET_MONTHS = [4, 5, 6]          # 2분기
TARGET_LABEL = "2분기(4~6월)"
START_DATE = "20150101"
END_DATE = datetime.today().strftime("%Y%m%d")

MIN_YEARS = 5                      # 최소 관측 연수
MIN_AVG_RETURN = 3.0               # 분기 평균수익률 하한 (%)
MIN_WIN_RATE = 60.0                # 분기 승률 하한 (%)
MIN_MARKET_CAP = 500               # 최소 시가총액 (억원)

OUTPUT_PATH = "output/screener_q2.csv"


def get_tickers_with_cap():
    """KOSPI + KOSDAQ 종목코드 + 시가총액 (FinanceDataReader 사용)"""
    frames = []
    for market in ["KOSPI", "KOSDAQ"]:
        df = fdr.StockListing(market)
        df = df[["Code", "Name", "Market", "Marcap"]].copy()
        df.columns = ["ticker", "name", "market", "marcap"]
        df["market_label"] = market
        df["market_cap"] = df["marcap"] / 1e8  # 억원
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df[all_df["market_cap"] >= MIN_MARKET_CAP].reset_index(drop=True)
    print(f"시가총액 {MIN_MARKET_CAP}억원 이상 종목: {len(all_df)}개")
    return all_df


def analyze_seasonality(ticker, target_months):
    """특정 종목의 target_months 계절성 분석"""
    try:
        df = stock.get_market_ohlcv(START_DATE, END_DATE, ticker)
        if len(df) < 250:
            return None

        df.index = pd.to_datetime(df.index)
        monthly_close = df["종가"].resample("ME").last()
        monthly_ret = monthly_close.pct_change().dropna()

        mdf = pd.DataFrame({
            "year": monthly_ret.index.year,
            "month": monthly_ret.index.month,
            "return": monthly_ret.values,
        })

        # 타겟 월 필터
        target = mdf[mdf["month"].isin(target_months)]
        if len(target) < MIN_YEARS * len(target_months) * 0.5:
            return None

        # 분기별 수익률 (각 연도의 타겟 월 합산)
        quarterly = target.groupby("year")["return"].apply(
            lambda x: (1 + x).prod() - 1
        )

        if len(quarterly) < MIN_YEARS:
            return None

        avg_ret = quarterly.mean() * 100
        med_ret = quarterly.median() * 100
        win_rate = (quarterly > 0).mean() * 100
        std_ret = quarterly.std() * 100
        n_years = len(quarterly)

        # 월별 평균
        monthly_avg = target.groupby("month")["return"].mean() * 100

        return {
            "avg_return": avg_ret,
            "median_return": med_ret,
            "win_rate": win_rate,
            "std": std_ret,
            "n_years": n_years,
            "monthly_avg": monthly_avg.to_dict(),
        }
    except Exception:
        return None


def main():
    print(f"\n{'='*60}")
    print(f"  계절성 스크리너: {TARGET_LABEL} 강세 종목 탐색")
    print(f"  분석기간: {START_DATE[:4]}~{END_DATE[:4]}")
    print(f"  조건: 평균수익률>{MIN_AVG_RETURN}%, 승률>{MIN_WIN_RATE}%")
    print(f"{'='*60}\n")

    # 1. 종목 리스트
    print("[1/3] 종목 리스트 수집 중...")
    tickers_df = get_tickers_with_cap()

    # 2. 스크리닝
    total = len(tickers_df)
    print(f"\n[2/3] {total}개 종목 계절성 분석 중...")
    results = []

    for idx, row in tickers_df.iterrows():
        if idx % 100 == 0:
            print(f"    진행: {idx}/{total} ({idx/total*100:.0f}%)")

        res = analyze_seasonality(row["ticker"], TARGET_MONTHS)
        if res is None:
            continue

        if res["avg_return"] >= MIN_AVG_RETURN and res["win_rate"] >= MIN_WIN_RATE:
            entry = {
                "종목코드": row["ticker"],
                "종목명": row["name"],
                "시장": row["market_label"],
                "시가총액(억)": round(row["market_cap"]),
                f"{TARGET_LABEL} 평균수익률(%)": round(res["avg_return"], 2),
                f"{TARGET_LABEL} 중앙값(%)": round(res["median_return"], 2),
                f"{TARGET_LABEL} 승률(%)": round(res["win_rate"], 1),
                f"{TARGET_LABEL} 표준편차(%)": round(res["std"], 2),
                "관측연수": res["n_years"],
            }
            for m in TARGET_MONTHS:
                entry[f"{m}월 평균(%)"] = round(res["monthly_avg"].get(m, 0), 2)
            results.append(entry)

        # pykrx 부하 방지
        if idx % 10 == 0:
            time.sleep(0.2)

    print(f"    진행: {total}/{total} (100%)")

    # 3. 결과 정리
    print(f"\n[3/3] 결과 정리...")
    if not results:
        print("조건을 만족하는 종목이 없습니다.")
        return

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(
        f"{TARGET_LABEL} 평균수익률(%)", ascending=False
    ).reset_index(drop=True)

    # 저장
    os.makedirs("output", exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    # 출력
    print(f"\n{'='*60}")
    print(f"  {TARGET_LABEL} 계절성 강세 종목 TOP 30")
    print(f"  조건: 평균수익률>{MIN_AVG_RETURN}%, 승률>{MIN_WIN_RATE}%, 시총>{MIN_MARKET_CAP}억")
    print(f"{'='*60}\n")

    display_cols = [
        "종목코드", "종목명", "시장", "시가총액(억)",
        f"{TARGET_LABEL} 평균수익률(%)", f"{TARGET_LABEL} 승률(%)",
        "4월 평균(%)", "5월 평균(%)", "6월 평균(%)", "관측연수",
    ]
    top30 = result_df[display_cols].head(30)
    print(top30.to_string(index=False))

    print(f"\n총 {len(result_df)}개 종목 발견 → {OUTPUT_PATH} 저장 완료")


if __name__ == "__main__":
    main()
