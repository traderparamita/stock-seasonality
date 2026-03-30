"""Top 10 2분기 계절성 종목 일괄 분석"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

from data import fetch_ohlcv, get_ticker_name
from strategies.seasonality import SeasonalityStrategy

TOP10 = [
    "298040", "241710", "006340", "251970", "009470",
    "950140", "036620", "007540", "196170", "033100",
]

strategy = SeasonalityStrategy()
os.makedirs("output/top10", exist_ok=True)

for ticker in TOP10:
    name = get_ticker_name(ticker)
    print(f"\n{'='*50}")
    print(f"  {name} ({ticker})")
    print(f"{'='*50}")

    df = fetch_ohlcv(ticker, "20100101", "20260330")
    if df.empty:
        print("  데이터 없음, 건너뜀")
        continue

    result = strategy.run(df, ticker, name)

    # 차트 저장
    out_dir = f"output/top10/{ticker}_{name}"
    os.makedirs(out_dir, exist_ok=True)
    for i, (title, fig) in enumerate(result.figures):
        fig.savefig(f"{out_dir}/{i+1:02d}_{title}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 리포트 저장
    with open(f"{out_dir}/report.md", "w") as f:
        f.write(result.report_text)

    # 핵심 지표 출력
    for k, v in result.metrics.items():
        print(f"  {k}: {v}")

print("\n\n분석 완료! output/top10/ 디렉토리를 확인하세요.")
