"""
파세코(037070) 계절성(Seasonality) 분석
- 월별 수익률 패턴 분석
- 통계적 유의성 검정
- 투자 전략 시뮬레이션
"""

import pandas as pd
import numpy as np
from scipy import stats
from pykrx import stock
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
import os
from datetime import datetime

warnings.filterwarnings("ignore")

# 한글 폰트 설정 (macOS)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TICKER = "037070"
TICKER_NAME = "파세코"
START_DATE = "20100101"
END_DATE = datetime.today().strftime("%Y%m%d")

MONTH_LABELS = [
    "1월", "2월", "3월", "4월", "5월", "6월",
    "7월", "8월", "9월", "10월", "11월", "12월",
]


# =============================================================================
# 1. 데이터 수집
# =============================================================================
def fetch_data():
    print(f"[1] {TICKER_NAME}({TICKER}) 데이터 수집 중 ({START_DATE} ~ {END_DATE})...")
    df = stock.get_market_ohlcv(START_DATE, END_DATE, TICKER)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    print(f"    수집 완료: {len(df)}일 ({df.index[0].date()} ~ {df.index[-1].date()})")
    return df


# =============================================================================
# 2. 월별 수익률 계산
# =============================================================================
def compute_monthly_returns(df):
    print("[2] 월별 수익률 계산 중...")
    monthly_close = df["종가"].resample("ME").last()
    monthly_ret = monthly_close.pct_change().dropna()

    monthly_df = pd.DataFrame(
        {
            "year": monthly_ret.index.year,
            "month": monthly_ret.index.month,
            "return": monthly_ret.values,
        }
    )
    return monthly_df


# =============================================================================
# 3. 월별 통계 요약
# =============================================================================
def monthly_summary(monthly_df):
    print("[3] 월별 통계 요약...")
    summary = monthly_df.groupby("month")["return"].agg(
        평균수익률="mean",
        중앙값="median",
        표준편차="std",
        승률=lambda x: (x > 0).mean(),
        관측수="count",
    )
    summary["평균수익률(%)"] = summary["평균수익률"] * 100
    summary["중앙값(%)"] = summary["중앙값"] * 100
    summary["표준편차(%)"] = summary["표준편차"] * 100
    summary["승률(%)"] = summary["승률"] * 100
    summary.index = MONTH_LABELS

    display_cols = ["평균수익률(%)", "중앙값(%)", "표준편차(%)", "승률(%)", "관측수"]
    print(summary[display_cols].round(2).to_string())
    print()
    return summary


# =============================================================================
# 4. 통계적 검정
# =============================================================================
def statistical_tests(monthly_df):
    print("[4] 통계적 유의성 검정...")

    groups = [g["return"].values for _, g in monthly_df.groupby("month")]

    # Kruskal-Wallis: 월별 수익률 분포 차이 검정
    kw_stat, kw_p = stats.kruskal(*groups)
    print(f"    Kruskal-Wallis H={kw_stat:.3f}, p={kw_p:.4f}", end="")
    print(" → 유의" if kw_p < 0.05 else " → 비유의")

    # 월별 t-test (수익률 ≠ 0 검정)
    print("\n    월별 t-test (H0: 평균수익률 = 0):")
    t_results = []
    for m in range(1, 13):
        data = monthly_df[monthly_df["month"] == m]["return"]
        t_stat, t_p = stats.ttest_1samp(data, 0)
        sig = "**" if t_p < 0.05 else "  "
        t_results.append(
            {"month": m, "t_stat": t_stat, "p_value": t_p, "significant": t_p < 0.05}
        )
        print(f"      {MONTH_LABELS[m-1]}: t={t_stat:+.3f}, p={t_p:.4f} {sig}")

    print()
    return {"kruskal_wallis": (kw_stat, kw_p), "t_tests": t_results}


# =============================================================================
# 5. 시각화
# =============================================================================
def plot_monthly_bar(summary):
    """월별 평균 수익률 바 차트"""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#d32f2f" if v > 0 else "#1976d2" for v in summary["평균수익률"]]
    ax.bar(MONTH_LABELS, summary["평균수익률"] * 100, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("평균 수익률 (%)")
    ax.set_title(f"{TICKER_NAME}({TICKER}) 월별 평균 수익률")

    for i, (ret, wr) in enumerate(
        zip(summary["평균수익률"] * 100, summary["승률(%)"])
    ):
        # 수익률 표시 (바 위/아래)
        offset = 0.5 if ret >= 0 else -1.0
        ax.text(i, ret + offset, f"{ret:+.1f}%", ha="center", fontsize=9, fontweight="bold")
        # 승률 표시 (바 안쪽, x축 근처)
        y_wr = 0.5 if ret >= 0 else -0.5
        ax.text(i, y_wr, f"승률{wr:.0f}%", ha="center", fontsize=8, color="black")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_monthly_avg_return.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    저장: {path}")


def plot_heatmap(monthly_df):
    """연도 × 월 수익률 히트맵"""
    pivot = monthly_df.pivot_table(index="year", columns="month", values="return")
    pivot.columns = MONTH_LABELS

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        pivot * 100,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "수익률 (%)"},
    )
    ax.set_title(f"{TICKER_NAME}({TICKER}) 연도별 × 월별 수익률 히트맵 (%)")
    ax.set_ylabel("연도")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    저장: {path}")


def plot_boxplot(monthly_df):
    """월별 수익률 분포 박스플롯"""
    fig, ax = plt.subplots(figsize=(12, 6))
    data_by_month = [
        monthly_df[monthly_df["month"] == m]["return"].values * 100
        for m in range(1, 13)
    ]
    bp = ax.boxplot(data_by_month, labels=MONTH_LABELS, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#e3f2fd")
    ax.axhline(0, color="red", linestyle="--", linewidth=0.5)
    ax.set_ylabel("수익률 (%)")
    ax.set_title(f"{TICKER_NAME}({TICKER}) 월별 수익률 분포")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_boxplot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    저장: {path}")


def plot_cumulative_seasonal(monthly_df):
    """연도별 누적 수익률 오버레이"""
    fig, ax = plt.subplots(figsize=(12, 6))
    for year, grp in monthly_df.groupby("year"):
        if len(grp) < 6:
            continue
        cum = (1 + grp["return"]).cumprod() - 1
        ax.plot(grp["month"].values, cum.values * 100, alpha=0.3, label=str(year))

    # 평균 누적
    avg = monthly_df.groupby("month")["return"].mean()
    cum_avg = (1 + avg).cumprod() - 1
    ax.plot(range(1, 13), cum_avg.values * 100, color="black", linewidth=2.5, label="평균")

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylabel("누적 수익률 (%)")
    ax.set_title(f"{TICKER_NAME}({TICKER}) 연도별 누적 수익률 패턴")
    ax.legend(fontsize=7, ncol=5, loc="upper left")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_cumulative_seasonal.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    저장: {path}")


def plot_seasonal_decomposition(df):
    """시계열 분해 (Trend / Seasonal / Residual)"""
    monthly_close = df["종가"].resample("ME").last().dropna()
    if len(monthly_close) < 24:
        print("    데이터 부족으로 시계열 분해 생략")
        return

    result = seasonal_decompose(monthly_close, model="multiplicative", period=12)
    fig = result.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle(f"{TICKER_NAME}({TICKER}) 시계열 분해", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "05_seasonal_decomposition.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    저장: {path}")


# =============================================================================
# 6. 계절성 기반 백테스트
# =============================================================================
def backtest_seasonal_strategy(df, monthly_df, summary):
    """
    전략: 평균수익률 > 0 이고 승률 > 50%인 월에만 투자 (매월 초 매수 → 매월 말 매도)
    벤치마크: Buy & Hold
    """
    print("[6] 계절성 기반 백테스트...")

    # 투자 대상 월 선정
    good_months = summary[
        (summary["평균수익률"] > 0) & (summary["승률(%)"] > 50)
    ].index.tolist()
    good_month_nums = [MONTH_LABELS.index(m) + 1 for m in good_months]
    print(f"    투자 대상 월: {good_months}")

    # 전략 수익률
    monthly_df = monthly_df.copy()
    monthly_df["strategy_return"] = monthly_df.apply(
        lambda r: r["return"] if r["month"] in good_month_nums else 0, axis=1
    )

    # 누적 수익률 계산
    monthly_df["cumulative_bnh"] = (1 + monthly_df["return"]).cumprod()
    monthly_df["cumulative_strategy"] = (1 + monthly_df["strategy_return"]).cumprod()

    # 결과 요약
    total_bnh = monthly_df["cumulative_bnh"].iloc[-1] - 1
    total_strat = monthly_df["cumulative_strategy"].iloc[-1] - 1
    n_years = monthly_df["year"].nunique()

    ann_bnh = (1 + total_bnh) ** (1 / n_years) - 1
    ann_strat = (1 + total_strat) ** (1 / n_years) - 1

    strat_returns = monthly_df[monthly_df["strategy_return"] != 0]["strategy_return"]
    strat_win_rate = (strat_returns > 0).mean() * 100

    print(f"\n    === 백테스트 결과 ({monthly_df['year'].min()}~{monthly_df['year'].max()}) ===")
    print(f"    Buy & Hold  : 총 수익률 {total_bnh*100:+.1f}%, 연환산 {ann_bnh*100:+.1f}%")
    print(f"    계절성 전략 : 총 수익률 {total_strat*100:+.1f}%, 연환산 {ann_strat*100:+.1f}%")
    print(f"    전략 승률   : {strat_win_rate:.1f}%")
    print(f"    투자 기간   : {len(good_month_nums)}개월/년 (나머지 현금 보유)")
    print()

    # 시각화
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        monthly_df.index,
        monthly_df["cumulative_bnh"],
        label=f"Buy & Hold ({total_bnh*100:+.1f}%)",
        linewidth=1.5,
    )
    ax.plot(
        monthly_df.index,
        monthly_df["cumulative_strategy"],
        label=f"계절성 전략 ({total_strat*100:+.1f}%)",
        linewidth=1.5,
        color="red",
    )
    ax.set_ylabel("누적 수익 (배수)")
    ax.set_title(f"{TICKER_NAME}({TICKER}) 계절성 전략 vs Buy & Hold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "06_backtest.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    저장: {path}")

    return {
        "good_months": good_months,
        "total_bnh": total_bnh,
        "total_strategy": total_strat,
        "annual_bnh": ann_bnh,
        "annual_strategy": ann_strat,
        "strategy_win_rate": strat_win_rate,
    }


# =============================================================================
# 7. 투자 판단 리포트
# =============================================================================
def generate_report(summary, test_results, backtest_results):
    print("[7] 투자 판단 리포트 생성...")

    kw_stat, kw_p = test_results["kruskal_wallis"]
    now_month = datetime.today().month

    lines = []
    lines.append(f"{'='*60}")
    lines.append(f" {TICKER_NAME}({TICKER}) 계절성 분석 리포트")
    lines.append(f" 분석기간: {START_DATE[:4]}~{END_DATE[:4]}")
    lines.append(f" 생성일: {datetime.today().strftime('%Y-%m-%d')}")
    lines.append(f"{'='*60}")
    lines.append("")

    lines.append("[ 계절성 존재 여부 ]")
    lines.append(f"  Kruskal-Wallis 검정: H={kw_stat:.3f}, p={kw_p:.4f}")
    if kw_p < 0.05:
        lines.append("  → 월별 수익률 분포에 통계적으로 유의한 차이가 있음 (계절성 존재)")
    else:
        lines.append("  → 통계적으로 유의한 계절성은 확인되지 않음")
        lines.append("    (그러나 실무적 패턴은 존재할 수 있음)")
    lines.append("")

    lines.append("[ 월별 패턴 요약 ]")
    # 강세 월
    strong = summary[summary["평균수익률"] > 0].sort_values("평균수익률", ascending=False)
    lines.append(f"  강세 월: {', '.join(strong.index[:3])} (평균수익률 양수 + 높은 승률)")
    weak = summary[summary["평균수익률"] < 0].sort_values("평균수익률")
    lines.append(f"  약세 월: {', '.join(weak.index[:3])} (평균수익률 음수)")
    lines.append("")

    lines.append("[ 백테스트 결과 ]")
    lines.append(f"  투자 대상 월: {', '.join(backtest_results['good_months'])}")
    lines.append(
        f"  Buy & Hold : 총 {backtest_results['total_bnh']*100:+.1f}%, "
        f"연 {backtest_results['annual_bnh']*100:+.1f}%"
    )
    lines.append(
        f"  계절성 전략: 총 {backtest_results['total_strategy']*100:+.1f}%, "
        f"연 {backtest_results['annual_strategy']*100:+.1f}%"
    )
    lines.append(f"  전략 승률  : {backtest_results['strategy_win_rate']:.1f}%")
    lines.append("")

    lines.append("[ 현재 시점 투자 판단 ]")
    current_label = MONTH_LABELS[now_month - 1]
    current_stats = summary.loc[current_label]
    lines.append(f"  현재 월: {current_label}")
    lines.append(f"  과거 평균수익률: {current_stats['평균수익률(%)']: .2f}%")
    lines.append(f"  과거 승률: {current_stats['승률(%)']:.0f}%")
    if current_stats["평균수익률"] > 0 and current_stats["승률(%)"] > 50:
        lines.append(f"  → 계절성 관점에서 {current_label}은 매수 유리한 시기")
    else:
        lines.append(f"  → 계절성 관점에서 {current_label}은 매수에 불리한 시기")
    lines.append("")

    lines.append("[ 주의사항 ]")
    lines.append("  - 과거 패턴이 미래를 보장하지 않음")
    lines.append("  - 계절성은 펀더멘털 분석과 함께 보조 지표로 활용")
    lines.append("  - 파세코는 난방/냉방 기기 회사로 계절적 수요 변동이 실적에 반영될 수 있음")
    lines.append("  - 거래량, 시장 전체 흐름, 실적 발표 일정 등도 고려 필요")
    lines.append(f"{'='*60}")

    report = "\n".join(lines)
    path = os.path.join(OUTPUT_DIR, "report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"    저장: {path}")
    print()
    print(report)


# =============================================================================
# Main
# =============================================================================
def main():
    print(f"\n{'='*60}")
    print(f"  {TICKER_NAME}({TICKER}) 계절성(Seasonality) 분석")
    print(f"{'='*60}\n")

    # 1. 데이터 수집
    df = fetch_data()

    # 2. 월별 수익률
    monthly_df = compute_monthly_returns(df)

    # 3. 통계 요약
    summary = monthly_summary(monthly_df)

    # 4. 통계 검정
    test_results = statistical_tests(monthly_df)

    # 5. 시각화
    print("[5] 차트 생성 중...")
    plot_monthly_bar(summary)
    plot_heatmap(monthly_df)
    plot_boxplot(monthly_df)
    plot_cumulative_seasonal(monthly_df)
    plot_seasonal_decomposition(df)

    # 6. 백테스트
    backtest_results = backtest_seasonal_strategy(df, monthly_df, summary)

    # 7. 리포트
    generate_report(summary, test_results, backtest_results)

    print("\n분석 완료! output/ 디렉토리에서 결과를 확인하세요.\n")


if __name__ == "__main__":
    main()
