"""계절성(Seasonality) 전략"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from .base import BaseStrategy, StrategyResult

MONTH_LABELS = [
    "1월", "2월", "3월", "4월", "5월", "6월",
    "7월", "8월", "9월", "10월", "11월", "12월",
]


class SeasonalityStrategy(BaseStrategy):
    name = "Seasonality (계절성)"
    description = "월별 수익률 패턴을 분석하여 강세/약세 월을 식별하고 백테스트합니다."

    def run(self, df: pd.DataFrame, ticker: str, ticker_name: str) -> StrategyResult:
        monthly_df = self._compute_monthly_returns(df)
        summary = self._monthly_summary(monthly_df)
        test_results = self._statistical_tests(monthly_df)
        backtest = self._backtest(monthly_df, summary)

        figures = [
            ("월별 평균 수익률", self._plot_monthly_bar(summary, ticker, ticker_name)),
            ("연도 × 월 히트맵", self._plot_heatmap(monthly_df, ticker, ticker_name)),
            ("월별 수익률 분포", self._plot_boxplot(monthly_df, ticker, ticker_name)),
            ("연도별 누적 수익률", self._plot_cumulative(monthly_df, ticker, ticker_name)),
            ("시계열 분해", self._plot_decomposition(df, ticker, ticker_name)),
            ("백테스트 결과", self._plot_backtest(monthly_df, backtest, ticker, ticker_name)),
        ]
        figures = [(t, f) for t, f in figures if f is not None]

        kw_stat, kw_p = test_results["kruskal_wallis"]
        report = self._build_report(
            summary, test_results, backtest, ticker, ticker_name,
            df.index[0].strftime("%Y"), df.index[-1].strftime("%Y"),
        )

        # 요약 테이블 정리
        display = summary[["평균수익률(%)", "중앙값(%)", "표준편차(%)", "승률(%)", "관측수"]].round(2)

        metrics = {
            "Kruskal-Wallis p-value": f"{kw_p:.4f}",
            "계절성 유의": "✅ 유의" if kw_p < 0.05 else "❌ 비유의",
            "Buy & Hold 총수익률": f"{backtest['total_bnh']*100:+.1f}%",
            "계절성전략 총수익률": f"{backtest['total_strategy']*100:+.1f}%",
            "전략 연환산": f"{backtest['annual_strategy']*100:+.1f}%",
            "전략 승률": f"{backtest['strategy_win_rate']:.1f}%",
            "투자 대상 월": ", ".join(backtest["good_months"]),
        }

        return StrategyResult(
            summary_df=display,
            figures=figures,
            metrics=metrics,
            report_text=report,
        )

    # ── 내부 계산 ─────────────────────────────────────────

    def _compute_monthly_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        monthly_close = df["종가"].resample("ME").last()
        monthly_ret = monthly_close.pct_change().dropna()
        return pd.DataFrame({
            "year": monthly_ret.index.year,
            "month": monthly_ret.index.month,
            "return": monthly_ret.values,
        })

    def _monthly_summary(self, monthly_df: pd.DataFrame) -> pd.DataFrame:
        summary = monthly_df.groupby("month")["return"].agg(
            평균수익률="mean", 중앙값="median", 표준편차="std",
            승률=lambda x: (x > 0).mean(), 관측수="count",
        )
        summary["평균수익률(%)"] = summary["평균수익률"] * 100
        summary["중앙값(%)"] = summary["중앙값"] * 100
        summary["표준편차(%)"] = summary["표준편차"] * 100
        summary["승률(%)"] = summary["승률"] * 100
        summary.index = MONTH_LABELS
        return summary

    def _statistical_tests(self, monthly_df: pd.DataFrame) -> dict:
        groups = [g["return"].values for _, g in monthly_df.groupby("month")]
        kw_stat, kw_p = stats.kruskal(*groups)

        t_results = []
        for m in range(1, 13):
            data = monthly_df[monthly_df["month"] == m]["return"]
            t_stat, t_p = stats.ttest_1samp(data, 0)
            t_results.append({
                "월": MONTH_LABELS[m - 1],
                "t-stat": round(t_stat, 3),
                "p-value": round(t_p, 4),
                "유의": "✅" if t_p < 0.05 else "",
            })

        return {
            "kruskal_wallis": (kw_stat, kw_p),
            "t_tests": pd.DataFrame(t_results),
        }

    def _backtest(self, monthly_df: pd.DataFrame, summary: pd.DataFrame) -> dict:
        good_months = summary[
            (summary["평균수익률"] > 0) & (summary["승률(%)"] > 50)
        ].index.tolist()
        good_month_nums = [MONTH_LABELS.index(m) + 1 for m in good_months]

        mdf = monthly_df.copy()
        mdf["strategy_return"] = mdf.apply(
            lambda r: r["return"] if r["month"] in good_month_nums else 0, axis=1
        )
        mdf["cumulative_bnh"] = (1 + mdf["return"]).cumprod()
        mdf["cumulative_strategy"] = (1 + mdf["strategy_return"]).cumprod()

        total_bnh = mdf["cumulative_bnh"].iloc[-1] - 1
        total_strat = mdf["cumulative_strategy"].iloc[-1] - 1
        n_years = mdf["year"].nunique()
        ann_bnh = (1 + total_bnh) ** (1 / n_years) - 1
        ann_strat = (1 + total_strat) ** (1 / n_years) - 1

        strat_rets = mdf[mdf["strategy_return"] != 0]["strategy_return"]
        win_rate = (strat_rets > 0).mean() * 100

        return {
            "good_months": good_months,
            "good_month_nums": good_month_nums,
            "total_bnh": total_bnh,
            "total_strategy": total_strat,
            "annual_bnh": ann_bnh,
            "annual_strategy": ann_strat,
            "strategy_win_rate": win_rate,
            "monthly_df": mdf,
        }

    # ── 시각화 ────────────────────────────────────────────

    def _plot_monthly_bar(self, summary, ticker, name):
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ["#d32f2f" if v > 0 else "#1976d2" for v in summary["평균수익률"]]
        bars = ax.bar(MONTH_LABELS, summary["평균수익률"] * 100, color=colors, edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("평균 수익률 (%)")
        ax.set_title(f"{name}({ticker}) 월별 평균 수익률")
        for i, (ret, wr) in enumerate(zip(summary["평균수익률"] * 100, summary["승률(%)"])):
            # 수익률 표시 (바 위/아래)
            offset = 0.5 if ret >= 0 else -1.0
            ax.text(i, ret + offset, f"{ret:+.1f}%", ha="center", fontsize=9, fontweight="bold")
            # 승률 표시 (바 안쪽, x축 근처)
            y_wr = 0.5 if ret >= 0 else -0.5
            ax.text(i, y_wr, f"승률{wr:.0f}%", ha="center", fontsize=8, color="black")
        fig.tight_layout()
        return fig

    def _plot_heatmap(self, monthly_df, ticker, name):
        pivot = monthly_df.pivot_table(index="year", columns="month", values="return")
        pivot.columns = MONTH_LABELS
        fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.5)))
        sns.heatmap(
            pivot * 100, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
            linewidths=0.5, ax=ax, cbar_kws={"label": "수익률 (%)"},
        )
        ax.set_title(f"{name}({ticker}) 연도별 × 월별 수익률 (%)")
        ax.set_ylabel("연도")
        fig.tight_layout()
        return fig

    def _plot_boxplot(self, monthly_df, ticker, name):
        fig, ax = plt.subplots(figsize=(12, 5))
        data_by_month = [
            monthly_df[monthly_df["month"] == m]["return"].values * 100
            for m in range(1, 13)
        ]
        bp = ax.boxplot(data_by_month, labels=MONTH_LABELS, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#e3f2fd")
        ax.axhline(0, color="red", linestyle="--", linewidth=0.5)
        ax.set_ylabel("수익률 (%)")
        ax.set_title(f"{name}({ticker}) 월별 수익률 분포")
        fig.tight_layout()
        return fig

    def _plot_cumulative(self, monthly_df, ticker, name):
        fig, ax = plt.subplots(figsize=(12, 5))
        for year, grp in monthly_df.groupby("year"):
            if len(grp) < 6:
                continue
            cum = (1 + grp["return"]).cumprod() - 1
            ax.plot(grp["month"].values, cum.values * 100, alpha=0.3, label=str(year))
        avg = monthly_df.groupby("month")["return"].mean()
        cum_avg = (1 + avg).cumprod() - 1
        ax.plot(range(1, 13), cum_avg.values * 100, color="black", linewidth=2.5, label="평균")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(MONTH_LABELS)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_ylabel("누적 수익률 (%)")
        ax.set_title(f"{name}({ticker}) 연도별 누적 수익률 패턴")
        ax.legend(fontsize=7, ncol=5, loc="upper left")
        fig.tight_layout()
        return fig

    def _plot_decomposition(self, df, ticker, name):
        monthly_close = df["종가"].resample("ME").last().dropna()
        if len(monthly_close) < 24:
            return None
        result = seasonal_decompose(monthly_close, model="multiplicative", period=12)
        fig = result.plot()
        fig.set_size_inches(12, 8)
        fig.suptitle(f"{name}({ticker}) 시계열 분해", fontsize=14, y=1.02)
        fig.tight_layout()
        return fig

    def _plot_backtest(self, monthly_df, backtest, ticker, name):
        mdf = backtest["monthly_df"]
        total_bnh = backtest["total_bnh"]
        total_strat = backtest["total_strategy"]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(range(len(mdf)), mdf["cumulative_bnh"],
                label=f"Buy & Hold ({total_bnh*100:+.1f}%)", linewidth=1.5)
        ax.plot(range(len(mdf)), mdf["cumulative_strategy"],
                label=f"계절성 전략 ({total_strat*100:+.1f}%)", linewidth=1.5, color="red")
        ax.set_ylabel("누적 수익 (배수)")
        ax.set_title(f"{name}({ticker}) 계절성 전략 vs Buy & Hold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig

    # ── 리포트 ────────────────────────────────────────────

    def _build_report(self, summary, test_results, backtest, ticker, name, start_y, end_y):
        kw_stat, kw_p = test_results["kruskal_wallis"]
        now_month = datetime.today().month
        current_label = MONTH_LABELS[now_month - 1]
        cs = summary.loc[current_label]

        strong = summary[summary["평균수익률"] > 0].sort_values("평균수익률", ascending=False)
        weak = summary[summary["평균수익률"] < 0].sort_values("평균수익률")

        lines = [
            f"## {name}({ticker}) 계절성 분석 리포트",
            f"분석기간: {start_y}~{end_y}  |  생성일: {datetime.today().strftime('%Y-%m-%d')}",
            "",
            "### 계절성 존재 여부",
            f"- Kruskal-Wallis 검정: H={kw_stat:.3f}, p={kw_p:.4f}",
            f"- {'월별 수익률 분포에 통계적으로 유의한 차이 존재 (계절성 있음)' if kw_p < 0.05 else '통계적으로 유의한 계절성은 확인되지 않음 (실무적 패턴은 존재 가능)'}",
            "",
            "### 월별 패턴",
            f"- 강세 월: {', '.join(strong.index[:3])}",
            f"- 약세 월: {', '.join(weak.index[:3])}",
            "",
            "### 백테스트",
            f"- 투자 대상 월: {', '.join(backtest['good_months'])}",
            f"- Buy & Hold: 총 {backtest['total_bnh']*100:+.1f}%, 연 {backtest['annual_bnh']*100:+.1f}%",
            f"- 계절성 전략: 총 {backtest['total_strategy']*100:+.1f}%, 연 {backtest['annual_strategy']*100:+.1f}%",
            f"- 전략 승률: {backtest['strategy_win_rate']:.1f}%",
            "",
            "### 현재 시점",
            f"- 현재 월: {current_label}",
            f"- 과거 평균수익률: {cs['평균수익률(%)']:.2f}%  |  승률: {cs['승률(%)']:.0f}%",
            f"- {'계절성 관점에서 매수 유리' if cs['평균수익률'] > 0 and cs['승률(%)'] > 50 else '계절성 관점에서 매수 불리'}",
            "",
            "### 주의사항",
            "- 과거 패턴이 미래를 보장하지 않음",
            "- 펀더멘털 분석과 함께 보조 지표로 활용",
            "- 거래량, 시장 흐름, 실적 발표 일정 등도 고려 필요",
        ]
        return "\n".join(lines)
