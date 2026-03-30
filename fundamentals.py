"""종목 소개 및 재무 분석 모듈"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from dataclasses import dataclass, field
from matplotlib.figure import Figure

BILLION = 1e9
HUNDRED_MILLION = 1e8  # 억


@dataclass
class FundamentalResult:
    company_info: dict = field(default_factory=dict)
    financials_df: pd.DataFrame | None = None
    quarterly_df: pd.DataFrame | None = None
    figures: list[tuple[str, Figure]] = field(default_factory=list)
    metrics: dict[str, str] = field(default_factory=dict)
    report_text: str = ""


def _safe_get(info: dict, key: str, default=None):
    v = info.get(key)
    return v if v is not None else default


def _fmt_krw(val, unit="억"):
    """숫자를 한국식 억/조 단위로 포맷"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    v = val / HUNDRED_MILLION
    if abs(v) >= 10000:
        return f"{v/10000:,.1f}조"
    return f"{v:,.0f}억"


def analyze_fundamentals(ticker: str, ticker_name: str, market: str = "KQ") -> FundamentalResult:
    """
    yfinance를 사용한 종목 소개 + 재무 분석
    market: "KQ" (KOSDAQ) or "KS" (KOSPI)
    """
    yf_ticker = f"{ticker}.{market}"
    t = yf.Ticker(yf_ticker)

    # 두 시장 모두 시도
    info = t.info or {}
    if not info.get("longBusinessSummary"):
        alt_market = "KS" if market == "KQ" else "KQ"
        t = yf.Ticker(f"{ticker}.{alt_market}")
        info = t.info or {}

    result = FundamentalResult()

    # ── 1. 기업 개요 ──
    result.company_info = {
        "종목명": ticker_name,
        "종목코드": ticker,
        "섹터": _safe_get(info, "sector", "N/A"),
        "산업": _safe_get(info, "industry", "N/A"),
        "사업 소개": _safe_get(info, "longBusinessSummary", "정보 없음"),
        "시가총액": _fmt_krw(_safe_get(info, "marketCap", 0)),
        "52주 최고": f"{_safe_get(info, 'fiftyTwoWeekHigh', 'N/A'):,}" if isinstance(_safe_get(info, 'fiftyTwoWeekHigh'), (int, float)) else "N/A",
        "52주 최저": f"{_safe_get(info, 'fiftyTwoWeekLow', 'N/A'):,}" if isinstance(_safe_get(info, 'fiftyTwoWeekLow'), (int, float)) else "N/A",
        "직원수": f"{_safe_get(info, 'fullTimeEmployees', 'N/A'):,}" if isinstance(_safe_get(info, 'fullTimeEmployees'), (int, float)) else "N/A",
    }

    # ── 핵심 지표 ──
    result.metrics = {
        "시가총액": _fmt_krw(_safe_get(info, "marketCap", 0)),
        "PER": f"{_safe_get(info, 'trailingPE', 'N/A')}",
        "PBR": f"{_safe_get(info, 'priceToBook', 'N/A')}",
        "ROE": f"{_safe_get(info, 'returnOnEquity', 0) * 100:.1f}%" if _safe_get(info, 'returnOnEquity') else "N/A",
        "배당수익률": f"{_safe_get(info, 'dividendYield', 0):.2f}%" if _safe_get(info, 'dividendYield') else "N/A",
        "매출성장률": f"{_safe_get(info, 'revenueGrowth', 0) * 100:+.1f}%" if _safe_get(info, 'revenueGrowth') else "N/A",
    }

    # ── 2. 연간 재무제표 ──
    try:
        financials = t.financials
        balance = t.balance_sheet
        cashflow = t.cashflow

        if financials is not None and not financials.empty:
            years = financials.columns
            rows = {}

            def _get_row(df, key):
                if df is not None and key in df.index:
                    return df.loc[key]
                return pd.Series([np.nan] * len(years), index=years)

            rows["매출액"] = _get_row(financials, "Total Revenue")
            rows["영업이익"] = _get_row(financials, "Operating Income")
            rows["순이익"] = _get_row(financials, "Net Income")
            rows["EBITDA"] = _get_row(financials, "EBITDA")
            rows["EPS"] = _get_row(financials, "Basic EPS")

            if balance is not None and not balance.empty:
                rows["총자산"] = _get_row(balance, "Total Assets")
                rows["총부채"] = _get_row(balance, "Total Debt")
                rows["자기자본"] = _get_row(balance, "Stockholders Equity")

            if cashflow is not None and not cashflow.empty:
                rows["영업CF"] = _get_row(cashflow, "Operating Cash Flow")
                rows["FCF"] = _get_row(cashflow, "Free Cash Flow")

            fin_df = pd.DataFrame(rows, index=years).T
            fin_df.columns = [c.strftime("%Y") for c in fin_df.columns]
            result.financials_df = fin_df

            # ── 차트: 매출/영업이익/순이익 추이 ──
            result.figures.append(
                ("매출 · 영업이익 · 순이익 추이", _plot_income(financials, ticker_name, ticker))
            )

            # ── 차트: 수익성 마진 ──
            margin_fig = _plot_margins(financials, ticker_name, ticker)
            if margin_fig:
                result.figures.append(("수익성 마진 추이", margin_fig))

    except Exception:
        pass

    # ── 3. 분기별 실적 ──
    try:
        q_fin = t.quarterly_financials
        if q_fin is not None and not q_fin.empty:
            q_rows = {}
            q_years = q_fin.columns

            def _qget(key):
                if key in q_fin.index:
                    return q_fin.loc[key]
                return pd.Series([np.nan] * len(q_years), index=q_years)

            q_rows["매출액"] = _qget("Total Revenue")
            q_rows["영업이익"] = _qget("Operating Income")
            q_rows["순이익"] = _qget("Net Income")

            q_df = pd.DataFrame(q_rows, index=q_years).T
            q_df.columns = [c.strftime("%Y.%m") for c in q_df.columns]
            result.quarterly_df = q_df

            # ── 차트: 분기별 매출/영업이익 ──
            result.figures.append(
                ("분기별 매출 · 영업이익", _plot_quarterly(q_fin, ticker_name, ticker))
            )

    except Exception:
        pass

    # ── 4. 주가 + 거래량 차트 ──
    try:
        hist = t.history(period="5y")
        if not hist.empty:
            result.figures.append(
                ("5년 주가 · 거래량", _plot_price_volume(hist, ticker_name, ticker))
            )
    except Exception:
        pass

    # ── 5. 재무 평가 보고서 생성 ──
    result.report_text = _generate_report(result, info)

    return result


# ── 보고서 생성 ───────────────────────────────────────────

def _generate_report(result: FundamentalResult, info: dict) -> str:
    """재무 데이터 기반 평가 보고서 자동 생성"""
    ci = result.company_info
    lines = []

    lines.append(f"## {ci['종목명']}({ci['종목코드']}) 재무 분석 보고서")
    lines.append("")

    # ── 1. 기업 개요 ──
    lines.append("### 1. 기업 개요")
    lines.append(f"- **섹터/산업**: {ci.get('섹터', 'N/A')} / {ci.get('산업', 'N/A')}")
    lines.append(f"- **시가총액**: {ci.get('시가총액', 'N/A')}")
    lines.append(f"- **52주 범위**: {ci.get('52주 최저', 'N/A')} ~ {ci.get('52주 최고', 'N/A')}")
    summary = ci.get("사업 소개", "")
    if summary and summary != "정보 없음":
        lines.append(f"- **사업 내용**: {summary}")
    lines.append("")

    # ── 2. 밸류에이션 평가 ──
    lines.append("### 2. 밸류에이션 평가")

    per = _safe_get(info, "trailingPE")
    pbr = _safe_get(info, "priceToBook")
    roe = _safe_get(info, "returnOnEquity")
    div_yield = _safe_get(info, "dividendYield")

    if per and isinstance(per, (int, float)):
        if per < 0:
            lines.append(f"- **PER {per:.1f}배**: 적자 상태로 PER 의미 제한적")
        elif per < 10:
            lines.append(f"- **PER {per:.1f}배**: 저평가 구간. 시장 대비 할인 거래 중")
        elif per < 20:
            lines.append(f"- **PER {per:.1f}배**: 적정 밸류에이션 구간")
        else:
            lines.append(f"- **PER {per:.1f}배**: 고평가 구간. 성장 기대감이 반영된 수준")
    else:
        lines.append("- **PER**: 데이터 없음 (적자 가능성)")

    if pbr and isinstance(pbr, (int, float)):
        if pbr < 1:
            lines.append(f"- **PBR {pbr:.2f}배**: 순자산 대비 할인 거래 (자산가치주)")
        elif pbr < 3:
            lines.append(f"- **PBR {pbr:.2f}배**: 적정 수준")
        else:
            lines.append(f"- **PBR {pbr:.2f}배**: 프리미엄 거래. 무형자산/성장성 반영")
    else:
        lines.append("- **PBR**: 데이터 없음")

    if div_yield and isinstance(div_yield, (int, float)) and div_yield > 0:
        dv = div_yield  # yfinance returns already in percent
        if dv >= 3:
            lines.append(f"- **배당수익률 {dv:.2f}%**: 고배당주. 안정적 현금 수익 기대")
        elif dv >= 1:
            lines.append(f"- **배당수익률 {dv:.2f}%**: 보통 수준")
        else:
            lines.append(f"- **배당수익률 {dv:.2f}%**: 저배당. 성장 재투자 중심")
    lines.append("")

    # ── 3. 수익성 평가 ──
    lines.append("### 3. 수익성 평가")

    if roe and isinstance(roe, (int, float)):
        roe_pct = roe * 100
        if roe_pct >= 15:
            lines.append(f"- **ROE {roe_pct:.1f}%**: 우수. 자기자본 대비 높은 수익 창출력")
        elif roe_pct >= 8:
            lines.append(f"- **ROE {roe_pct:.1f}%**: 양호. 업종 평균 수준")
        elif roe_pct > 0:
            lines.append(f"- **ROE {roe_pct:.1f}%**: 저조. 수익성 개선 필요")
        else:
            lines.append(f"- **ROE {roe_pct:.1f}%**: 자기자본 잠식 우려")
    else:
        lines.append("- **ROE**: 데이터 없음")

    # 재무제표 기반 분석
    fin_df = result.financials_df
    if fin_df is not None and not fin_df.empty:
        cols = fin_df.columns.tolist()  # 최신 → 과거 순

        def _get_vals(row_name):
            if row_name in fin_df.index:
                return [fin_df.loc[row_name, c] for c in cols]
            return None

        revenue_vals = _get_vals("매출액")
        op_vals = _get_vals("영업이익")
        net_vals = _get_vals("순이익")

        if revenue_vals and len(revenue_vals) >= 2:
            # 매출 추이
            latest_rev = revenue_vals[0]
            prev_rev = revenue_vals[1]
            if pd.notna(latest_rev) and pd.notna(prev_rev) and prev_rev != 0:
                rev_growth = (latest_rev - prev_rev) / abs(prev_rev) * 100
                if rev_growth > 10:
                    lines.append(f"- **매출 성장률 {rev_growth:+.1f}%** ({cols[1]}→{cols[0]}): 양호한 성장세")
                elif rev_growth > 0:
                    lines.append(f"- **매출 성장률 {rev_growth:+.1f}%** ({cols[1]}→{cols[0]}): 소폭 성장")
                elif rev_growth > -10:
                    lines.append(f"- **매출 성장률 {rev_growth:+.1f}%** ({cols[1]}→{cols[0]}): 소폭 역성장. 주의 필요")
                else:
                    lines.append(f"- **매출 성장률 {rev_growth:+.1f}%** ({cols[1]}→{cols[0]}): 큰 폭 역성장. 실적 부진 우려")

            # 영업이익률
            if op_vals and pd.notna(op_vals[0]) and pd.notna(latest_rev) and latest_rev != 0:
                op_margin = op_vals[0] / latest_rev * 100
                if op_margin >= 15:
                    lines.append(f"- **영업이익률 {op_margin:.1f}%**: 높은 수익성. 원가/비용 관리 우수")
                elif op_margin >= 5:
                    lines.append(f"- **영업이익률 {op_margin:.1f}%**: 보통 수준")
                elif op_margin > 0:
                    lines.append(f"- **영업이익률 {op_margin:.1f}%**: 낮은 수익성. 마진 개선 필요")
                else:
                    lines.append(f"- **영업이익률 {op_margin:.1f}%**: 영업적자 상태")

            # 순이익 추세 (연속 흑자/적자)
            if net_vals:
                valid_net = [(c, v) for c, v in zip(cols, net_vals) if pd.notna(v)]
                if len(valid_net) >= 2:
                    profit_years = sum(1 for _, v in valid_net if v > 0)
                    loss_years = sum(1 for _, v in valid_net if v <= 0)
                    if loss_years == 0:
                        lines.append(f"- **순이익**: {len(valid_net)}년 연속 흑자 유지")
                    elif profit_years == 0:
                        lines.append(f"- **순이익**: {len(valid_net)}년 연속 적자. 턴어라운드 필요")
                    else:
                        lines.append(f"- **순이익**: {len(valid_net)}년 중 {profit_years}년 흑자, {loss_years}년 적자")
        lines.append("")

        # ── 4. 재무 안정성 ──
        lines.append("### 4. 재무 안정성")

        debt_vals = _get_vals("총부채")
        equity_vals = _get_vals("자기자본")
        asset_vals = _get_vals("총자산")

        if debt_vals and equity_vals and pd.notna(debt_vals[0]) and pd.notna(equity_vals[0]) and equity_vals[0] != 0:
            debt_ratio = debt_vals[0] / equity_vals[0] * 100
            if debt_ratio < 50:
                lines.append(f"- **부채비율 {debt_ratio:.0f}%**: 매우 안정적. 재무 건전성 우수")
            elif debt_ratio < 100:
                lines.append(f"- **부채비율 {debt_ratio:.0f}%**: 양호. 적정 레버리지 수준")
            elif debt_ratio < 200:
                lines.append(f"- **부채비율 {debt_ratio:.0f}%**: 다소 높음. 재무 부담 주의")
            else:
                lines.append(f"- **부채비율 {debt_ratio:.0f}%**: 위험 수준. 재무구조 개선 시급")
        else:
            lines.append("- **부채비율**: 데이터 부족")

        fcf_vals = _get_vals("FCF")
        opcf_vals = _get_vals("영업CF")

        if fcf_vals:
            valid_fcf = [(c, v) for c, v in zip(cols, fcf_vals) if pd.notna(v)]
            if valid_fcf:
                latest_fcf = valid_fcf[0][1]
                fcf_positive = sum(1 for _, v in valid_fcf if v > 0)
                if latest_fcf > 0:
                    lines.append(f"- **FCF {latest_fcf / HUNDRED_MILLION:,.0f}억원**: 양(+)의 잉여현금흐름. 배당/투자 여력 확보")
                else:
                    lines.append(f"- **FCF {latest_fcf / HUNDRED_MILLION:,.0f}억원**: 음(-)의 잉여현금흐름. 투자 확대 or 현금유출 주의")
                lines.append(f"- **FCF 안정성**: {len(valid_fcf)}년 중 {fcf_positive}년 양(+)의 FCF")

        if opcf_vals:
            valid_opcf = [(c, v) for c, v in zip(cols, opcf_vals) if pd.notna(v)]
            if valid_opcf and net_vals:
                valid_net_for_accrual = [(c, v) for c, v in zip(cols, net_vals) if pd.notna(v)]
                if valid_opcf and valid_net_for_accrual:
                    lo, ln = valid_opcf[0][1], valid_net_for_accrual[0][1]
                    if pd.notna(lo) and pd.notna(ln) and ln != 0:
                        accrual_ratio = lo / ln
                        if accrual_ratio >= 1:
                            lines.append(f"- **이익의 질**: 영업CF/순이익 = {accrual_ratio:.1f}배. 현금 기반 이익으로 질적 수준 양호")
                        else:
                            lines.append(f"- **이익의 질**: 영업CF/순이익 = {accrual_ratio:.1f}배. 발생 기준 이익 비중 높아 주의")
        lines.append("")

    # ── 5. 분기 실적 모멘텀 ──
    q_df = result.quarterly_df
    if q_df is not None and not q_df.empty:
        lines.append("### 5. 분기 실적 모멘텀")
        q_cols = q_df.columns.tolist()

        if "매출액" in q_df.index and len(q_cols) >= 5:
            # 최근 분기 vs 전년 동기
            latest_q_rev = q_df.loc["매출액", q_cols[0]]
            yoy_q_rev = q_df.loc["매출액", q_cols[4]] if len(q_cols) > 4 else np.nan

            if pd.notna(latest_q_rev) and pd.notna(yoy_q_rev) and yoy_q_rev != 0:
                yoy_growth = (latest_q_rev - yoy_q_rev) / abs(yoy_q_rev) * 100
                lines.append(f"- **매출 YoY**: {yoy_growth:+.1f}% ({q_cols[4]} → {q_cols[0]})")

        if "영업이익" in q_df.index and len(q_cols) >= 2:
            latest_q_op = q_df.loc["영업이익", q_cols[0]]
            prev_q_op = q_df.loc["영업이익", q_cols[1]]
            if pd.notna(latest_q_op) and pd.notna(prev_q_op):
                if latest_q_op > 0 and prev_q_op > 0 and latest_q_op > prev_q_op:
                    lines.append(f"- **영업이익 QoQ 개선**: {prev_q_op/HUNDRED_MILLION:,.0f}억 → {latest_q_op/HUNDRED_MILLION:,.0f}억")
                elif latest_q_op > 0 and prev_q_op <= 0:
                    lines.append(f"- **흑자전환**: 영업이익 {latest_q_op/HUNDRED_MILLION:,.0f}억 (전분기 적자에서 회복)")
                elif latest_q_op <= 0:
                    lines.append(f"- **영업적자 지속/전환**: {latest_q_op/HUNDRED_MILLION:,.0f}억. 실적 부진")

            # 분기 영업이익 추세 (최근 4분기)
            op_trend = []
            for c in q_cols[:4]:
                v = q_df.loc["영업이익", c] if "영업이익" in q_df.index else np.nan
                if pd.notna(v):
                    op_trend.append(v)
            if len(op_trend) >= 3:
                improving = all(op_trend[i] >= op_trend[i+1] for i in range(len(op_trend)-1))
                declining = all(op_trend[i] <= op_trend[i+1] for i in range(len(op_trend)-1))
                if improving:
                    lines.append("- **추세**: 최근 분기 영업이익 연속 개선 중")
                elif declining:
                    lines.append("- **추세**: 최근 분기 영업이익 연속 하락 중. 주의")
        lines.append("")

    # ── 6. 종합 의견 ──
    lines.append("### 6. 종합 의견")

    # 점수 기반 간단 평가
    score = 0
    reasons_pos = []
    reasons_neg = []

    if per and isinstance(per, (int, float)):
        if 0 < per < 15:
            score += 1
            reasons_pos.append("합리적 PER")
        elif per > 30:
            score -= 1
            reasons_neg.append("높은 PER")

    if roe and isinstance(roe, (int, float)):
        if roe > 0.12:
            score += 1
            reasons_pos.append("높은 ROE")
        elif roe < 0:
            score -= 1
            reasons_neg.append("음(-)의 ROE")

    rev_growth_val = _safe_get(info, "revenueGrowth")
    if rev_growth_val and isinstance(rev_growth_val, (int, float)):
        if rev_growth_val > 0.05:
            score += 1
            reasons_pos.append("매출 성장")
        elif rev_growth_val < -0.1:
            score -= 1
            reasons_neg.append("매출 역성장")

    if fin_df is not None:
        debt_v = _get_vals("총부채")
        eq_v = _get_vals("자기자본")
        if debt_v and eq_v and pd.notna(debt_v[0]) and pd.notna(eq_v[0]) and eq_v[0] > 0:
            dr = debt_v[0] / eq_v[0]
            if dr < 0.5:
                score += 1
                reasons_pos.append("낮은 부채비율")
            elif dr > 2:
                score -= 1
                reasons_neg.append("높은 부채비율")

        fcf_v = _get_vals("FCF")
        if fcf_v and pd.notna(fcf_v[0]) and fcf_v[0] > 0:
            score += 1
            reasons_pos.append("양(+)의 FCF")

    if score >= 3:
        grade = "긍정적"
        comment = "재무 건전성과 수익성이 양호하며, 투자 매력이 있는 종목입니다."
    elif score >= 1:
        grade = "중립"
        comment = "일부 긍정적 요소가 있으나, 추가 검토가 필요한 종목입니다."
    elif score >= -1:
        grade = "주의"
        comment = "재무적 리스크가 일부 존재합니다. 신중한 접근이 필요합니다."
    else:
        grade = "부정적"
        comment = "재무 상태가 불안정합니다. 투자 시 높은 리스크를 감수해야 합니다."

    lines.append(f"- **종합 등급**: {grade}")
    lines.append(f"- **평가**: {comment}")
    if reasons_pos:
        lines.append(f"- **긍정 요인**: {', '.join(reasons_pos)}")
    if reasons_neg:
        lines.append(f"- **부정 요인**: {', '.join(reasons_neg)}")
    lines.append("")
    lines.append("> *본 보고서는 공시된 재무 데이터에 기반한 자동 분석이며, 투자 판단의 참고 자료로만 활용하시기 바랍니다.*")

    return "\n".join(lines)


# ── 시각화 함수 ───────────────────────────────────────────

def _plot_income(financials, name, ticker):
    years = [c.strftime("%Y") for c in financials.columns][::-1]

    def _vals(key):
        if key in financials.index:
            return (financials.loc[key].values / HUNDRED_MILLION)[::-1]
        return [0] * len(years)

    revenue = _vals("Total Revenue")
    op_income = _vals("Operating Income")
    net_income = _vals("Net Income")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(years))
    w = 0.25

    ax.bar(x - w, revenue, w, label="매출액", color="#42a5f5")
    ax.bar(x, op_income, w, label="영업이익", color="#66bb6a")
    ax.bar(x + w, net_income, w, label="순이익", color="#ffa726")

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("억원")
    ax.set_title(f"{name}({ticker}) 매출 · 영업이익 · 순이익")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 매출액 숫자 표시
    for i, v in enumerate(revenue):
        if not np.isnan(v):
            ax.text(i - w, v + abs(v) * 0.02, f"{v:,.0f}", ha="center", fontsize=7)

    fig.tight_layout()
    return fig


def _plot_margins(financials, name, ticker):
    if "Total Revenue" not in financials.index or "Operating Income" not in financials.index:
        return None

    years = [c.strftime("%Y") for c in financials.columns][::-1]
    revenue = financials.loc["Total Revenue"].values[::-1]
    op_income = financials.loc["Operating Income"].values[::-1]
    net_income = financials.loc["Net Income"].values[::-1] if "Net Income" in financials.index else [0] * len(years)

    # 마진 계산
    op_margin = np.where(revenue != 0, op_income / revenue * 100, 0)
    net_margin = np.where(revenue != 0, net_income / revenue * 100, 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years, op_margin, "o-", label="영업이익률", color="#66bb6a", linewidth=2)
    ax.plot(years, net_margin, "s-", label="순이익률", color="#ffa726", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylabel("%")
    ax.set_title(f"{name}({ticker}) 수익성 마진 추이")
    ax.legend()
    ax.grid(alpha=0.3)

    for i, (om, nm) in enumerate(zip(op_margin, net_margin)):
        ax.text(i, om + 0.5, f"{om:.1f}%", ha="center", fontsize=8, color="#388e3c")
        ax.text(i, nm - 1.5, f"{nm:.1f}%", ha="center", fontsize=8, color="#e65100")

    fig.tight_layout()
    return fig


def _plot_quarterly(q_fin, name, ticker):
    cols = q_fin.columns[:8]  # 최근 8분기
    labels = [c.strftime("%y.%m") for c in cols][::-1]

    def _vals(key):
        if key in q_fin.index:
            return (q_fin.loc[key][cols].values / HUNDRED_MILLION)[::-1]
        return [0] * len(labels)

    revenue = _vals("Total Revenue")
    op_income = _vals("Operating Income")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))

    ax1.bar(x, revenue, 0.4, label="매출액", color="#42a5f5", alpha=0.8)
    ax1.set_ylabel("매출액 (억원)", color="#42a5f5")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(x, op_income, "o-", color="#ef5350", linewidth=2, label="영업이익")
    ax2.set_ylabel("영업이익 (억원)", color="#ef5350")

    # 영업이익 숫자 표시
    for i, v in enumerate(op_income):
        if not np.isnan(v):
            ax2.text(i, v + abs(v) * 0.05 + 1, f"{v:.0f}", ha="center", fontsize=8, color="#ef5350")

    fig.suptitle(f"{name}({ticker}) 분기별 실적", fontsize=13)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    return fig


def _plot_price_volume(hist, name, ticker):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[3, 1], sharex=True)

    ax1.plot(hist.index, hist["Close"], color="#1976d2", linewidth=1.2)
    ax1.fill_between(hist.index, hist["Close"], alpha=0.1, color="#1976d2")
    ax1.set_ylabel("주가 (원)")
    ax1.set_title(f"{name}({ticker}) 5년 주가 · 거래량")
    ax1.grid(alpha=0.3)

    ax2.bar(hist.index, hist["Volume"], color="#78909c", alpha=0.6, width=2)
    ax2.set_ylabel("거래량")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig
