"""수급 분석 모듈 — 외국인/기관 매매동향"""

import pandas as pd
import numpy as np
import requests
from io import StringIO
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


HEADERS = {"User-Agent": "Mozilla/5.0"}


@dataclass
class SupplyDemandResult:
    data: pd.DataFrame | None = None
    figures: list[tuple[str, Figure]] = field(default_factory=list)
    metrics: dict[str, str] = field(default_factory=dict)
    report_text: str = ""


def fetch_investor_data(ticker: str, pages: int = 20) -> pd.DataFrame:
    """네이버 금융에서 외국인/기관 매매동향 수집"""
    all_data = []
    for page in range(1, pages + 1):
        try:
            url = f"https://finance.naver.com/item/frgn.naver?code={ticker}&page={page}"
            r = requests.get(url, headers=HEADERS, timeout=5)
            tables = pd.read_html(StringIO(r.text))
            if len(tables) > 2:
                all_data.append(tables[2])
        except Exception:
            break

    if not all_data:
        return pd.DataFrame()

    raw = pd.concat(all_data, ignore_index=True)
    raw = raw.dropna(how="all")

    # 컬럼 정리 (멀티인덱스 → 단일)
    df = pd.DataFrame()
    df["날짜"] = raw.iloc[:, 0]
    df["종가"] = raw.iloc[:, 1]
    df["등락률"] = raw.iloc[:, 3]
    df["거래량"] = raw.iloc[:, 4]
    df["기관순매매"] = raw.iloc[:, 5]
    df["외국인순매매"] = raw.iloc[:, 6]
    df["외국인보유주수"] = raw.iloc[:, 7]
    df["외국인보유율"] = raw.iloc[:, 8]

    # 타입 변환
    df["날짜"] = pd.to_datetime(df["날짜"], format="%Y.%m.%d", errors="coerce")
    df = df.dropna(subset=["날짜"])

    for col in ["종가", "거래량", "기관순매매", "외국인순매매", "외국인보유주수"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["외국인보유율"] = df["외국인보유율"].astype(str).str.replace("%", "")
    df["외국인보유율"] = pd.to_numeric(df["외국인보유율"], errors="coerce")

    df = df.sort_values("날짜").reset_index(drop=True)

    # 개인 순매매 추정 (거래량 기준 — 외국인/기관 이외)
    df["개인순매매"] = -(df["기관순매매"].fillna(0) + df["외국인순매매"].fillna(0))

    # 누적 순매수
    df["외국인누적"] = df["외국인순매매"].cumsum()
    df["기관누적"] = df["기관순매매"].cumsum()
    df["개인누적"] = df["개인순매매"].cumsum()

    return df


def analyze_supply_demand(ticker: str, ticker_name: str, pages: int = 20) -> SupplyDemandResult:
    """수급 분석 실행"""
    result = SupplyDemandResult()

    df = fetch_investor_data(ticker, pages)
    if df.empty:
        result.report_text = "수급 데이터를 가져올 수 없습니다."
        return result

    result.data = df

    # ── 핵심 지표 ──
    recent_20 = df.tail(20)
    recent_60 = df.tail(60)

    frgn_20 = recent_20["외국인순매매"].sum()
    inst_20 = recent_20["기관순매매"].sum()
    frgn_60 = recent_60["외국인순매매"].sum()
    inst_60 = recent_60["기관순매매"].sum()

    latest_frgn_pct = df["외국인보유율"].iloc[-1] if not df["외국인보유율"].isna().all() else 0

    result.metrics = {
        "외국인 20일": f"{frgn_20:+,.0f}주",
        "기관 20일": f"{inst_20:+,.0f}주",
        "외국인 60일": f"{frgn_60:+,.0f}주",
        "기관 60일": f"{inst_60:+,.0f}주",
        "외국인보유율": f"{latest_frgn_pct:.2f}%",
    }

    # ── 차트 ──
    result.figures.append(
        ("주가 + 외국인/기관 순매매", _plot_price_and_net(df, ticker_name, ticker))
    )
    result.figures.append(
        ("투자자별 누적 순매수", _plot_cumulative(df, ticker_name, ticker))
    )
    result.figures.append(
        ("외국인 보유율 추이", _plot_foreign_holding(df, ticker_name, ticker))
    )

    # ── 리포트 ──
    result.report_text = _generate_report(df, ticker_name, ticker, result.metrics)

    return result


# ── 시각화 ────────────────────────────────────────────────

def _plot_price_and_net(df, name, ticker):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), height_ratios=[3, 1, 1], sharex=True)

    # 주가
    ax1.plot(df["날짜"], df["종가"], color="#1976d2", linewidth=1.2)
    ax1.fill_between(df["날짜"], df["종가"], alpha=0.08, color="#1976d2")
    ax1.set_ylabel("주가 (원)")
    ax1.set_title(f"{name}({ticker}) 주가 · 수급 동향")
    ax1.grid(alpha=0.3)

    # 외국인 순매매
    colors_f = ["#d32f2f" if v >= 0 else "#1976d2" for v in df["외국인순매매"]]
    ax2.bar(df["날짜"], df["외국인순매매"], color=colors_f, alpha=0.7, width=2)
    ax2.set_ylabel("외국인")
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.grid(alpha=0.3)

    # 기관 순매매
    colors_i = ["#d32f2f" if v >= 0 else "#1976d2" for v in df["기관순매매"]]
    ax3.bar(df["날짜"], df["기관순매매"], color=colors_i, alpha=0.7, width=2)
    ax3.set_ylabel("기관")
    ax3.axhline(0, color="gray", linewidth=0.5)
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def _plot_cumulative(df, name, ticker):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), height_ratios=[1, 1], sharex=True)

    # 주가
    ax1.plot(df["날짜"], df["종가"], color="#1976d2", linewidth=1.2)
    ax1.set_ylabel("주가 (원)")
    ax1.set_title(f"{name}({ticker}) 투자자별 누적 순매수")
    ax1.grid(alpha=0.3)

    # 누적 순매수
    ax2.plot(df["날짜"], df["외국인누적"], label="외국인", color="#d32f2f", linewidth=1.5)
    ax2.plot(df["날짜"], df["기관누적"], label="기관", color="#388e3c", linewidth=1.5)
    ax2.plot(df["날짜"], df["개인누적"], label="개인(추정)", color="#1976d2", linewidth=1, alpha=0.6)
    ax2.axhline(0, color="gray", linewidth=0.5)
    ax2.set_ylabel("누적 순매수 (주)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def _plot_foreign_holding(df, name, ticker):
    df_valid = df.dropna(subset=["외국인보유율"])
    if df_valid.empty:
        return plt.figure()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[1, 1], sharex=True)

    ax1.plot(df_valid["날짜"], df_valid["종가"], color="#1976d2", linewidth=1.2)
    ax1.set_ylabel("주가 (원)")
    ax1.set_title(f"{name}({ticker}) 외국인 보유율 추이")
    ax1.grid(alpha=0.3)

    ax2.fill_between(df_valid["날짜"], df_valid["외국인보유율"], alpha=0.3, color="#d32f2f")
    ax2.plot(df_valid["날짜"], df_valid["외국인보유율"], color="#d32f2f", linewidth=1.2)
    ax2.set_ylabel("보유율 (%)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    return fig


# ── 리포트 ────────────────────────────────────────────────

def _generate_report(df, name, ticker, metrics):
    lines = []
    lines.append(f"### 수급 분석: {name}({ticker})")
    lines.append(f"분석기간: {df['날짜'].iloc[0].strftime('%Y-%m-%d')} ~ {df['날짜'].iloc[-1].strftime('%Y-%m-%d')} ({len(df)}거래일)")
    lines.append("")

    # 최근 동향
    recent_5 = df.tail(5)
    recent_20 = df.tail(20)
    recent_60 = df.tail(60)

    frgn_5 = recent_5["외국인순매매"].sum()
    inst_5 = recent_5["기관순매매"].sum()
    frgn_20 = recent_20["외국인순매매"].sum()
    inst_20 = recent_20["기관순매매"].sum()
    frgn_60 = recent_60["외국인순매매"].sum()
    inst_60 = recent_60["기관순매매"].sum()

    lines.append("**기간별 순매수 현황 (주)**")
    lines.append("")
    lines.append("| 기간 | 외국인 | 기관 | 개인(추정) |")
    lines.append("|---|---|---|---|")
    for label, fv, iv in [
        ("5일", frgn_5, inst_5),
        ("20일", frgn_20, inst_20),
        ("60일", frgn_60, inst_60),
    ]:
        pv = -(fv + iv)
        lines.append(f"| {label} | {fv:+,.0f} | {iv:+,.0f} | {pv:+,.0f} |")
    lines.append("")

    # 외국인 보유율 변화
    if not df["외국인보유율"].isna().all():
        latest_pct = df["외국인보유율"].iloc[-1]
        earliest_pct = df["외국인보유율"].iloc[0]
        pct_change = latest_pct - earliest_pct

        lines.append("**외국인 보유율**")
        lines.append(f"- 현재: {latest_pct:.2f}%")
        lines.append(f"- 기간 변화: {pct_change:+.2f}%p ({earliest_pct:.2f}% → {latest_pct:.2f}%)")
        lines.append("")

    # 수급 해석
    lines.append("**수급 해석**")

    # 외국인 추세
    frgn_consecutive = 0
    for i in range(len(df) - 1, -1, -1):
        v = df.iloc[i]["외국인순매매"]
        if pd.isna(v):
            break
        if frgn_consecutive == 0:
            direction = "매수" if v > 0 else "매도"
            frgn_consecutive = 1
        elif (v > 0 and direction == "매수") or (v <= 0 and direction == "매도"):
            frgn_consecutive += 1
        else:
            break

    if frgn_consecutive >= 3:
        lines.append(f"- 외국인 {frgn_consecutive}일 연속 순{direction} 중")

    if frgn_20 > 0 and frgn_60 > 0:
        lines.append("- 외국인: 단기·중기 모두 순매수 — **수급 우호적**")
    elif frgn_20 > 0 and frgn_60 <= 0:
        lines.append("- 외국인: 단기 순매수 전환, 중기 누적은 음(-)  — 추세 전환 관찰 필요")
    elif frgn_20 <= 0 and frgn_60 > 0:
        lines.append("- 외국인: 단기 순매도, 중기 누적은 양(+) — 일시적 차익실현 가능성")
    else:
        lines.append("- 외국인: 단기·중기 모두 순매도 — **수급 부정적**")

    if inst_20 > 0 and inst_60 > 0:
        lines.append("- 기관: 단기·중기 모두 순매수 — **기관 관심 증가**")
    elif inst_20 == 0 and inst_60 == 0:
        lines.append("- 기관: 매매 없음 — 소형주 특성")
    elif inst_20 <= 0 and inst_60 <= 0:
        lines.append("- 기관: 단기·중기 모두 순매도 — **기관 이탈 주의**")
    else:
        lines.append(f"- 기관: 단기 {inst_20:+,.0f}주, 중기 {inst_60:+,.0f}주")

    # 종합 수급 점수
    score = 0
    if frgn_20 > 0:
        score += 1
    if frgn_60 > 0:
        score += 1
    if inst_20 > 0:
        score += 1
    if inst_60 > 0:
        score += 1

    if score >= 3:
        lines.append("- **종합 수급: 양호** — 외국인/기관 매수세 확인")
    elif score >= 2:
        lines.append("- **종합 수급: 중립** — 투자자간 방향 혼재")
    else:
        lines.append("- **종합 수급: 부정적** — 외국인/기관 매도 우위")

    return "\n".join(lines)
