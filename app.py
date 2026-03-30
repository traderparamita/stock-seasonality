"""
주식 전략 테스트 웹앱
실행: streamlit run app.py
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os

from data import fetch_ohlcv, get_ticker_name
from strategies import STRATEGIES

# ── 설정 ──────────────────────────────────────────────────
if platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
else:
    plt.rcParams["font.family"] = "NanumGothic"
    fm._load_fontmanager(try_read_cache=False)
plt.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="전략 테스터", page_icon="📊", layout="wide")
st.title("📊 주식 전략 테스터")

# ── 사이드바: 모드 선택 ───────────────────────────────────
with st.sidebar:
    mode = st.radio("모드", ["📈 단일 종목 분석", "🔍 계절성 스크리너", "⚖️ 종목 비교"], index=0)
    st.divider()

# ══════════════════════════════════════════════════════════
#  모드 1: 단일 종목 분석
# ══════════════════════════════════════════════════════════
if mode == "📈 단일 종목 분석":
    from fundamentals import analyze_fundamentals

    with st.sidebar:
        st.header("설정")
        ticker = st.text_input("종목코드 (6자리)", value="037070", max_chars=6)
        strategy_name = st.selectbox("전략 선택", list(STRATEGIES.keys()))
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("시작일", value=datetime(2010, 1, 1))
        with col2:
            end_date = st.date_input("종료일", value=datetime.today())
        run_btn = st.button("🚀 분석 실행", use_container_width=True, type="primary")

    if run_btn:
        ticker_name = get_ticker_name(ticker)
        if not ticker_name:
            st.error(f"❌ 종목코드 '{ticker}'를 찾을 수 없습니다.")
            st.stop()

        st.subheader(f"{ticker_name} ({ticker})")

        # 데이터 수집
        with st.spinner("데이터 수집 중..."):
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")
            df = fetch_ohlcv(ticker, start_str, end_str)

        if df.empty:
            st.error("데이터를 가져올 수 없습니다.")
            st.stop()

        st.caption(f"수집: {len(df)}일 ({df.index[0].date()} ~ {df.index[-1].date()})")

        # 계절성 분석
        strategy = STRATEGIES[strategy_name]
        with st.spinner(f"'{strategy_name}' 분석 중..."):
            result = strategy.run(df, ticker, ticker_name)

        # 재무 분석
        try:
            import FinanceDataReader as fdr
            listing = fdr.StockListing("KOSPI")
            market = "KS" if ticker in listing["Code"].values else "KQ"
        except Exception:
            market = "KQ"

        with st.spinner("재무 분석 중..."):
            fa_result = analyze_fundamentals(ticker, ticker_name, market)

        # ── 탭 구성 ──
        tab_season, tab_finance, tab_report = st.tabs(["📈 계절성 분석", "🏢 재무 분석", "📝 종합 리포트"])

        # ── 탭 1: 계절성 분석 ──
        with tab_season:
            cols = st.columns(len(result.metrics))
            for col, (label, value) in zip(cols, result.metrics.items()):
                col.metric(label, value)

            if result.summary_df is not None:
                st.divider()
                st.subheader("월별 통계 요약")
                st.dataframe(result.summary_df, use_container_width=True)

            st.divider()
            for title, fig in result.figures:
                st.caption(title)
                st.pyplot(fig)
                plt.close(fig)

        # ── 탭 2: 재무 분석 ──
        with tab_finance:
            # 기업 개요
            info = fa_result.company_info
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**섹터**: {info.get('섹터', 'N/A')}")
                st.markdown(f"**산업**: {info.get('산업', 'N/A')}")
                st.markdown(f"**시가총액**: {info.get('시가총액', 'N/A')}")
                st.markdown(f"**52주 범위**: {info.get('52주 최저', 'N/A')} ~ {info.get('52주 최고', 'N/A')}")
                st.markdown(f"**직원수**: {info.get('직원수', 'N/A')}")
            with col2:
                st.markdown("**사업 소개**")
                st.caption(info.get("사업 소개", "정보 없음"))

            # 핵심 지표
            st.divider()
            mcols = st.columns(len(fa_result.metrics))
            for col, (label, value) in zip(mcols, fa_result.metrics.items()):
                col.metric(label, value)

            # 재무제표
            if fa_result.financials_df is not None:
                st.divider()
                st.markdown("**연간 재무제표 (억원)**")
                display_df = fa_result.financials_df.copy()
                for col in display_df.columns:
                    display_df[col] = display_df.apply(
                        lambda row: f"{row[col]:,.0f}" if pd.notna(row[col]) and row.name != "EPS"
                        else (f"{row[col]:,.0f}" if pd.notna(row[col]) else "N/A"),
                        axis=1,
                    )
                st.dataframe(display_df, use_container_width=True)

            if fa_result.quarterly_df is not None:
                with st.expander("📊 분기별 실적 (억원)"):
                    q_display = fa_result.quarterly_df.copy()
                    for col in q_display.columns:
                        q_display[col] = q_display[col].apply(
                            lambda v: f"{v:,.0f}" if pd.notna(v) else "N/A"
                        )
                    st.dataframe(q_display, use_container_width=True)

            # 차트 (2열)
            st.divider()
            fig_pairs = list(zip(fa_result.figures[::2], fa_result.figures[1::2]))
            remainders = fa_result.figures[len(fig_pairs)*2:]

            for (t1, f1), (t2, f2) in fig_pairs:
                c1, c2 = st.columns(2)
                with c1:
                    st.caption(t1)
                    st.pyplot(f1)
                    plt.close(f1)
                with c2:
                    st.caption(t2)
                    st.pyplot(f2)
                    plt.close(f2)

            for title, fig in remainders:
                st.caption(title)
                st.pyplot(fig)
                plt.close(fig)

        # ── 탭 3: 종합 리포트 ──
        with tab_report:
            st.markdown(result.report_text)
            st.divider()
            if fa_result.report_text:
                st.markdown(fa_result.report_text)

    else:
        st.info("👈 사이드바에서 종목코드와 전략을 선택한 뒤 **분석 실행** 버튼을 누르세요.")
        st.subheader("사용 가능한 전략")
        for name, strat in STRATEGIES.items():
            st.markdown(f"- **{name}**: {strat.description}")

# ══════════════════════════════════════════════════════════
#  모드 2: 계절성 스크리너
# ══════════════════════════════════════════════════════════
elif mode == "🔍 계절성 스크리너":
    from pykrx import stock as pykrx_stock
    import FinanceDataReader as fdr

    with st.sidebar:
        st.header("스크리너 설정")

        quarter = st.selectbox("타겟 분기/월", [
            "1분기 (1~3월)", "2분기 (4~6월)", "3분기 (7~9월)", "4분기 (10~12월)",
            "사용자 지정",
        ], index=1)

        if quarter == "사용자 지정":
            target_months = st.multiselect("월 선택", list(range(1, 13)), default=[4, 5, 6])
        else:
            q_map = {
                "1분기 (1~3월)": [1, 2, 3],
                "2분기 (4~6월)": [4, 5, 6],
                "3분기 (7~9월)": [7, 8, 9],
                "4분기 (10~12월)": [10, 11, 12],
            }
            target_months = q_map[quarter]

        st.caption(f"분석 대상 월: {target_months}")

        col1, col2 = st.columns(2)
        with col1:
            scr_start = st.date_input("시작일 ", value=datetime(2015, 1, 1), key="scr_start")
        with col2:
            scr_end = st.date_input("종료일 ", value=datetime.today(), key="scr_end")

        min_cap = st.number_input("최소 시가총액 (억원)", value=500, step=100)
        min_avg = st.number_input("최소 평균수익률 (%)", value=3.0, step=1.0)
        min_wr = st.number_input("최소 승률 (%)", value=60.0, step=5.0)
        min_years = st.number_input("최소 관측 연수", value=5, step=1)

        # 저장된 결과 불러오기 옵션
        csv_path = "output/screener_q2.csv"
        has_saved = os.path.exists(csv_path)

        scr_run = st.button("🔍 스크리닝 실행", use_container_width=True, type="primary")
        if has_saved:
            load_saved = st.button("📂 저장된 결과 불러오기", use_container_width=True)
        else:
            load_saved = False

    def run_screener(target_months, start_str, end_str, min_cap, min_avg, min_wr, min_years):
        """스크리너 실행"""
        month_label = ", ".join([f"{m}월" for m in target_months])

        # 종목 리스트
        frames = []
        for market in ["KOSPI", "KOSDAQ"]:
            df = fdr.StockListing(market)
            df = df[["Code", "Name", "Market", "Marcap"]].copy()
            df.columns = ["ticker", "name", "market", "marcap"]
            df["market_label"] = market
            df["market_cap"] = df["marcap"] / 1e8
            frames.append(df)

        all_tickers = pd.concat(frames, ignore_index=True)
        all_tickers = all_tickers[all_tickers["market_cap"] >= min_cap].reset_index(drop=True)

        total = len(all_tickers)
        progress = st.progress(0, text=f"0/{total} 종목 분석 중...")
        results = []

        for idx, row in all_tickers.iterrows():
            if idx % 20 == 0:
                progress.progress(idx / total, text=f"{idx}/{total} 종목 분석 중...")

            try:
                ohlcv = pykrx_stock.get_market_ohlcv(start_str, end_str, row["ticker"])
                if len(ohlcv) < 250:
                    continue

                ohlcv.index = pd.to_datetime(ohlcv.index)
                monthly_close = ohlcv["종가"].resample("ME").last()
                monthly_ret = monthly_close.pct_change().dropna()

                mdf = pd.DataFrame({
                    "year": monthly_ret.index.year,
                    "month": monthly_ret.index.month,
                    "return": monthly_ret.values,
                })

                target = mdf[mdf["month"].isin(target_months)]
                quarterly = target.groupby("year")["return"].apply(
                    lambda x: (1 + x).prod() - 1
                )

                if len(quarterly) < min_years:
                    continue

                avg_ret = quarterly.mean() * 100
                win_rate = (quarterly > 0).mean() * 100

                if avg_ret >= min_avg and win_rate >= min_wr:
                    monthly_avg = target.groupby("month")["return"].mean() * 100
                    entry = {
                        "종목코드": row["ticker"],
                        "종목명": row["name"],
                        "시장": row["market_label"],
                        "시가총액(억)": round(row["market_cap"]),
                        "평균수익률(%)": round(avg_ret, 2),
                        "중앙값(%)": round(quarterly.median() * 100, 2),
                        "승률(%)": round(win_rate, 1),
                        "표준편차(%)": round(quarterly.std() * 100, 2),
                        "관측연수": len(quarterly),
                    }
                    for m in target_months:
                        entry[f"{m}월(%)"] = round(monthly_avg.get(m, 0), 2)
                    results.append(entry)
            except Exception:
                continue

            if idx % 10 == 0:
                time.sleep(0.2)

        progress.progress(1.0, text="완료!")
        if not results:
            return pd.DataFrame()

        result_df = pd.DataFrame(results).sort_values("평균수익률(%)", ascending=False).reset_index(drop=True)
        return result_df

    def display_screener_results(result_df, target_months):
        """스크리너 결과 표시 + 종목 클릭 시 상세 분석"""
        month_label = ", ".join([f"{m}월" for m in target_months])
        st.subheader(f"🔍 스크리닝 결과: {month_label} 강세 종목 ({len(result_df)}개)")

        # 결과 테이블
        st.dataframe(
            result_df.style.format({
                "시가총액(억)": "{:,.0f}",
                "평균수익률(%)": "{:+.2f}",
                "중앙값(%)": "{:+.2f}",
                "승률(%)": "{:.1f}",
                "표준편차(%)": "{:.2f}",
            }).background_gradient(subset=["평균수익률(%)"], cmap="RdYlGn")
            .background_gradient(subset=["승률(%)"], cmap="RdYlGn"),
            use_container_width=True,
            height=400,
        )

        # CSV 다운로드
        csv = result_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 CSV 다운로드", csv, "screener_result.csv", "text/csv")

        st.divider()

        # 상세 분석할 종목 선택
        st.subheader("📈 종목 상세 분석")
        top_options = [
            f"{r['종목명']} ({r['종목코드']}) - 수익률:{r['평균수익률(%)']:+.1f}%, 승률:{r['승률(%)']:.0f}%"
            for _, r in result_df.head(30).iterrows()
        ]

        selected = st.multiselect(
            "분석할 종목 선택 (복수 선택 가능)",
            top_options,
            default=top_options[:3] if len(top_options) >= 3 else top_options,
        )

        if selected and st.button("📊 선택 종목 분석", type="primary"):
            strategy = STRATEGIES["Seasonality (계절성)"]

            for sel in selected:
                # 종목코드 파싱
                code = sel.split("(")[1].split(")")[0]
                name = get_ticker_name(code)

                st.divider()
                st.subheader(f"{name} ({code})")

                with st.spinner(f"{name} 분석 중..."):
                    df = fetch_ohlcv(code, "20100101", datetime.today().strftime("%Y%m%d"))
                    if df.empty:
                        st.warning(f"{name}: 데이터 없음")
                        continue
                    result = strategy.run(df, code, name)

                # 지표
                cols = st.columns(len(result.metrics))
                for col, (label, value) in zip(cols, result.metrics.items()):
                    col.metric(label, value)

                # 요약 테이블
                if result.summary_df is not None:
                    st.dataframe(result.summary_df, use_container_width=True)

                # 차트 (2열)
                fig_pairs = list(zip(result.figures[::2], result.figures[1::2]))
                remainders = result.figures[len(fig_pairs)*2:]

                for (t1, f1), (t2, f2) in fig_pairs:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.caption(t1)
                        st.pyplot(f1)
                        plt.close(f1)
                    with c2:
                        st.caption(t2)
                        st.pyplot(f2)
                        plt.close(f2)

                for title, fig in remainders:
                    st.caption(title)
                    st.pyplot(fig)
                    plt.close(fig)

                # 리포트 (접기)
                with st.expander("📝 상세 리포트"):
                    st.markdown(result.report_text)

    # ── 실행 ──
    if scr_run:
        start_str = scr_start.strftime("%Y%m%d")
        end_str = scr_end.strftime("%Y%m%d")
        result_df = run_screener(target_months, start_str, end_str, min_cap, min_avg, min_wr, min_years)

        if result_df.empty:
            st.warning("조건을 만족하는 종목이 없습니다.")
        else:
            # 저장
            os.makedirs("output", exist_ok=True)
            result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            display_screener_results(result_df, target_months)

    elif load_saved:
        result_df = pd.read_csv(csv_path)
        st.success(f"저장된 결과 로드 ({len(result_df)}개 종목)")
        display_screener_results(result_df, target_months)

    else:
        st.info("👈 사이드바에서 조건을 설정한 뒤 **스크리닝 실행** 버튼을 누르세요.")
        st.markdown("""
        **사용법**
        1. 타겟 분기/월 선택 (예: 2분기)
        2. 시가총액, 수익률, 승률 기준 설정
        3. 스크리닝 실행 → 결과에서 종목 선택 → 상세 분석
        """)

# ══════════════════════════════════════════════════════════
#  모드 3: 종목 비교
# ══════════════════════════════════════════════════════════
elif mode == "⚖️ 종목 비교":
    import seaborn as sns
    from pykrx import stock as pykrx_stock
    from strategies.seasonality import MONTH_LABELS

    with st.sidebar:
        st.header("비교 설정")

        # 저장된 스크리너 결과에서 불러오기
        csv_path = "output/screener_q2.csv"
        preset_tickers = []
        if os.path.exists(csv_path):
            saved_df = pd.read_csv(csv_path, dtype={"종목코드": str})
            preset_tickers = [
                f"{r['종목명']} ({r['종목코드']})"
                for _, r in saved_df.head(30).iterrows()
            ]

        ticker_input_mode = st.radio("종목 입력 방식", ["스크리너 Top 종목", "직접 입력"])

        if ticker_input_mode == "스크리너 Top 종목" and preset_tickers:
            selected_tickers = st.multiselect(
                "비교 종목 선택",
                preset_tickers,
                default=preset_tickers[:10],
            )
            codes = [s.split("(")[1].split(")")[0] for s in selected_tickers]
        else:
            raw = st.text_area(
                "종목코드 (쉼표 구분)",
                value="298040,241710,006340,251970,009470,950140,036620,007540,196170,033100",
            )
            codes = [c.strip() for c in raw.split(",") if c.strip()]

        col1, col2 = st.columns(2)
        with col1:
            cmp_start = st.date_input("시작일  ", value=datetime(2015, 1, 1), key="cmp_s")
        with col2:
            cmp_end = st.date_input("종료일  ", value=datetime.today(), key="cmp_e")

        cmp_run = st.button("⚖️ 비교 분석", use_container_width=True, type="primary")

    if cmp_run and codes:
        start_str = cmp_start.strftime("%Y%m%d")
        end_str = cmp_end.strftime("%Y%m%d")

        # ── 데이터 수집 ──
        all_monthly = []
        stock_info = {}
        progress = st.progress(0, text="데이터 수집 중...")

        for i, code in enumerate(codes):
            progress.progress((i + 1) / len(codes), text=f"{i+1}/{len(codes)} 수집 중...")
            name = get_ticker_name(code)
            if not name:
                continue
            stock_info[code] = name

            df = fetch_ohlcv(code, start_str, end_str)
            if df.empty:
                continue

            monthly_close = df["종가"].resample("ME").last()
            monthly_ret = monthly_close.pct_change().dropna()
            for dt, ret in monthly_ret.items():
                all_monthly.append({
                    "code": code, "name": name, "label": f"{name}\n({code})",
                    "year": dt.year, "month": dt.month, "return": ret,
                })

        progress.empty()

        if not all_monthly:
            st.error("데이터를 수집할 수 없습니다.")
            st.stop()

        mdf = pd.DataFrame(all_monthly)
        labels = list(dict.fromkeys(mdf["label"]))  # 순서 유지

        st.subheader(f"⚖️ {len(stock_info)}개 종목 비교 분석")

        # ── 1. 월별 평균수익률 비교 히트맵 ──
        st.divider()
        st.subheader("1. 월별 평균수익률 비교 (%)")

        pivot = mdf.pivot_table(index="label", columns="month", values="return", aggfunc="mean")
        pivot = pivot.reindex(labels)
        pivot.columns = MONTH_LABELS

        fig, ax = plt.subplots(figsize=(14, max(4, len(labels) * 0.6)))
        sns.heatmap(
            pivot * 100, annot=True, fmt=".1f", cmap="RdYlGn", center=0,
            linewidths=0.5, ax=ax, cbar_kws={"label": "평균수익률 (%)"},
        )
        ax.set_title("종목별 × 월별 평균수익률 (%)")
        ax.set_ylabel("")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # ── 2. 월별 승률 비교 히트맵 ──
        st.divider()
        st.subheader("2. 월별 승률 비교 (%)")

        wr_pivot = mdf.pivot_table(
            index="label", columns="month", values="return",
            aggfunc=lambda x: (x > 0).mean(),
        )
        wr_pivot = wr_pivot.reindex(labels)
        wr_pivot.columns = MONTH_LABELS

        fig2, ax2 = plt.subplots(figsize=(14, max(4, len(labels) * 0.6)))
        sns.heatmap(
            wr_pivot * 100, annot=True, fmt=".0f", cmap="RdYlGn", center=50,
            linewidths=0.5, ax=ax2, vmin=0, vmax=100,
            cbar_kws={"label": "승률 (%)"},
        )
        ax2.set_title("종목별 × 월별 승률 (%)")
        ax2.set_ylabel("")
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # ── 3. 분기별 수익률 비교 바 차트 ──
        st.divider()
        st.subheader("3. 분기별 평균수익률 비교")

        quarter_map = {1: "Q1", 2: "Q1", 3: "Q1", 4: "Q2", 5: "Q2", 6: "Q2",
                       7: "Q3", 8: "Q3", 9: "Q3", 10: "Q4", 11: "Q4", 12: "Q4"}
        mdf["quarter"] = mdf["month"].map(quarter_map)

        qtr_ret = mdf.groupby(["label", "year", "quarter"])["return"].apply(
            lambda x: (1 + x).prod() - 1
        ).reset_index(name="qtr_return")
        qtr_avg = qtr_ret.groupby(["label", "quarter"])["qtr_return"].mean().reset_index()
        qtr_avg["qtr_return_pct"] = qtr_avg["qtr_return"] * 100

        fig3, ax3 = plt.subplots(figsize=(14, 6))
        quarters = ["Q1", "Q2", "Q3", "Q4"]
        x = np.arange(len(labels))
        width = 0.2
        colors = ["#42a5f5", "#66bb6a", "#ffa726", "#ef5350"]

        for i, q in enumerate(quarters):
            vals = []
            for lbl in labels:
                row = qtr_avg[(qtr_avg["label"] == lbl) & (qtr_avg["quarter"] == q)]
                vals.append(row["qtr_return_pct"].values[0] if len(row) > 0 else 0)
            bars = ax3.bar(x + i * width, vals, width, label=q, color=colors[i])
            for j, v in enumerate(vals):
                if abs(v) > 1:
                    ax3.text(x[j] + i * width, v + (0.5 if v >= 0 else -1.5),
                             f"{v:.0f}", ha="center", fontsize=7)

        ax3.set_xticks(x + width * 1.5)
        ax3.set_xticklabels([l.replace("\n", " ") for l in labels], rotation=45, ha="right", fontsize=8)
        ax3.set_ylabel("평균 수익률 (%)")
        ax3.set_title("종목별 분기 평균수익률 비교")
        ax3.legend()
        ax3.axhline(0, color="black", linewidth=0.5)
        ax3.grid(axis="y", alpha=0.3)
        fig3.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        # ── 4. 요약 테이블 ──
        st.divider()
        st.subheader("4. 종합 요약")

        summary_rows = []
        for code in codes:
            name = stock_info.get(code)
            if not name:
                continue
            sub = mdf[mdf["code"] == code]

            # 분기별 수익률
            q2 = sub[sub["month"].isin([4, 5, 6])].groupby("year")["return"].apply(
                lambda x: (1 + x).prod() - 1
            )
            annual = sub.groupby("year")["return"].apply(lambda x: (1 + x).prod() - 1)

            # 최고/최저 월
            month_avg = sub.groupby("month")["return"].mean()
            best_m = month_avg.idxmax()
            worst_m = month_avg.idxmin()

            summary_rows.append({
                "종목명": name,
                "종목코드": code,
                "연평균수익률(%)": round(annual.mean() * 100, 1),
                "Q2 평균(%)": round(q2.mean() * 100, 1) if len(q2) > 0 else None,
                "Q2 승률(%)": round((q2 > 0).mean() * 100, 0) if len(q2) > 0 else None,
                "최강월": f"{MONTH_LABELS[best_m - 1]} ({month_avg[best_m]*100:+.1f}%)",
                "최약월": f"{MONTH_LABELS[worst_m - 1]} ({month_avg[worst_m]*100:+.1f}%)",
                "관측연수": sub["year"].nunique(),
            })

        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(
            summary_df.style
            .background_gradient(subset=["Q2 평균(%)"], cmap="RdYlGn")
            .background_gradient(subset=["Q2 승률(%)"], cmap="RdYlGn"),
            use_container_width=True,
        )

        # CSV
        csv = summary_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 비교 결과 CSV", csv, "comparison.csv", "text/csv")

    elif cmp_run:
        st.warning("종목코드를 입력하세요.")
    else:
        st.info("👈 사이드바에서 비교할 종목을 선택한 뒤 **비교 분석** 버튼을 누르세요.")
