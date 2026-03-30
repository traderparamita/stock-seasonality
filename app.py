"""
주식 전략 테스트 웹앱
실행: streamlit run app.py
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
from datetime import datetime

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

# ── 사이드바: 입력 ────────────────────────────────────────
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

# ── 메인: 분석 결과 ───────────────────────────────────────
if run_btn:
    # 종목 유효성 확인
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

    # 전략 실행
    strategy = STRATEGIES[strategy_name]
    with st.spinner(f"'{strategy_name}' 분석 중..."):
        result = strategy.run(df, ticker, ticker_name)

    # ── 핵심 지표 ──
    st.divider()
    cols = st.columns(len(result.metrics))
    for col, (label, value) in zip(cols, result.metrics.items()):
        col.metric(label, value)

    # ── 요약 테이블 ──
    if result.summary_df is not None:
        st.divider()
        st.subheader("📋 월별 통계 요약")
        st.dataframe(result.summary_df, use_container_width=True)

    # ── 차트 ──
    st.divider()
    st.subheader("📈 차트")
    for title, fig in result.figures:
        st.caption(title)
        st.pyplot(fig)
        plt.close(fig)

    # ── 리포트 ──
    st.divider()
    st.subheader("📝 리포트")
    st.markdown(result.report_text)

else:
    st.info("👈 사이드바에서 종목코드와 전략을 선택한 뒤 **분석 실행** 버튼을 누르세요.")

    # 사용 가능한 전략 목록
    st.subheader("사용 가능한 전략")
    for name, strat in STRATEGIES.items():
        st.markdown(f"- **{name}**: {strat.description}")
