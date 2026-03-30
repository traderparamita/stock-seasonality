"""전략 레지스트리 — 새 전략 추가 시 여기에 한 줄 추가"""

from .seasonality import SeasonalityStrategy

STRATEGIES = {
    "Seasonality (계절성)": SeasonalityStrategy(),
}
