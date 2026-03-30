"""전략 인터페이스 정의"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from matplotlib.figure import Figure


@dataclass
class StrategyResult:
    summary_df: pd.DataFrame | None = None
    figures: list[tuple[str, Figure]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    report_text: str = ""


class BaseStrategy(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, df: pd.DataFrame, ticker: str, ticker_name: str) -> StrategyResult:
        ...
