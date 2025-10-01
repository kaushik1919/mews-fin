"""Predefined crisis case study scenarios for MEWS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CaseStudyScenario:
    """Configuration for a crisis case study replay."""

    name: str
    slug: str
    start_date: str
    end_date: str
    description: str
    crisis_markers: Sequence[Tuple[str, str]] = field(default_factory=tuple)
    symbols: Optional[Sequence[str]] = None
    risk_threshold: float = 0.6
    drawdown_window: int = 5
    top_feature_count: int = 5

    def marker_dates(self) -> List[Tuple[str, str]]:
        """Return crisis markers as a list of (date, label) tuples."""

        return [(date, label) for date, label in self.crisis_markers]


PREDEFINED_CASE_STUDIES: Tuple[CaseStudyScenario, ...] = (
    CaseStudyScenario(
        name="Global Financial Crisis 2008",
        slug="gfc_2008",
        start_date="2007-07-01",
        end_date="2009-12-31",
        description=(
            "Replay the Global Financial Crisis with focus on risk signals during "
            "the subprime mortgage collapse, Lehman Brothers failure, and the "
            "ensuing market turmoil."
        ),
        crisis_markers=(
            ("2007-08-09", "BNP Paribas Freezes Funds"),
            ("2008-09-15", "Lehman Brothers Bankruptcy"),
            ("2008-10-03", "TARP Signed Into Law"),
        ),
        symbols=("AAPL", "MSFT", "JPM"),
        risk_threshold=0.62,
        drawdown_window=10,
        top_feature_count=6,
    ),
    CaseStudyScenario(
        name="COVID Crash 2020",
        slug="covid_crash",
        start_date="2019-06-01",
        end_date="2021-12-31",
        description=(
            "Examine the COVID-19 market shock, highlighting early warning "
            "signals, liquidity stresses, and recovery dynamics across major "
            "technology and financial stocks."
        ),
        crisis_markers=(
            ("2020-02-24", "Global Markets Sell-off"),
            ("2020-03-09", "Oil Shock & Limit Down"),
            ("2020-03-23", "Fed QE Infinity"),
        ),
        symbols=("AAPL", "MSFT", "JPM"),
        risk_threshold=0.6,
        drawdown_window=7,
        top_feature_count=5,
    ),
    CaseStudyScenario(
        name="Fed Hiking Cycle 2022-2023",
        slug="fed_hikes",
        start_date="2021-01-01",
        end_date="2024-12-31",
        description=(
            "Analyze the Federal Reserve tightening cycle, tracing risk build-up "
            "around rate hikes, quantitative tightening, and banking stresses."
        ),
        crisis_markers=(
            ("2022-03-16", "First Post-Pandemic Rate Hike"),
            ("2022-09-21", "0.75% Hike & Dot Plot"),
            ("2023-03-12", "SVB Resolution"),
        ),
        symbols=("AAPL", "MSFT", "JPM"),
        risk_threshold=0.58,
        drawdown_window=14,
        top_feature_count=5,
    ),
)
