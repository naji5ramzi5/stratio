"""quant_x — institutional-grade AI crypto research engine (QUANT-X).

A data-driven implementation of the 10-step QUANT-X analysis framework.
Every step is computed from real data where possible; anything that requires a
paid or unavailable data source is reported as MISSING rather than invented.
"""
from quant_x.data import collect_market_data
from quant_x.report import build_report, render_markdown

__all__ = ["collect_market_data", "build_report", "render_markdown"]
__version__ = "1.0.0"
