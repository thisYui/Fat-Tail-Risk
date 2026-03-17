"""
visualization/report.py
------------------------
Tạo báo cáo tự động (PDF / HTML) tổng hợp kết quả phân tích:

  - RiskReport      : class chính build báo cáo
  - Section builder  : add_section, add_figure, add_table, add_metrics
  - Exporters        : to_pdf(), to_html(), to_markdown()
  - Convenience fn   : generate_full_report()
"""

from __future__ import annotations

import io
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# Section & content block data classes
# ---------------------------------------------------------------------------

@dataclass
class ReportSection:
    """Một section trong báo cáo."""
    title: str
    content: List[dict] = field(default_factory=list)  # list of blocks

    def add_text(self, text: str) -> "ReportSection":
        self.content.append({"type": "text", "data": text})
        return self

    def add_figure(self, fig: Figure, caption: str = "") -> "ReportSection":
        self.content.append({"type": "figure", "data": fig, "caption": caption})
        return self

    def add_table(
        self,
        df: pd.DataFrame,
        caption: str = "",
        fmt: Optional[Dict[str, str]] = None,
    ) -> "ReportSection":
        self.content.append({
            "type": "table", "data": df,
            "caption": caption, "fmt": fmt or {},
        })
        return self

    def add_metrics(
        self,
        metrics: pd.Series,
        title: str = "",
    ) -> "ReportSection":
        self.content.append({"type": "metrics", "data": metrics, "title": title})
        return self


# ---------------------------------------------------------------------------
# RiskReport builder
# ---------------------------------------------------------------------------

class RiskReport:
    """
    Builder tạo báo cáo rủi ro tổng hợp.

    Example
    -------
    >>> report = RiskReport(title="Portfolio Risk Report", author="Quant Team")
    >>> sec = report.add_section("Return Analysis")
    >>> sec.add_figure(fig_cumulative, "Cumulative return of portfolio")
    >>> sec.add_table(metrics_df, "Key risk metrics")
    >>> report.to_html("report.html")
    >>> report.to_pdf("report.pdf")
    """

    def __init__(
        self,
        title: str = "Fat-Tail Risk Report",
        author: str = "Fat-Tail-Risk System",
        date: Optional[str] = None,
        description: str = "",
    ):
        self.title = title
        self.author = author
        self.date = date or datetime.now().strftime("%Y-%m-%d %H:%M")
        self.description = description
        self.sections: List[ReportSection] = []

    def add_section(self, title: str) -> ReportSection:
        """Thêm section mới và trả về section đó để chain."""
        sec = ReportSection(title=title)
        self.sections.append(sec)
        return sec

    # ------------------------------------------------------------------
    # HTML export
    # ------------------------------------------------------------------

    def to_html(self, path: Optional[Union[str, Path]] = None) -> str:
        """
        Xuất báo cáo ra HTML.

        Parameters
        ----------
        path : đường dẫn file output (None → trả về HTML string)

        Returns
        -------
        str – nội dung HTML
        """
        html = self._build_html()
        if path:
            Path(path).write_text(html, encoding="utf-8")
        return html

    def _build_html(self) -> str:
        css = textwrap.dedent("""
        <style>
          body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                 margin: 40px auto; max-width: 1100px; color: #2D3748; line-height: 1.6; }
          h1   { color: #2B6CB0; border-bottom: 3px solid #2B6CB0; padding-bottom: 10px; }
          h2   { color: #2B6CB0; margin-top: 40px; border-left: 5px solid #2B6CB0;
                 padding-left: 12px; }
          h3   { color: #4A5568; }
          .meta { color: #718096; font-size: 0.9em; margin-bottom: 30px; }
          .description { background: #EBF8FF; border-left: 4px solid #3182CE;
                          padding: 12px 16px; border-radius: 4px; margin-bottom: 20px; }
          table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 0.88em; }
          th    { background: #2B6CB0; color: white; padding: 8px 12px; text-align: left; }
          td    { padding: 7px 12px; border-bottom: 1px solid #E2E8F0; }
          tr:nth-child(even) td { background: #F7FAFC; }
          .caption { font-size: 0.85em; color: #718096; margin-top: -10px;
                     margin-bottom: 20px; font-style: italic; }
          .metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
                           gap: 12px; margin: 16px 0; }
          .metric-card  { background: #F7FAFC; border: 1px solid #E2E8F0; border-radius: 8px;
                           padding: 14px 16px; }
          .metric-label { font-size: 0.82em; color: #718096; text-transform: uppercase;
                           letter-spacing: 0.05em; }
          .metric-value { font-size: 1.3em; font-weight: 700; color: #2B6CB0; margin-top: 4px; }
          img  { max-width: 100%; height: auto; border-radius: 6px;
                 box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
          p    { margin: 10px 0; }
        </style>
        """)

        body_parts = [
            f"<h1>{self.title}</h1>",
            f'<p class="meta">Author: {self.author} &nbsp;|&nbsp; Date: {self.date}</p>',
        ]
        if self.description:
            body_parts.append(f'<div class="description">{self.description}</div>')

        for sec in self.sections:
            body_parts.append(f"<h2>{sec.title}</h2>")
            for block in sec.content:
                body_parts.append(self._render_block_html(block))

        html = f"<!DOCTYPE html><html><head><meta charset='utf-8'>{css}</head><body>{''.join(body_parts)}</body></html>"
        return html

    def _render_block_html(self, block: dict) -> str:
        btype = block["type"]

        if btype == "text":
            return f"<p>{block['data']}</p>"

        elif btype == "figure":
            fig = block["data"]
            caption = block.get("caption", "")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
            import base64
            b64 = base64.b64encode(buf.getvalue()).decode()
            img_tag = f'<img src="data:image/png;base64,{b64}" alt="{caption}">'
            cap_tag = f'<p class="caption">{caption}</p>' if caption else ""
            return img_tag + cap_tag

        elif btype == "table":
            df = block["data"]
            caption = block.get("caption", "")
            fmt = block.get("fmt", {})
            formatters = {col: (lambda x, f=f: f.format(x) if pd.notna(x) else "–")
                          for col, f in fmt.items()}
            html_table = df.to_html(
                classes="",
                border=0,
                formatters=formatters if formatters else None,
                na_rep="–",
            )
            cap_tag = f"<h3>{caption}</h3>" if caption else ""
            return cap_tag + html_table

        elif btype == "metrics":
            metrics = block["data"]
            title = block.get("title", "")
            cards = ""
            for key, val in metrics.items():
                if isinstance(val, float):
                    formatted = f"{val:.4f}" if abs(val) < 100 else f"{val:.2f}"
                else:
                    formatted = str(val)
                label = str(key).replace("_", " ").title()
                cards += f"""
                <div class="metric-card">
                  <div class="metric-label">{label}</div>
                  <div class="metric-value">{formatted}</div>
                </div>"""
            header = f"<h3>{title}</h3>" if title else ""
            return f'{header}<div class="metrics-grid">{cards}</div>'

        return ""

    # ------------------------------------------------------------------
    # Markdown export
    # ------------------------------------------------------------------

    def to_markdown(self, path: Optional[Union[str, Path]] = None) -> str:
        """
        Xuất báo cáo ra Markdown.

        Parameters
        ----------
        path : đường dẫn file output (None → trả về Markdown string)
        """
        lines = [
            f"# {self.title}",
            f"**Author:** {self.author}  |  **Date:** {self.date}",
            "",
        ]
        if self.description:
            lines += [f"> {self.description}", ""]

        for sec in self.sections:
            lines.append(f"## {sec.title}")
            lines.append("")
            for block in sec.content:
                lines += self._render_block_md(block)
                lines.append("")

        md = "\n".join(lines)
        if path:
            Path(path).write_text(md, encoding="utf-8")
        return md

    def _render_block_md(self, block: dict) -> List[str]:
        btype = block["type"]
        if btype == "text":
            return [block["data"]]
        elif btype == "figure":
            caption = block.get("caption", "Figure")
            return [f"*[Figure: {caption}]*"]
        elif btype == "table":
            df = block["data"]
            caption = block.get("caption", "")
            lines = []
            if caption:
                lines.append(f"**{caption}**")
            try:
                lines.append(df.to_markdown(floatfmt=".4f"))
            except Exception:
                lines.append(df.to_string())
            return lines
        elif btype == "metrics":
            metrics = block["data"]
            title = block.get("title", "")
            lines = []
            if title:
                lines.append(f"**{title}**")
            for k, v in metrics.items():
                val = f"{v:.4f}" if isinstance(v, float) else str(v)
                lines.append(f"- **{k}**: {val}")
            return lines
        return []

    # ------------------------------------------------------------------
    # PDF export (requires matplotlib)
    # ------------------------------------------------------------------

    def to_pdf(self, path: Union[str, Path]) -> None:
        """
        Xuất báo cáo ra PDF bằng matplotlib PdfPages.

        Parameters
        ----------
        path : đường dẫn file .pdf
        """
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(str(path)) as pdf:
            # Cover page
            fig_cover = plt.figure(figsize=(11, 8.5))
            fig_cover.patch.set_facecolor("#2B6CB0")
            ax = fig_cover.add_axes([0, 0, 1, 1])
            ax.axis("off")
            ax.text(0.5, 0.6, self.title, ha="center", va="center",
                    fontsize=28, fontweight="bold", color="white",
                    transform=ax.transAxes, wrap=True)
            ax.text(0.5, 0.45, f"Author: {self.author}", ha="center",
                    fontsize=14, color="#BEE3F8", transform=ax.transAxes)
            ax.text(0.5, 0.38, f"Date: {self.date}", ha="center",
                    fontsize=14, color="#BEE3F8", transform=ax.transAxes)
            if self.description:
                ax.text(0.5, 0.28, self.description, ha="center",
                        fontsize=11, color="#EBF8FF", transform=ax.transAxes,
                        wrap=True, style="italic")
            pdf.savefig(fig_cover, bbox_inches="tight")
            plt.close(fig_cover)

            # Sections
            for sec in self.sections:
                # Section header page (small banner)
                fig_sec = plt.figure(figsize=(11, 1.5))
                fig_sec.patch.set_facecolor("#EBF8FF")
                ax_sec = fig_sec.add_axes([0.05, 0.1, 0.9, 0.8])
                ax_sec.axis("off")
                ax_sec.text(0.0, 0.5, sec.title, va="center",
                            fontsize=18, fontweight="bold", color="#2B6CB0")
                pdf.savefig(fig_sec, bbox_inches="tight")
                plt.close(fig_sec)

                for block in sec.content:
                    if block["type"] == "figure":
                        fig = block["data"]
                        pdf.savefig(fig, bbox_inches="tight")

                    elif block["type"] == "table":
                        df = block["data"]
                        caption = block.get("caption", "")
                        fig_tbl = self._df_to_figure(df, caption)
                        pdf.savefig(fig_tbl, bbox_inches="tight")
                        plt.close(fig_tbl)

                    elif block["type"] == "metrics":
                        metrics = block["data"]
                        title = block.get("title", "Metrics")
                        fig_m = self._metrics_to_figure(metrics, title)
                        pdf.savefig(fig_m, bbox_inches="tight")
                        plt.close(fig_m)

                    elif block["type"] == "text":
                        fig_txt = plt.figure(figsize=(11, 2))
                        ax_txt = fig_txt.add_axes([0.05, 0.1, 0.9, 0.8])
                        ax_txt.axis("off")
                        ax_txt.text(0, 0.5, block["data"], va="center", fontsize=11,
                                    transform=ax_txt.transAxes, wrap=True)
                        pdf.savefig(fig_txt, bbox_inches="tight")
                        plt.close(fig_txt)

    def _df_to_figure(self, df: pd.DataFrame, caption: str = "") -> Figure:
        """Chuyển DataFrame thành matplotlib Figure (table)."""
        n_rows, n_cols = df.shape
        fig_h = max(3, min(n_rows * 0.45 + 1.5, 10))
        fig = plt.figure(figsize=(11, fig_h))
        ax = fig.add_axes([0.02, 0.02, 0.96, 0.94])
        ax.axis("off")

        if caption:
            ax.set_title(caption, fontsize=13, fontweight="bold",
                         color="#2B6CB0", pad=10)

        col_labels = [str(c) for c in df.columns]
        row_labels = [str(i) for i in df.index]
        cell_text = []
        for _, row in df.iterrows():
            cell_text.append([
                f"{v:.4f}" if isinstance(v, float) else str(v)
                for v in row
            ])

        tbl = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.5)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_facecolor("#2B6CB0")
                cell.set_text_props(color="white", fontweight="bold")
            elif col == -1:
                cell.set_facecolor("#EBF8FF")
                cell.set_text_props(fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#F7FAFC")
        return fig

    def _metrics_to_figure(self, metrics: pd.Series, title: str = "") -> Figure:
        """Chuyển metrics Series thành grid của cards."""
        items = list(metrics.items())
        n = len(items)
        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(11, max(2.5, nrows * 2)))
        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold", color="#2B6CB0")
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        for i, (k, v) in enumerate(items):
            ax = axes_flat[i]
            ax.axis("off")
            ax.set_facecolor("#F7FAFC")
            label = str(k).replace("_", " ").title()
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            ax.text(0.5, 0.65, val_str, ha="center", va="center",
                    fontsize=16, fontweight="bold", color="#2B6CB0",
                    transform=ax.transAxes)
            ax.text(0.5, 0.25, label, ha="center", va="center",
                    fontsize=8, color="#718096", transform=ax.transAxes)
            rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96,
                                  fill=True, facecolor="#EBF8FF",
                                  edgecolor="#BEE3F8", linewidth=1.5,
                                  transform=ax.transAxes)
            ax.add_patch(rect)

        for i in range(n, len(axes_flat)):
            axes_flat[i].axis("off")

        return fig


# ---------------------------------------------------------------------------
# Convenience: generate full risk report
# ---------------------------------------------------------------------------

def generate_full_report(
    returns: pd.Series,
    title: str = "Fat-Tail Risk Report",
    author: str = "Fat-Tail-Risk System",
    risk_free: float = 0.0,
    freq: int = 252,
    output_html: Optional[str] = None,
    output_pdf: Optional[str] = None,
) -> RiskReport:
    """
    Tạo báo cáo đầy đủ từ một chuỗi return.

    Tự động generate các section:
      1. Summary Metrics
      2. Return Analysis
      3. Risk Analysis (VaR/CVaR)
      4. Tail Risk (EVT)
      5. Drawdown Analysis

    Parameters
    ----------
    returns    : chuỗi return (DatetimeIndex)
    output_html: đường dẫn HTML output (None = không lưu)
    output_pdf : đường dẫn PDF output (None = không lưu)

    Returns
    -------
    RiskReport object
    """
    from ..risk.metrics import compute_risk_metrics
    from .plots import (
        plot_cumulative, plot_drawdown,
        plot_rolling_vol, plot_return_dist, plot_qq,
    )
    from .tail_plots import plot_var_cvar, plot_hill, plot_mean_excess

    r = returns.dropna()
    report = RiskReport(
        title=title,
        author=author,
        description=f"Comprehensive fat-tail risk analysis | "
                    f"Period: {r.index[0].date()} to {r.index[-1].date()} | "
                    f"N = {len(r)} observations",
    )

    # ── Section 1: Summary ────────────────────────────────────────────
    sec1 = report.add_section("1. Executive Summary")
    try:
        rm = compute_risk_metrics(r.values, risk_free=risk_free, freq=freq)
        sec1.add_metrics(rm.as_series(), title="Risk Metrics Summary")
    except Exception as e:
        sec1.add_text(f"Could not compute risk metrics: {e}")

    # ── Section 2: Return Analysis ────────────────────────────────────
    sec2 = report.add_section("2. Return & Performance Analysis")
    try:
        fig_cum = plot_cumulative(r, title="Cumulative Return")
        sec2.add_figure(fig_cum, "Growth of $1 invested at the start of the period.")
        plt.close(fig_cum)

        fig_dist = plot_return_dist(r.values, title="Return Distribution")
        sec2.add_figure(fig_dist, "Empirical return distribution with Normal and Student-t fits.")
        plt.close(fig_dist)

        fig_qq = plot_qq(r.values, dist="norm", title="Q-Q Plot vs Normal")
        sec2.add_figure(fig_qq, "Deviations from normality indicate fat tails.")
        plt.close(fig_qq)
    except Exception as e:
        sec2.add_text(f"Error generating return charts: {e}")

    # ── Section 3: VaR / CVaR ─────────────────────────────────────────
    sec3 = report.add_section("3. VaR & CVaR Analysis")
    try:
        fig_vc = plot_var_cvar(r.values, confidence=0.95)
        sec3.add_figure(fig_vc, "Value-at-Risk (95%) and CVaR shaded on return distribution.")
        plt.close(fig_vc)

        fig_rv = plot_rolling_vol(r, windows=[21, 63, 252], title="Rolling Volatility")
        sec3.add_figure(fig_rv, "Rolling annualised volatility across multiple horizons.")
        plt.close(fig_rv)
    except Exception as e:
        sec3.add_text(f"Error generating VaR charts: {e}")

    # ── Section 4: Tail Risk ──────────────────────────────────────────
    sec4 = report.add_section("4. Extreme Value Theory – Tail Analysis")
    try:
        fig_hill = plot_hill(r.values, title="Hill Plot – Tail Index")
        sec4.add_figure(fig_hill, "Hill estimator: lower α → heavier tail.")
        plt.close(fig_hill)

        fig_mef = plot_mean_excess(r.values, title="Mean Excess Function")
        sec4.add_figure(fig_mef, "Linear MEF above threshold u suggests GPD fit is appropriate.")
        plt.close(fig_mef)
    except Exception as e:
        sec4.add_text(f"Error generating tail risk charts: {e}")

    # ── Section 5: Drawdown ───────────────────────────────────────────
    sec5 = report.add_section("5. Drawdown Analysis")
    try:
        fig_dd = plot_drawdown(r, title="Drawdown from High-Water Mark")
        sec5.add_figure(fig_dd, "Drawdown series showing peak-to-trough declines.")
        plt.close(fig_dd)
    except Exception as e:
        sec5.add_text(f"Error generating drawdown chart: {e}")

    # Export
    if output_html:
        report.to_html(output_html)
    if output_pdf:
        report.to_pdf(output_pdf)

    return report