from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

@dataclass
class ReportAgentConfig:
    outdir: str | Path

class ReportAgent:
    def __init__(self, cfg: ReportAgentConfig):
        self.cfg = cfg
        Path(self.cfg.outdir).mkdir(parents=True, exist_ok=True)

    def save_simple_pdf(self, title: str, cards: list[dict], filename: str = "daily_report.pdf") -> str:
        path = Path(self.cfg.outdir) / filename
        c = canvas.Canvas(str(path), pagesize=A4)
        width, height = A4
        y = height - 2*cm

        c.setFont("Helvetica-Bold", 16); c.drawString(2*cm, y, title); y -= 1*cm
        c.setFont("Helvetica", 10)

        for card in cards:
            block = [
                f"Ticker: {card.get('ticker')}   |   Conviction: {card.get('conviction')}   |   P(up): {card.get('p_up'):.2f}",
                f"Reason: {card.get('reason_text')}",
                f"Stop: {card.get('stop_loss')}   Target: {card.get('target')}   Last Close: {card.get('last_close')}",
            ]
            for line in block:
                c.drawString(2*cm, y, line[:110]); y -= 0.6*cm
            y -= 0.4*cm
            if y < 3*cm:
                c.showPage(); y = height - 2*cm; c.setFont("Helvetica", 10)

        c.showPage(); c.save()
        return str(path)
