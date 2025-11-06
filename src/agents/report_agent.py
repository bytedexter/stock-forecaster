from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
import os

@dataclass
class ReportAgentConfig:
    outdir: str | Path
    gemini_api_key: str | None = None
    enhance_with_llm: bool = True

class ReportAgent:
    def __init__(self, cfg: ReportAgentConfig):
        self.cfg = cfg
        Path(self.cfg.outdir).mkdir(parents=True, exist_ok=True)
        # LLM enhancement disabled - now done in train.py
    
    def enhance_explanation(self, card: dict) -> str:
        """Return the explanation as-is (enhancement done in train.py)"""
        return card.get('reason_text', 'No explanation available')

    def save_simple_pdf(self, title: str, cards: list[dict], filename: str = "daily_report.pdf") -> str:
        path = Path(self.cfg.outdir) / filename
        c = canvas.Canvas(str(path), pagesize=A4)
        width, height = A4
        y = height - 2*cm

        # Title
        c.setFont("Helvetica-Bold", 18)
        c.drawString(2*cm, y, title)
        y -= 0.8*cm
        
        # Date
        from datetime import datetime
        c.setFont("Helvetica", 10)
        c.drawString(2*cm, y, f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
        y -= 1.2*cm

        if not cards:
            c.setFont("Helvetica", 12)
            c.drawString(2*cm, y, "No stock picks available at this time.")
            c.showPage()
            c.save()
            return str(path)

        # Process each stock pick
        for idx, card in enumerate(cards, 1):
            # Enhance explanation with LLM if available
            enhanced_reason = self.enhance_explanation(card)
            
            # Stock number header
            c.setFont("Helvetica-Bold", 14)
            c.setFillColor(HexColor('#1a73e8'))
            c.drawString(2*cm, y, f"#{idx}. {card.get('ticker')}")
            y -= 0.7*cm
            
            # Key metrics in a box
            c.setFillColor(HexColor('#000000'))
            c.setFont("Helvetica-Bold", 11)
            c.drawString(2.5*cm, y, f"Conviction: {card.get('conviction')}")
            c.drawString(7*cm, y, f"Probability: {card.get('p_up'):.1%}")
            c.drawString(12*cm, y, f"Last Price: ₹{card.get('last_close')}")
            y -= 0.6*cm
            
            c.setFont("Helvetica", 10)
            c.drawString(2.5*cm, y, f"Target: ₹{card.get('target')}")
            c.drawString(7*cm, y, f"Stop Loss: ₹{card.get('stop_loss')}")
            upside = ((card.get('target') - card.get('last_close')) / card.get('last_close') * 100)
            c.drawString(12*cm, y, f"Upside: {upside:.1f}%")
            y -= 0.8*cm
            
            # Explanation section
            c.setFont("Helvetica-Bold", 10)
            c.drawString(2.5*cm, y, "Analysis:")
            y -= 0.5*cm
            
            # Wrap text for explanation
            c.setFont("Helvetica", 9)
            max_width = 16*cm
            words = enhanced_reason.split()
            line = ""
            for word in words:
                test_line = line + word + " "
                if c.stringWidth(test_line, "Helvetica", 9) < max_width:
                    line = test_line
                else:
                    c.drawString(2.5*cm, y, line.strip())
                    y -= 0.45*cm
                    line = word + " "
            if line:
                c.drawString(2.5*cm, y, line.strip())
                y -= 0.45*cm
            
            # Separator line
            y -= 0.3*cm
            c.setStrokeColor(HexColor('#cccccc'))
            c.line(2*cm, y, width - 2*cm, y)
            y -= 0.8*cm
            
            # Page break if needed
            if y < 4*cm and idx < len(cards):
                c.showPage()
                y = height - 2*cm
                c.setFillColor(HexColor('#000000'))

        # Footer
        c.setFont("Helvetica-Oblique", 8)
        c.setFillColor(HexColor('#666666'))
        c.drawString(2*cm, 1.5*cm, "⚠ Educational tool. Not investment advice. Please do your own research.")
        
        c.showPage()
        c.save()
        return str(path)
