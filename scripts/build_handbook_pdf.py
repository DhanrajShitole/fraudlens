"""
FraudLens: Complete Project Handbook PDF Generator (Omnibus Edition)
Compiles all 9 documentation modules into a single publication-ready PDF:
- Cover Page
- Master Table of Contents
- Section A: Executive Summary
- Section B: System Architecture Deep Dive
- Section C: Machine Learning & Explainability Deep Dive
- Section D: AI Agent & LLM Copilot Deep Dive
- Section E: API & Integration Reference
- Section F: DevOps & MLOps Guide
- Section G: Setup & Troubleshooting Guide
- Section H: Presentation, Demo & Viva Voce Guide
- Section I: Master 20-Chapter Learning Handbook & Technical Glossary
"""

import os
import re
import html
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable, Preformatted
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            return

        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#475569"))  # Slate grey

        # Running Header
        self.drawString(45, 11 * 72 - 36, "FRAUDLENS: COMPLETE PROJECT HANDBOOK & DEFENSE GUIDE")
        self.drawRightString(8.5 * 72 - 45, 11 * 72 - 36, "BTech Final-Year Capstone Project")
        
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.75)
        self.line(45, 11 * 72 - 42, 8.5 * 72 - 45, 11 * 72 - 42)

        # Running Footer
        self.line(45, 45, 8.5 * 72 - 45, 45)
        self.setFont("Helvetica", 8)
        self.drawString(45, 32, "Confidential — Academic & Technical Interview Handbook | Author: Dhanraj Shitole")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 45, 32, page_str)
        
        self.restoreState()


def clean_markdown_inline(text):
    if not text:
        return ""
    text = html.escape(text)
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    text = re.sub(r'`(.+?)`', r'<font face="Courier" color="#991B1B"><b>\1</b></font>', text)
    text = re.sub(r'\$(.+?)\$', r'<font face="Helvetica-Oblique" color="#1E3A8A"><i>\1</i></font>', text)
    return text


def build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='CoverTitle',
        fontName='Helvetica-Bold',
        fontSize=30,
        leading=36,
        textColor=colors.HexColor('#0F172A'),
        alignment=1,
        spaceAfter=14
    ))
    styles.add(ParagraphStyle(
        name='CoverSubtitle',
        fontName='Helvetica',
        fontSize=15,
        leading=20,
        textColor=colors.HexColor('#2563EB'),
        alignment=1,
        spaceAfter=25
    ))
    styles.add(ParagraphStyle(
        name='CoverMeta',
        fontName='Helvetica',
        fontSize=10,
        leading=16,
        textColor=colors.HexColor('#475569'),
        alignment=1,
        spaceAfter=8
    ))

    styles.add(ParagraphStyle(
        name='SectionHeader',
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        name='HandbookH1',
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        name='HandbookH2',
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=10,
        spaceAfter=5,
        keepWithNext=True
    ))
    styles.add(ParagraphStyle(
        name='HandbookH3',
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13.5,
        textColor=colors.HexColor('#334155'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        name='HandbookBody',
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=5
    ))
    styles.add(ParagraphStyle(
        name='HandbookBullet',
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.5,
        textColor=colors.HexColor('#1E293B'),
        leftIndent=14,
        firstLineIndent=-9,
        spaceAfter=2.5
    ))
    styles.add(ParagraphStyle(
        name='HandbookCallout',
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor('#1E3A8A'),
        leftIndent=12,
        rightIndent=12,
        spaceBefore=5,
        spaceAfter=5
    ))

    styles.add(ParagraphStyle(
        name='HandbookCode',
        fontName='Courier',
        fontSize=7,
        leading=9.5,
        textColor=colors.HexColor('#0F172A')
    ))

    styles.add(ParagraphStyle(
        name='TableHeader',
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=0
    ))
    styles.add(ParagraphStyle(
        name='TableCell',
        fontName='Helvetica',
        fontSize=7,
        leading=9,
        textColor=colors.HexColor('#1E293B'),
        alignment=0
    ))

    return styles


def parse_markdown_to_flowables(md_content, styles, printable_width=522):
    flowables = []
    lines = md_content.splitlines()
    i = 0
    total_lines = len(lines)

    in_code_block = False
    code_lines = []

    while i < total_lines:
        line = lines[i]
        stripped = line.strip()

        # Handle fenced code block
        if stripped.startswith('```'):
            if in_code_block:
                code_text = "\n".join(code_lines)
                p = Preformatted(clean_markdown_inline(code_text), styles['HandbookCode'])
                code_table = Table([[p]], colWidths=[printable_width])
                code_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
                    ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor('#CBD5E1')),
                    ('PADDING', (0,0), (-1,-1), 5),
                    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ]))
                flowables.append(code_table)
                flowables.append(Spacer(1, 5))
                code_lines = []
                in_code_block = False
            else:
                in_code_block = True
                code_lines = []
            i += 1
            continue

        if in_code_block:
            code_lines.append(line)
            i += 1
            continue

        if not stripped:
            i += 1
            continue

        if stripped in ['---', '***', '___']:
            flowables.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor('#CBD5E1'), spaceBefore=6, spaceAfter=6))
            i += 1
            continue

        # Handle Markdown Tables
        if stripped.startswith('|') and stripped.endswith('|'):
            table_lines = []
            while i < total_lines and lines[i].strip().startswith('|') and lines[i].strip().endswith('|'):
                table_lines.append(lines[i].strip())
                i += 1
            
            parsed_rows = []
            for t_line in table_lines:
                raw_cells = [c.strip() for c in t_line.split('|')[1:-1]]
                if all(re.match(r'^:?-+:?$', c) for c in raw_cells):
                    continue
                parsed_rows.append(raw_cells)

            if parsed_rows:
                num_cols = max(len(r) for r in parsed_rows)
                col_width = printable_width / num_cols
                col_widths = [col_width] * num_cols

                table_data = []
                for row_idx, r in enumerate(parsed_rows):
                    row_cells = []
                    for c_idx in range(num_cols):
                        cell_txt = r[c_idx] if c_idx < len(r) else ""
                        formatted_cell = clean_markdown_inline(cell_txt)
                        if row_idx == 0:
                            row_cells.append(Paragraph(formatted_cell, styles['TableHeader']))
                        else:
                            row_cells.append(Paragraph(formatted_cell, styles['TableCell']))
                    table_data.append(row_cells)

                t = Table(table_data, colWidths=col_widths, repeatRows=1)
                t_style = [
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
                    ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor('#CBD5E1')),
                    ('PADDING', (0,0), (-1,-1), 3.5),
                ]
                for r_i in range(1, len(table_data)):
                    bg = colors.HexColor('#F8FAFC') if r_i % 2 == 1 else colors.white
                    t_style.append(('BACKGROUND', (0, r_i), (-1, r_i), bg))
                
                t.setStyle(TableStyle(t_style))
                flowables.append(t)
                flowables.append(Spacer(1, 5))
            continue

        # Headings
        if stripped.startswith('#### '):
            h_text = clean_markdown_inline(stripped[5:].strip())
            flowables.append(Paragraph(h_text, styles['HandbookH3']))
            i += 1
            continue
        elif stripped.startswith('### '):
            h_text = clean_markdown_inline(stripped[4:].strip())
            flowables.append(Paragraph(h_text, styles['HandbookH3']))
            i += 1
            continue
        elif stripped.startswith('## '):
            h_text = clean_markdown_inline(stripped[3:].strip())
            flowables.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor('#94A3B8'), spaceBefore=10, spaceAfter=6))
            flowables.append(Paragraph(h_text, styles['HandbookH2']))
            i += 1
            continue
        elif stripped.startswith('# '):
            h_text = clean_markdown_inline(stripped[2:].strip())
            flowables.append(Paragraph(h_text, styles['HandbookH1']))
            i += 1
            continue

        # Blockquotes
        if stripped.startswith('>'):
            callout_text = stripped.lstrip('> ').strip()
            callout_text = re.sub(r'\[!(NOTE|IMPORTANT|TIP|WARNING|CAUTION)\]\s*', r'<b>\1:</b> ', callout_text)
            p = Paragraph(clean_markdown_inline(callout_text), styles['HandbookCallout'])
            callout_table = Table([[p]], colWidths=[printable_width])
            callout_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#EFF6FF')),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor('#BFDBFE')),
                ('LINEBEFORE', (0,0), (0,-1), 3, colors.HexColor('#2563EB')),
                ('PADDING', (0,0), (-1,-1), 5),
            ]))
            flowables.append(callout_table)
            flowables.append(Spacer(1, 3))
            i += 1
            continue

        # Bullets
        if stripped.startswith(('- ', '* ')):
            bullet_txt = "&bull; " + clean_markdown_inline(stripped[2:].strip())
            flowables.append(Paragraph(bullet_txt, styles['HandbookBullet']))
            i += 1
            continue

        # Numbered List Items
        m_num = re.match(r'^(\d+)\.\s+(.*)$', stripped)
        if m_num:
            num_str = m_num.group(1)
            item_txt = f"<b>{num_str}.</b> " + clean_markdown_inline(m_num.group(2).strip())
            flowables.append(Paragraph(item_txt, styles['HandbookBullet']))
            i += 1
            continue

        # Regular Body Paragraph
        p_text = clean_markdown_inline(stripped)
        flowables.append(Paragraph(p_text, styles['HandbookBody']))
        i += 1

    return flowables


def generate_omnibus_handbook_pdf(output_pdf_path="docs/FRAUDLENS_COMPLETE_PROJECT_HANDBOOK.pdf"):
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=45,
        rightMargin=45,
        topMargin=54,
        bottomMargin=54
    )

    styles = build_styles()
    printable_width = 8.5 * 72 - 90  # 522 pt

    story = []

    # =========================================================================
    # 1. COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 35))
    story.append(Paragraph("FRAUDLENS", styles['CoverTitle']))
    story.append(Paragraph("A Real-Time, Explainable Fraud Detection Platform<br/>with an Agentic AI Investigation Copilot", styles['CoverSubtitle']))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor('#2563EB'), spaceBefore=5, spaceAfter=20))
    
    cover_meta = """
    <b>Complete Technical Project Documentation & Learning Handbook (Omnibus Edition)</b><br/>
    <b>Author:</b> Dhanraj Shitole<br/>
    <b>Degree:</b> Bachelor of Technology (BTech) — Final-Year Capstone Project<br/>
    <b>Academic Session:</b> 2025 – 2026<br/>
    <b>Repository:</b> <font color="#2563EB">https://github.com/DhanrajShitole/fraudlens</font><br/>
    <br/>
    <b>Core Technology Stack:</b><br/>
    Python 3.13 &bull; LightGBM &bull; SHAP TreeExplainer &bull; SMOTE &bull; Ollama Llama 3.1 8B<br/>
    FastAPI &bull; React Dashboard &bull; SQLite &bull; MLflow Registry &bull; PSI Drift &bull; Docker
    """
    story.append(Paragraph(cover_meta, styles['CoverMeta']))
    story.append(Spacer(1, 20))

    box_summary = """
    <b>EXECUTIVE NOTICE & SCOPE OF THIS COMPREHENSIVE COMPILATION:</b><br/>
    This publication-grade handbook represents the definitive single source of truth for the FraudLens platform. It consolidates all technical guides, architectural schematics, mathematical formulations, evidence-grounded agent prompt contracts, anti-hallucination validation proofs, REST API references, DevOps containerization workflows, defense viva questions, and self-assessment checklists.
    """
    p_box = Paragraph(box_summary, styles['HandbookCallout'])
    t_box = Table([[p_box]], colWidths=[printable_width])
    t_box.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F1F5F9')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#94A3B8')),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_box)
    story.append(PageBreak())

    # =========================================================================
    # 2. TABLE OF CONTENTS
    # =========================================================================
    story.append(Paragraph("Handbook Table of Contents", styles['HandbookH1']))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#0F172A'), spaceBefore=4, spaceAfter=10))

    sections = [
        ("PART I: EXECUTIVE OVERVIEW & SYSTEM DESIGN", [
            ("1. Executive Summary", "Problem statement, core architectural innovations, and verified metrics"),
            ("2. Technical Architecture Deep Dive", "Offline vs online topology, sequence diagrams, and schema specifications"),
        ]),
        ("PART II: MACHINE LEARNING & EXPLAINABLE AI", [
            ("3. Machine Learning Deep Dive", "Mathematical formulations: LightGBM objective, SMOTE synthesis, TreeSHAP"),
            ("4. AI Agent & LLM Copilot Deep Dive", "Deterministic evidence tools, TF-IDF policy engine, prompt template, validator"),
        ]),
        ("PART III: SYSTEM INTEGRATION & MLOPS", [
            ("5. API & Integration Reference", "FastAPI architecture, lifespan loader, 5 endpoints with JSON schemas"),
            ("6. DevOps & MLOps Guide", "MLflow registry with 'production' alias, PSI drift detection, Docker config"),
            ("7. Setup & Troubleshooting Guide", "Cold-start reproduction, environment setup, CLI triggers, troubleshooting"),
        ]),
        ("PART IV: ACADEMIC DEFENSE & VIVA PREPARATION", [
            ("8. Presentation, Demo & Viva Guide", "Slide-by-slide structure, live demonstration script, 50+ viva Q&As"),
        ]),
        ("PART V: MASTER 20-CHAPTER LEARNING HANDBOOK", [
            ("Chapters 1 – 20 Comprehensive", "Full textbook-grade chapters, file walkthroughs, trade-offs, glossary & checklist"),
        ])
    ]

    for part_title, items in sections:
        story.append(Paragraph(f"<b>{part_title}</b>", styles['HandbookH2']))
        toc_table_data = []
        for title, desc in items:
            row = [
                Paragraph(f"<b>{title}</b>", styles['HandbookBody']),
                Paragraph(f"<i>{desc}</i>", styles['HandbookCallout'])
            ]
            toc_table_data.append(row)
        t_part = Table(toc_table_data, colWidths=[230, 292])
        t_part.setStyle(TableStyle([
            ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
            ('PADDING', (0,0), (-1,-1), 3),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(t_part)
        story.append(Spacer(1, 4))

    story.append(PageBreak())

    # =========================================================================
    # 3. APPEND EACH DOCUMENTATION MODULE
    # =========================================================================
    doc_manifest = [
        ("PART I: EXECUTIVE SUMMARY", "docs/FRAUDLENS_EXECUTIVE_SUMMARY.md"),
        ("PART II: TECHNICAL ARCHITECTURE DEEP DIVE", "docs/FRAUDLENS_TECHNICAL_ARCHITECTURE.md"),
        ("PART III: MACHINE LEARNING & EXPLAINABILITY DEEP DIVE", "docs/FRAUDLENS_ML_DEEP_DIVE.md"),
        ("PART IV: AI AGENT & LLM COPILOT DEEP DIVE", "docs/FRAUDLENS_AI_AGENT_DEEP_DIVE.md"),
        ("PART V: REST API & INTEGRATION REFERENCE", "docs/FRAUDLENS_API_REFERENCE.md"),
        ("PART VI: DEVOPS, MLOPS & CONTAINERIZATION GUIDE", "docs/FRAUDLENS_DEVOPS_MLOPS_GUIDE.md"),
        ("PART VII: SETUP, EXECUTION & TROUBLESHOOTING GUIDE", "docs/FRAUDLENS_SETUP_AND_TROUBLESHOOTING.md"),
        ("PART VIII: PRESENTATION, DEMO & VIVA VOCE DEFENSE GUIDE", "docs/FRAUDLENS_PRESENTATION_AND_VIVA_GUIDE.md"),
        ("PART IX: MASTER 20-CHAPTER LEARNING HANDBOOK", "docs/FRAUDLENS_COMPLETE_PROJECT_HANDBOOK.md"),
    ]

    for part_banner, file_path in doc_manifest:
        if os.path.exists(file_path):
            print(f"Adding {file_path}...")
            story.append(Paragraph(part_banner, styles['SectionHeader']))
            story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1E3A8A'), spaceBefore=4, spaceAfter=12))
            
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()

            raw_text = re.sub(r'^#\s+FraudLens:.*?\n', '', raw_text, flags=re.MULTILINE)
            flowables = parse_markdown_to_flowables(raw_text, styles, printable_width)
            story.extend(flowables)
            story.append(PageBreak())
        else:
            print(f"Warning: File not found: {file_path}")

    print(f"Compiling final PDF at {output_pdf_path}...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print("Omnibus PDF compilation successful!")

if __name__ == "__main__":
    generate_omnibus_handbook_pdf()
