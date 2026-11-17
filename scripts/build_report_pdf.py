"""Render the research manuscript to a paginated, linked A4 PDF.

The default builds a clearly labeled draft. Pass --final only after all pending
results are replaced and reviewed; final mode refuses placeholder text.
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import pymupdf


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "report_draft.md"


def inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)",
                  lambda m: f'<a href="{html.escape(m.group(2), quote=True)}">{m.group(1)}</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    return text


def markdown_to_html(source: str, draft: bool) -> str:
    lines = source.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith("!["):
            match = re.match(r"!\[([^]]*)\]\(([^)]+)\)", line)
            if match:
                path = (SOURCE.parent / match.group(2)).resolve()
                if not path.is_relative_to(ROOT):
                    raise ValueError("Figure path escapes repository")
                next_i = i + 1
                while next_i < len(lines) and not lines[next_i].strip():
                    next_i += 1
                caption = ""
                if next_i < len(lines) and lines[next_i].startswith("**Figure "):
                    caption = f"<p class='caption'>{inline(lines[next_i].strip())}</p>"
                    i = next_i
                blocks.append(f'<div class="figblock"><p class="figure"><img src="{path.relative_to(ROOT).as_posix()}" alt="{html.escape(match.group(1))}"/></p>{caption}</div>')
            i += 1
            continue
        if line.startswith("# "):
            blocks.append(f"<h1>{inline(line[2:])}</h1>")
            i += 1
            continue
        if line.startswith("## "):
            cls = ' class="appendix"' if line.startswith("## Appendix") else ""
            blocks.append(f"<h2{cls}>{inline(line[3:])}</h2>")
            i += 1
            continue
        if line.startswith("**["):
            match = re.match(r"\*\*(.+?)\*\*\s*(.*)", line)
            if match:
                labels = ("Question/contribution", "Modalities/assumptions", "Mechanism/objective",
                          "Data/protocol", "Baselines/evidence", "Resources/reproducibility",
                          "Limitations/connections")
                pattern = r"(?=(?:" + "|".join(re.escape(v) for v in labels) + r"):)"
                fields = re.split(pattern, match.group(2))
                inner = [f"<h3>{inline(match.group(1))}</h3>"]
                for field in fields:
                    if field.strip():
                        label, value = field.strip().split(":", 1)
                        inner.append(f"<p class='field'><b>{html.escape(label)}:</b> {inline(value.strip())}</p>")
                cls = "study break-before" if match.group(1).startswith("[5]") else "study"
                blocks.append(f"<div class='{cls}'>" + "".join(inner) + "</div>")
            i += 1
            continue
        if line.startswith("|"):
            table = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    tag = "th" if not table else "td"
                    table.append("<tr>" + "".join(f"<{tag}>{inline(cell)}</{tag}>" for cell in cells) + "</tr>")
                i += 1
            blocks.append('<table class="metrics">' + "".join(table) + "</table>")
            continue
        if line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "):
                items.append(f"<li>{inline(lines[i].strip()[2:])}</li>")
                i += 1
            blocks.append("<ul>" + "".join(items) + "</ul>")
            continue
        if re.match(r"^\d+\. ", line):
            items = []
            while i < len(lines) and re.match(r"^\d+\. ", lines[i].strip()):
                items.append(f"<li>{inline(re.sub(r'^\d+\. ', '', lines[i].strip()))}</li>")
                i += 1
            blocks.append("<ol>" + "".join(items) + "</ol>")
            continue
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "![", "|", "- ")):
            para.append(lines[i].strip())
            i += 1
        blocks.append(f"<p>{inline(' '.join(para))}</p>")
    label = '<div class="draft">DRAFT - improvement results pending</div>' if draft else ""
    return "<html><body>" + label + "".join(blocks) + "</body></html>"


CSS = """
body { font-family: sans-serif; font-size: 9.3pt; color: #192536; line-height: 1.42; }
h1 { font-size: 20pt; color: #133b50; margin: 0 0 7pt; }
h2 { font-size: 12.5pt; color: #13536a; margin: 17pt 0 6pt; page-break-after: avoid; }
h2.appendix { page-break-before: always; }
h3 { font-size: 10pt; color: #13536a; margin: 0 0 5pt; }
p { margin: 0 0 8pt; text-align: justify; }
p.figure { text-align: center; margin: 11pt 0 5pt; }
.figblock { page-break-inside: avoid; }
.caption { font-size: 8.3pt; margin-bottom: 9pt; }
img { max-width: 100%; max-height: 220pt; }
table { border-collapse: collapse; width: 100%; margin: 9pt 0; font-size: 7.5pt; }
th { background: #dcebf0; color: #153e53; font-weight: bold; text-align: left; }
td, th { padding: 5pt 4pt; border-bottom: 0.4pt solid #c5d2d8; }
li { margin: 0 0 4pt; }
ul, ol { margin: 3pt 0 9pt 16pt; }
code { font-family: monospace; font-size: 8pt; color: #375c70; }
a { color: #126a85; text-decoration: none; }
.draft { color: #a55326; font-size: 9pt; font-weight: bold; margin-bottom: 8pt; }
.study { border-left: 2pt solid #7db3be; padding-left: 8pt; margin: 0 0 12pt; page-break-inside: avoid; }
.break-before { page-break-before: always; }
.study .field { margin: 0 0 3pt; font-size: 8.4pt; }
"""


def build(final: bool) -> Path:
    source = SOURCE.read_text(encoding="utf-8")
    if final and ("pending JSON" in source or "pending measured" in source or "working draft" in source.lower()):
        raise ValueError("Report still contains draft or pending-result markers")
    target = ROOT / "output" / "pdf" / ("Ipek_YZM5110_Research.pdf" if final else "Ipek_YZM5110_Research_DRAFT.pdf")
    target.parent.mkdir(parents=True, exist_ok=True)
    story = pymupdf.Story(markdown_to_html(source, not final), user_css=CSS,
                         archive=str(ROOT))
    writer = pymupdf.DocumentWriter(str(target))
    page = pymupdf.paper_rect("a4")
    body = pymupdf.Rect(47, 43, page.width - 47, page.height - 50)
    while True:
        more, _ = story.place(body)
        device = writer.begin_page(page)
        story.draw(device)
        writer.end_page()
        if not more:
            break
    writer.close()
    doc = pymupdf.open(target)
    for n, p in enumerate(doc, 1):
        p.draw_line((47, page.height - 36), (page.width - 47, page.height - 36), color=(0.76, 0.82, 0.85), width=0.5)
        p.insert_text((47, page.height - 24), "YZM5110  |  Multimodal Skin Lesion Classification",
                      fontsize=7.4, color=(0.35, 0.43, 0.49))
        p.insert_text((page.width - 75, page.height - 24), str(n), fontsize=7.4,
                      color=(0.35, 0.43, 0.49))
    doc.saveIncr()
    print(target, "pages", len(doc))
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    build(args.final)

