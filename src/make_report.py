"""Render the report to a self-contained, print-ready HTML file.

Run: ``python -m src.make_report``  ->  ``report/report.html``
Then open it and print to PDF (Ctrl+P, "Save as PDF").

Why not pandoc or a markdown library: neither is installed, and adding a
dependency for a formatting step would make the project harder to reproduce
for no gain in the result. This handles exactly the subset of Markdown the
report uses -- headings, tables, fenced code, lists, blockquotes, rules,
inline emphasis, code and links -- and nothing else. If the report ever needs
more, use a real library rather than growing this.

The stylesheet targets paper: serif body, tabular figures so columns of
numbers line up, and page-break rules so a table is never split across a page
boundary.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

__all__ = ["render", "main"]

CSS = """
@page { size: A4; margin: 18mm 16mm; }
:root {
  --ink: #16191d; --muted: #5b6470; --rule: #d7dce3;
  --accent: #1f4e79; --code-bg: #f5f7fa;
}
* { box-sizing: border-box; }
body {
  font: 10.5pt/1.55 "Charter", "Georgia", "Times New Roman", serif;
  color: var(--ink); max-width: 46rem; margin: 2rem auto; padding: 0 1rem;
  -webkit-font-smoothing: antialiased;
}
h1 { font-size: 1.9rem; line-height: 1.2; margin: 0 0 .2rem; letter-spacing: -.01em; }
h2 {
  font-size: 1.22rem; margin: 2.1rem 0 .7rem; padding-bottom: .28rem;
  border-bottom: 2px solid var(--accent); color: var(--accent);
  page-break-after: avoid;
}
h3 { font-size: 1.02rem; margin: 1.5rem 0 .45rem; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
a { color: var(--accent); }
hr { border: 0; border-top: 1px solid var(--rule); margin: 2rem 0; }
blockquote {
  margin: 1rem 0; padding: .6rem .95rem; background: #fffbe9;
  border-left: 3px solid #e0b93c; color: #5c4b12;
}
blockquote p { margin: .25rem 0; }
table {
  border-collapse: collapse; width: 100%; margin: .9rem 0;
  font-size: 9.4pt; font-variant-numeric: tabular-nums;
  page-break-inside: avoid;
}
th, td { border: 1px solid var(--rule); padding: .34rem .5rem; text-align: left;
         vertical-align: top; }
th { background: #eef2f7; font-weight: 600; }
tbody tr:nth-child(even) { background: #fafbfc; }
code {
  font-family: "Cascadia Mono", "Consolas", monospace; font-size: .87em;
  background: var(--code-bg); padding: .06em .3em; border-radius: 3px;
}
pre {
  background: var(--code-bg); border: 1px solid var(--rule); border-radius: 4px;
  padding: .7rem .9rem; overflow-x: auto; page-break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: .84em; }
.subtitle { color: var(--muted); font-size: 1rem; margin: 0 0 1.6rem; }
figure { margin: 1.1rem 0; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; height: auto; border: 1px solid var(--rule);
             border-radius: 4px; }
figcaption { font-size: 8.6pt; color: var(--muted); margin-top: .35rem; }
@media print { body { margin: 0; max-width: none; } a { color: var(--ink); } }
"""

_INLINE = (
    (re.compile(r"`([^`]+)`"), lambda m: f"<code>{html.escape(m.group(1))}</code>"),
    (re.compile(r"\*\*([^*]+)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])"), r"<em>\1</em>"),
    # Images MUST precede links: `![alt](src)` also matches the link pattern,
    # and if the link rule wins you get a hyperlink whose text is "!alt" and
    # no picture. The first render silently dropped all four figures this way.
    (re.compile(r"!\[([^\]]*)\]\(([^)]+)\)"),
     r'<figure><img src="\2" alt="\1"><figcaption>\1</figcaption></figure>'),
    (re.compile(r"\[([^\]]+)\]\(([^)]+)\)"), r'<a href="\2">\1</a>'),
)


def _inline(text: str) -> str:
    """Escape, then apply inline markup. Code spans are protected first."""
    slots: list[str] = []

    def stash(m: re.Match) -> str:
        slots.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x00{len(slots)-1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text)
    for pat, rep in _INLINE[1:]:
        text = pat.sub(rep, text)
    return re.sub(r"\x00(\d+)\x00", lambda m: slots[int(m.group(1))], text)


def _table(rows: list[str]) -> str:
    def cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    head, body = cells(rows[0]), [cells(r) for r in rows[2:]]
    out = ["<table><thead><tr>"]
    out += [f"<th>{_inline(c)}</th>" for c in head]
    out.append("</tr></thead><tbody>")
    for r in body:
        out.append("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def render(md: str, title: str = "Report") -> str:
    lines, out, i = md.split("\n"), [], 0
    while i < len(lines):
        ln = lines[i]

        if ln.startswith("```"):                                  # fenced code
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            code = html.escape("\n".join(lines[i + 1:j]))
            out.append(f"<pre><code>{code}</code></pre>")
            i = j + 1
            continue

        if re.match(r"^\|.*\|\s*$", ln) and i + 1 < len(lines) \
                and re.match(r"^\|[\s:|-]+\|\s*$", lines[i + 1]):
            j = i
            while j < len(lines) and re.match(r"^\|.*\|\s*$", lines[j]):
                j += 1
            out.append(_table(lines[i:j]))
            i = j
            continue

        if re.match(r"^(---|\*\*\*|___)\s*$", ln):
            out.append("<hr>"); i += 1; continue

        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>")
            i += 1
            continue

        if ln.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].startswith(">"):
                buf.append(lines[i].lstrip("> ").rstrip()); i += 1
            out.append("<blockquote><p>" + _inline(" ".join(buf)) + "</p></blockquote>")
            continue

        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", ln)
        if m:
            ordered = bool(re.match(r"^\d+\.$", m.group(2)))
            tag = "ol" if ordered else "ul"
            items = []
            while i < len(lines):
                mm = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[i])
                if not mm:
                    # a wrapped continuation line belongs to the current item
                    if items and lines[i].strip() and lines[i].startswith(("  ", "\t")):
                        items[-1] += " " + lines[i].strip(); i += 1; continue
                    break
                items.append(mm.group(3)); i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(x)}</li>" for x in items)
                       + f"</{tag}>")
            continue

        if ln.strip().startswith("!["):
            out.append(_inline(ln.strip()))
            i += 1
            continue

        if ln.strip():
            buf = []
            while i < len(lines) and lines[i].strip() \
                    and not re.match(r"^(#{1,4}\s|\||>|```|---)", lines[i]) \
                    and not re.match(r"^(\s*)([-*]|\d+\.)\s+", lines[i]):
                buf.append(lines[i].strip()); i += 1
            cls = ' class="subtitle"' if len(out) == 1 and out[0].startswith("<h1") else ""
            out.append(f"<p{cls}>" + _inline(" ".join(buf)) + "</p>")
            continue
        i += 1

    return (f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body>{''.join(out)}</body></html>")


def main() -> None:
    src = Path("report/report.md")
    dst = Path("report/report.html")
    dst.write_text(render(src.read_text(encoding="utf-8"),
                          "Patient Timeline Forecasting"), encoding="utf-8")
    print(f"wrote {dst}  ({dst.stat().st_size:,} bytes)")
    print("open it and print to PDF (Ctrl+P -> Save as PDF), then rename to "
          "<your_name>_results.pdf")


if __name__ == "__main__":
    main()
