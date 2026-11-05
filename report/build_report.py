"""Build the article PDF from article.md, regenerating its figures from the analysis code.

Usage:
    python report/build_report.py
    python report/build_report.py --style mono --output report/thief-zone-knn-mono.pdf
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import tempfile
from pathlib import Path

import markdown
import matplotlib
from bs4 import BeautifulSoup
from matplotlib import font_manager
from playwright.sync_api import sync_playwright

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import plots
import thiefzone as tz

FIGURES = {
    "well-section": 164,
    "knn-rock-space": 140,
    "property-fingerprint": 152,
    "lorenz-plot": 110,
}

THEMES = {"accent": "report", "mono": "report-mono"}

TITLE = "Locating a thief zone with k-nearest neighbours"

MATH_BLOCK = re.compile(r"^```math\n(.*?)\n```$", re.M | re.S)
MATH_INLINE = re.compile(r"(?<![\\$])\$(?=\S)(.+?)(?<=\S)\$")
PLACEHOLDER = re.compile(r"MATH(BLOCK|INLINE)(\d{3})Z")

RENDER_MATH = """
window.mathErrors = [];
for (const el of document.querySelectorAll("[data-tex]")) {
  try {
    katex.render(el.dataset.tex, el, {displayMode: el.tagName === "DIV", throwOnError: true});
  } catch (error) {
    window.mathErrors.push(String(error));
  }
}
"""


def register_fonts() -> None:
    for path in sorted((HERE / "fonts").glob("*.ttf")):
        font_manager.fontManager.addfont(str(path))


def make_figures(style: str, directory: Path, formats: tuple[str, ...]) -> None:
    theme = THEMES[style]
    table = tz.load_zone_table()
    train = tz.leaf_catalogue(table)
    test = tz.suspect_frame()
    _, neighbours = tz.locate(train, test, tz.FEATURE_VIEWS["petrophysical"], k=3)
    nearest = tz.nearest_by_property(train, test, tz.FEATURE_VIEWS["all"])
    profile = tz.flow_profile(tz.devonian_column(table))
    directory.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        drawers = {
            "well-section": lambda p: plots.well_section(table, theme, p),
            "knn-rock-space": lambda p: plots.rock_space(train, test, neighbours["zone"].head(3).tolist(), theme, p),
            "property-fingerprint": lambda p: plots.property_fingerprint(train, test, nearest, theme, p),
            "lorenz-plot": lambda p: plots.lorenz_plot(profile, theme, p),
        }
        for name, draw in drawers.items():
            matplotlib.pyplot.close(draw(directory / f"{name}.{extension}"))


def is_number(text: str) -> bool:
    cleaned = text.replace(",", "").replace("%", "").strip()
    if cleaned in {"-", ""}:
        return True
    try:
        float(cleaned)
    except ValueError:
        return False
    return True


def protect_math(text: str) -> tuple[str, list[str]]:
    """Swap LaTeX for placeholders so Markdown leaves it untouched."""
    formulas: list[str] = []

    def stash(kind: str, tex: str) -> str:
        formulas.append(tex.strip())
        return f"MATH{kind}{len(formulas) - 1:03d}Z"

    text = MATH_BLOCK.sub(lambda m: "\n\n" + stash("BLOCK", m.group(1)) + "\n\n", text)
    text = MATH_INLINE.sub(lambda m: stash("INLINE", m.group(1)), text)
    return text, formulas


def restore_math(soup: BeautifulSoup, formulas: list[str]) -> None:
    for paragraph in soup.find_all("p"):
        match = PLACEHOLDER.fullmatch(paragraph.get_text().strip())
        if match and match.group(1) == "BLOCK":
            block = soup.new_tag("div", attrs={"class": "equation", "data-tex": formulas[int(match.group(2))]})
            paragraph.replace_with(block)
    for node in soup.find_all(string=PLACEHOLDER):
        pieces = []
        position = 0
        for match in PLACEHOLDER.finditer(node):
            pieces.append(node[position:match.start()])
            pieces.append(soup.new_tag("span", attrs={"class": "math", "data-tex": formulas[int(match.group(2))]}))
            position = match.end()
        pieces.append(node[position:])
        for piece in pieces:
            node.insert_before(piece)
        node.extract()


def build_html(source: Path, figure_dir: Path) -> str:
    text, formulas = protect_math(source.read_text(encoding="utf-8"))
    soup = BeautifulSoup(markdown.markdown(text, extensions=["tables"]), "html.parser")
    restore_math(soup, formulas)

    for rule in soup.find_all("hr"):
        rule.decompose()

    h1 = soup.find("h1")
    subtitle = h1.find_next_sibling("h3")
    byline = subtitle.find_next_sibling("p")
    parts = [part.strip() for part in byline.get_text().split("·")]
    header = soup.new_tag("header", attrs={"class": "title-block"})
    kicker = soup.new_tag("p", attrs={"class": "kicker"})
    kicker.string = "Technical note"
    header.append(kicker)
    title = soup.new_tag("h1")
    heading_text = h1.get_text()
    keep = "k-nearest neighbours"
    if keep in heading_text:
        title.append(heading_text.split(keep)[0])
        unbroken = soup.new_tag("span", attrs={"style": "white-space: nowrap"})
        unbroken.string = keep
        title.append(unbroken)
    else:
        title.string = heading_text
    header.append(title)
    sub = soup.new_tag("p", attrs={"class": "subtitle"})
    sub.string = subtitle.get_text()
    header.append(sub)
    line = soup.new_tag("p", attrs={"class": "byline"})
    author = soup.new_tag("span", attrs={"class": "author"})
    author.string = parts[0]
    rest = soup.new_tag("span")
    rest.string = "  ·  ".join(p.replace("Code and data: ", "") for p in parts[1:])
    line.append(author)
    line.append(rest)
    header.append(line)
    h1.replace_with(header)
    subtitle.decompose()
    byline.decompose()

    for paragraph in soup.find_all("p"):
        lead = paragraph.find("strong")
        if lead is None or paragraph.contents[0] is not lead:
            continue
        label = lead.get_text().strip()
        if label in ("Summary.", "Data and code."):
            wrapper = soup.new_tag("div", attrs={"class": "summary" if label == "Summary." else "note"})
            tag = soup.new_tag("span", attrs={"class": "label"})
            tag.string = label.rstrip(".")
            lead.replace_with(tag)
            paragraph.wrap(wrapper)

    for heading in soup.find_all("h2"):
        match = re.match(r"(\d+)\.\s+(.*)", heading.get_text())
        if match:
            heading.clear()
            number = soup.new_tag("span", attrs={"class": "number"})
            number.string = match.group(1)
            heading.append(number)
            heading.append(match.group(2))

    for image in soup.find_all("img"):
        paragraph = image.parent
        caption = paragraph.find_next_sibling("p")
        name = Path(image["src"]).stem
        figure = soup.new_tag("figure", attrs={"class": "chart"})
        img = soup.new_tag("img", attrs={"src": (figure_dir / f"{name}.svg").as_uri(), "alt": image.get("alt", "")})
        img["style"] = f"width: {FIGURES[name]}mm; margin: 0 auto;"
        figure.append(img)
        figcaption = soup.new_tag("figcaption")
        for child in list(caption.contents):
            figcaption.append(child.extract())
        figure.append(figcaption)
        paragraph.replace_with(figure)
        caption.decompose()

    for table in soup.find_all("table"):
        caption = table.find_previous_sibling("p")
        columns = len(table.find("tr").find_all(["th", "td"]))
        figure = soup.new_tag("figure", attrs={"class": "table" + (" narrow" if columns <= 4 else "")})
        if caption is not None and caption.get_text().startswith("Table"):
            figcaption = soup.new_tag("figcaption")
            for child in list(caption.contents):
                figcaption.append(child.extract())
            figure.append(figcaption)
            caption.decompose()
        rows = table.find("tbody").find_all("tr")
        for index in range(1, columns):
            cells = [row.find_all("td")[index] for row in rows]
            if all(is_number(cell.get_text()) for cell in cells):
                for cell in cells:
                    cell["class"] = "num"
                table.find("thead").find_all("th")[index]["class"] = "num"
        table.wrap(figure)

    references = next(h for h in soup.find_all("h2") if h.get_text().strip() == "References")
    wrapper = soup.new_tag("div", attrs={"class": "references"})
    references.insert_before(wrapper)
    node = references
    while node is not None:
        following = node.find_next_sibling()
        wrapper.append(node.extract())
        node = following

    return (
        "<!doctype html><html lang='en-GB'><head><meta charset='utf-8'>"
        f"<base href='{HERE.as_uri()}/'><title>{html.escape(TITLE)}</title>"
        "<link rel='stylesheet' href='katex/katex.min.css'>"
        "<link rel='stylesheet' href='style.css'>"
        "<script src='katex/katex.min.js'></script></head>"
        f"<body><main>{soup}</main><script>{RENDER_MATH}</script></body></html>"
    )


def render_pdf(html: str, output: Path) -> None:
    with tempfile.TemporaryDirectory() as scratch:
        page_path = Path(scratch) / "article.html"
        page_path.write_text(html, encoding="utf-8")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page()
            page.goto(page_path.as_uri(), wait_until="networkidle")
            page.evaluate("document.fonts.ready")
            errors = page.evaluate("window.mathErrors")
            if errors:
                raise SystemExit("Equation errors:\n" + "\n".join(errors))
            page.pdf(path=str(output), print_background=True, prefer_css_page_size=True)
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--style", choices=sorted(THEMES), default="accent")
    parser.add_argument("--output", type=Path, default=HERE / "thief-zone-knn.pdf")
    parser.add_argument("--figures", type=Path, default=HERE / "figures")
    args = parser.parse_args()

    register_fonts()
    make_figures(args.style, args.figures, ("svg", "png"))
    render_pdf(build_html(HERE / "article.md", args.figures), args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
