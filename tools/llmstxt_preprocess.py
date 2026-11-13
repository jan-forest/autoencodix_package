"""HTML pre-processing for the mkdocs-llmstxt plugin (see `llmstxt.preprocess` in mkdocs.yml).

Keeps llms.txt / llms-full.txt compact enough for LLM context windows:
- drops the "Source code in ..." blocks of the API reference (agents read the source in the repo),
- truncates long notebook cell outputs (training logs, large DataFrames) to a short excerpt.
Images are already removed by the plugin itself.
"""

from __future__ import annotations

from bs4 import BeautifulSoup, NavigableString

MAX_OUTPUT_CHARS = 800


def preprocess(soup: BeautifulSoup, output: str) -> None:  # noqa: ARG001
    for element in soup.find_all("details", class_="mkdocstrings-source"):
        element.decompose()

    for element in soup.find_all(class_="jp-OutputArea-output"):
        text = element.get_text()
        if len(text) > MAX_OUTPUT_CHARS:
            excerpt = text[:MAX_OUTPUT_CHARS].rstrip()
            pre = soup.new_tag("pre")
            pre.append(NavigableString(f"{excerpt}\n... [output truncated]"))
            element.replace_with(pre)
