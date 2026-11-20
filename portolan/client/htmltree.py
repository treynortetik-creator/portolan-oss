"""A small HTML tree over the standard-library parser, and `Page`, a
response with a lazily parsed DOM. The templates are our own and well
formed, so no new dependency is needed. Shared by `scripts/verify.py` and
the `portolan` CLI."""

from __future__ import annotations

from html.parser import HTMLParser
from typing import Any

import httpx

_VOID_TAGS = frozenset(
    "area base br col embed hr img input link meta param source track wbr".split()
)
_NON_TEXT_TAGS = frozenset({"script", "style", "template"})


class Node:
    def __init__(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tag = tag
        self.attrs = {k: (v if v is not None else "") for k, v in attrs}
        self.children: list[Node | str] = []

    def has_class(self, name: str) -> bool:
        return name in self.attrs.get("class", "").split()

    def descendants(self) -> list[Node]:
        out: list[Node] = []
        stack = [c for c in reversed(self.children) if isinstance(c, Node)]
        while stack:
            node = stack.pop()
            out.append(node)
            stack.extend(c for c in reversed(node.children) if isinstance(c, Node))
        return out

    def find_all(
        self, tag: str | None = None, cls: str | None = None, attrs: dict[str, str] | None = None
    ) -> list[Node]:
        return [
            n
            for n in self.descendants()
            if (tag is None or n.tag == tag)
            and (cls is None or n.has_class(cls))
            and all(n.attrs.get(k) == v for k, v in (attrs or {}).items())
        ]

    def find(
        self, tag: str | None = None, cls: str | None = None, attrs: dict[str, str] | None = None
    ) -> Node | None:
        found = self.find_all(tag, cls, attrs)
        return found[0] if found else None

    def text(self) -> str:
        parts: list[str] = []

        def walk(node: Node) -> None:
            if node.tag in _NON_TEXT_TAGS:
                return
            for child in node.children:
                if isinstance(child, str):
                    parts.append(child)
                else:
                    walk(child)

        walk(self)
        return " ".join("".join(parts).split())


class _TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("#root", [])
        self._stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag, attrs)
        self._stack[-1].children.append(node)
        if tag not in _VOID_TAGS:
            self._stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._stack[-1].children.append(Node(tag, attrs))

    def handle_endtag(self, tag: str) -> None:
        for i in range(len(self._stack) - 1, 0, -1):
            if self._stack[i].tag == tag:
                del self._stack[i:]
                return

    def handle_data(self, data: str) -> None:
        self._stack[-1].children.append(data)


def parse_html(text: str) -> Node:
    builder = _TreeBuilder()
    builder.feed(text)
    builder.close()
    return builder.root


class Page:
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self._dom: Node | None = None

    @property
    def text(self) -> str:
        return self.response.text

    @property
    def dom(self) -> Node:
        if self._dom is None:
            self._dom = parse_html(self.response.text)
        return self._dom

    def json(self) -> Any:
        return self.response.json()
