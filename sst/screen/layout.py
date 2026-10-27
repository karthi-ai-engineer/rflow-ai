"""OCR lines as blocks to translate: a wrapped paragraph's lines are merged, so Gemini sees whole sentences; menus,
lists and table cells stay one block a line. Lines with no Japanese, and a lone kanji (almost always a piece the OCR cut
out of a longer word: 伊藤 -> 伊 -> "Italy"), aren't translated."""
import re
from dataclasses import dataclass, field

from sst.screen.ocr import OcrLine
from sst.screen.translate import UI_GLOSSARY

_JP = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ々〆ヵヶ]")
_SENTENCE_END = tuple("。．.！!？?：:」』)）")


def has_japanese(text: str) -> bool:
    return bool(_JP.search(text))


@dataclass
class Block:
    lines: list[OcrLine] = field(default_factory=list)

    @property
    def text(self) -> str:
        out = ""
        for ln in self.lines:
            if out and out[-1].isascii() and out[-1].isalnum() and ln.text[:1].isascii():
                out += " "
            out += ln.text
        return out

    @property
    def box(self) -> tuple[int, int, int, int]:
        x0, y0 = min(ln.x for ln in self.lines), min(ln.y for ln in self.lines)
        x1, y1 = max(ln.right for ln in self.lines), max(ln.bottom for ln in self.lines)
        return int(x0), int(y0), int(round(x1 - x0)) + 1, int(round(y1 - y0)) + 1

    @property
    def char_h(self) -> float:
        hs = sorted(ln.char_h for ln in self.lines)
        return hs[len(hs) // 2]


def _continues(block: Block, ln: OcrLine) -> bool:
    last = block.lines[-1]
    h = max(last.char_h, ln.char_h)
    bx, _, bw, _ = block.box
    return (0 <= ln.y - last.bottom < 0.9 * h  # the next line, a small gap
            and abs(ln.x - bx) < 1.2 * h  # left aligned
            and 0.75 < ln.char_h / max(last.char_h, 1) < 1.33  # the same font size
            and len(last.text) >= 8 and last.right >= bx + bw - 2.5 * h  # the line before is long and reached the edge
            and not last.text.endswith(_SENTENCE_END) and has_japanese(ln.text))


def build_blocks(lines: list[OcrLine], merge: bool = True) -> list[Block]:
    blocks: list[Block] = []
    for ln in sorted(lines, key=lambda ln: (round(ln.y / 4), ln.x)):
        target = next((b for b in reversed(blocks[-12:]) if _continues(b, ln)), None) if merge else None
        if target:
            target.lines.append(ln)
        else:
            blocks.append(Block([ln]))
    return [b for b in blocks if has_japanese(b.text) and _worth_translating(b.text)]


def _worth_translating(text: str) -> bool:
    t = text.strip()
    return len(t) > 1 or t in UI_GLOSSARY
