"""Screen translation (phase 51, the owner's own screen translator brought into Rflow): a frame put over any app, whose
Japanese text is covered by English stickers in the same place, in matching colours.

The frame is hollow: its inside is click-through, so the app underneath is used as always, and a click on a sticker
reaches the button under it. The stickers are painted in a window of their own that screen capture leaves out
(WDA_EXCLUDEFROMCAPTURE), so Rflow never reads its own English.

    capture.py    the frame's inside, copied from the screen (GDI BitBlt), as a Frame (BGRA, absolute screen pixels)
    winocr.py     Windows' own OCR through ctypes: on this laptop, free
    ocr.py        its words as lines: Japanese rebuilt without the spaces Windows puts in, table cells split, the lines
                  a frame edge cuts off dropped
    layout.py     lines as blocks to translate: a wrapped paragraph is one block, menus and table cells one per line
    tracker.py    each sticker remembers the pixels it was read from and follows them (scrolling) or hides; the change
                  detector says when the screen has settled and needs reading again
    translate.py  Gemini (the profile's key, sst.gateway) translates a screen's new strings in one request, cached;
                  a glossary for single UI words; every call counted in the cost tracker as "screen" (sst.usage)
    render.py     the stickers, painted: the background matched, the font fitted to the box
    lens.py       the frame, the stickers' window and the worker thread (capture, track, OCR, translate)
    control.py    what Rflow's app calls: start, stop, the shortcut, the one-time notice

Its own package, kept apart like live translation (sst.live): it never imports sst.pipeline, sst.dictate or sst.audio
(tests/test_screen_lens.py checks). Standard library, numpy, PySide6 and ctypes only: no OCR or vision package.
From the owner's screen translator (kaipulla, 2026-10-09), its README's measurements: OCR 76-132 ms a full frame,
0.5-0.8% of a core idle on a static page.
"""
