# The demo workspace

A desktop in a page for recording Rflow's demos with Rflow really working: **Notes**, **Docs**, **Chat** and a **Call**,
look-alike apps in a neutral style of their own (no real product's name, logo or look), with a made-up company (Aozora
Systems) and made-up people. Every take is the same, nobody real is on screen, and no accounts are needed.

Rflow does its real work here: dictation types into Notes, Docs and the chat box; Ctrl+C+C translates a chat message;
Text Transform rewrites the document; the Screen translator reads the Japanese chat; live translation and Meeting mode
hear the call, whose voices play through the speakers.

## Open it

Full screen, as its own window (Edge's app mode), at 1920x1080 and 100% scaling:

```
"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --app="file:///C:/Users/10520/karthi-tech/sst/demo/workspace/index.html" --start-fullscreen --autoplay-policy=no-user-gesture-required
```

Address options: `?open=chat` (or `notes`, `docs`, `call`; several with commas) brings a window to the front, `&join=1`
joins the call, `&theme=dark`, `&director=1`.

## Keys

- **F9**: the director (hidden while recording): reset everything, light or dark, open a window, the chat's next message,
  join the call, play the next line or all of them, and which voice files are there.
- **F8**: the next step in the window in front: in Chat the next scripted message ("Yui Tanaka is typing…" first), in the
  Call the next line.
- Windows move by their title bar, double-click maximises, the taskbar brings them back. Enter sends a chat message.

## The takes

| Take | In the workspace | Rflow |
|---|---|---|
| Dictation | Notes in front, click in it | Hold Ctrl+Win, say an email; it's typed |
| Translate | Chat: select Yui's message | Ctrl+C+C: the popup in English |
| Reply in Japanese | Chat: click the box, dictate "Sure, I'll email the customer today and confirm Thursday", then say "Japanese Teams message" | Text Transform: polite Japanese; Enter sends |
| Text Transform | Docs: click in the first paragraph | Say "make it concise", then "professional", "bullet points" |
| Screen translator | Chat in front | Ctrl+Alt+S: the Japanese channel in English, in place |
| Live translation | Call: Join, then F8 (or the director's Play all) | Ctrl+Alt+L, source "what the laptop plays": the captions; Speak the translation on: heard in English |
| Meeting mode | Call: Join (the window is then titled "… · Demo call", which Rflow's Meeting mode lists as **Demo call**), Meeting mode → Start, Play all, then Stop | The minutes: the release moved to Thursday, Haruto's fix by Tuesday, Karthi to contact the customer by Friday |

The story fits together: the chat says the release may move, the call decides it, and the minutes and the Docs update
follow from it.

## The voices

`audio/01-yui.wav` … `06-yui.wav`, one per line of the call in `scenario.js`. The ones in git are placeholders said by
Windows' own Japanese voices (`uv run --no-sync python scripts/make_demo_voices.py`, which plays nothing). For the real
demo, record each line (a colleague, or ourselves) and save it under the same name as a 16-bit WAV, then run the script
again: it measures the files that are there, so the speaking rings follow the real voices.

## Changing the story

Everything said and written is in `scenario.js`: the people, the chat (`start` and `next`), the document, the call's
lines (with their English, for the script) and the scene's clock.
