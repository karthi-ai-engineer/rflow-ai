# Handoff

Where the project stands, so work can continue on any device. Updated at the end of every phase.
The product is **Rflow**. The repository is **rflow-ai** (renamed from Rach_flow on 2026-10-06, and from
Rach_Darling_Flow on 2026-10-01); the Python package keeps its name, `sst`. **Never create a repository named
Rach_flow or Rach_Darling_Flow:** installed copies up to 2.3.0 (and up to 1.6.0) check for updates under those names,
which GitHub redirects only while they stay free (checked after each rename: the old API names answer 301 and
reach the latest release). Locally: `git remote set-url origin https://github.com/karthi-ai-engineer/rflow-ai.git`.

_Last updated: 2026-10-23_

## Start here (a new session, or the owner's other laptop)

**Where things stand (2026-10-23):**
- **Rflow 2.6.1** (2026-10-23, the owner's "release it") = phases 45, 42 and 46:
  - Phase 45, live translation (#200): `_Checked` keeps comparing process loopback with the whole mix (3 s energy
    window, the start of the call kept); the meeting stays lowered (the owner's #160 choice) and the capture lifts it
    back by the Ducker's factor; Both on speakers gates the mic; Gemini refusals said plainly and not retried; no
    silence streamed; session resumption; per-session numbers logged; Gemini's own spoken translation played, Piper
    en_US-ryan-medium as the local voice (its data is CC BY-NC-SA: the owner accepted it; revisit before anything paid).
  - Phase 42, the wake word (#162): `sst/wake/` + `sst/wakeui.py`; the spotter from k2-fsa's GitHub release, the voice
    print from csukuangfj's repo; a name is learned from the user's own takes (sound-alike spellings); a miss stays on
    the try; "Listen on my Bluetooth headset" (`wake_bluetooth`); takes under −50 dBFS refused. The owner's "karthi"
    still missed in their first real tries (loud enough): suggest "hello karthi" or "okay type".
  - Phase 46 (#210): Gemini only — `gateway.OFFERED = ("gemini", "ollama", "vllm")` hides OpenAI, Groq and Anthropic
    everywhere (their code stays); a cost tracker (`sst/usage.py`, per-profile `usage.json`, a card on AI & models;
    an estimate: no API key can read the real spend); model lists corrected (gpt-4.1-nano gone, Haiku 5.5, 3.5 Flash →
    3.6). Next: the free tier (the `gemini_free_tier` setting is a placeholder), then "one key per provider".
  - Researched, not built: two-way interpreting through a virtual "Rflow Mic" (needs a driver: VB-CABLE bundled under
    its donationware terms first).
- **Rflow 2.6.0 = phase 44, a light, fast Rflow** (#184, the owner's "build all 16"; built in 6 parts by parallel builders, merged on
  `phase-44/speed`). Measured first (2026-10-22, 311 logged dictations, idle profiling, network, models): the window's
  orb, 4 keyboard hooks and 15 ms timers, OpenBLAS's thread buffers and the eager pages made Rflow heavy; Gemini's speech
  (~2.3 s) and cleanup (~0.9 s) the wait. Parts: A #185 light window (orb still unless live, `window.pages` built on
  demand, `OPENBLAS_NUM_THREADS=1` in `sst/__init__.py`, `sst/instance.py` hands over before heavy imports), B #186
  quiet when idle (one `hotkey.KeyboardHook`, no polling, 50 ms mic blocks, level on read), C #187 faster dictation
  (cleanup of finished sentences while speaking, `LLMConfig.early`; clipboard restored in the background, `paste.settle()`
  before any other clipboard use; a short last part sent with 3 s of context, `join_context_ms`; Parakeet warmed on the
  cloud's first failure), D #188 errors (`sst/netconnect.py`: 3 s connect budget, refusals 402/401/403/quota remembered
  5 min, never retried; a backup from another provider the user has a key for), E #189 streaming speech
  (`sst/engines/gemini_stream.py`, Gemini 3.5 Transcribe Live in the Gemini menu, batch fallback in
  `sst/pipeline/streaming.py`; **not yet tried against real Gemini**), F #190 setups chosen by language (`setups.resolve`).
  Real app, before → after: idle CPU hidden 3.2% → 0.1% of a core, window open 35% → 0.1%, private memory 491 → 107 MB,
  threads 25 → 10, a second start shows the window in 0.5 s (was 1.4 s). Still to check with real keys: Gemini Live, and
  `sst eval` for the short-last-part change.
- **Rflow 2.5.3** (2026-10-20, the owner's "publish it quickly"): about 50 small "Rflow" windows flashed open and
  closed at start-up (#181): widgets made visible before they had a parent show as windows of their own. Only `hide()`
  while building; `test_building_and_opening_the_window_flashes_no_other_window` watches Show events. To see such
  flashes, log Rflow's top-level windows while it starts (EnumWindows), not only console windows.
- **Rflow 2.5.2** (2026-10-14, the owner's "take control"): live translation from the computer heard nothing in the
  owner's Teams meetings on Bluetooth earbuds (#176): process loopback, even of Teams alone, got silence while the
  output's whole mix carried the call (measured live in the meeting; the output meters read 0 too). Now process
  loopback is checked against the whole mix (`wasapi._Checked`): 1.5 s of sound there and none through process
  loopback (not Rflow's own voice, `_voice_until`) switches to the whole mix for that stream (`hears_self` then). A
  console run with the whole mix translated the owner's meeting. Wake word (phase 42) is paused on `phase-42/check`.
- **Rflow 2.5.1** (2026-10-08): **phase 41, the spoken translation** (#158). (1) The voice repeated sentences (the
  owner's log: 3 lines → 10 sentences said, 9 → 30): a leading space on a line's first piece, Gemini re-sending a line
  revised (appended, not replaced: `gemini._join`), and a new line restarting the last; now `Speaker` remembers what
  it said by its words and says each sentence once (#159; seven Gemini patterns in `tests/test_live_voice.py`).
  (2) Other apps stay at the chosen level **the whole time** the voice is on (`Ducker`), the slider 0-100%; 0% is
  `INAUDIBLE` (0.03%: inaudible, still translated) (#160). Checked live with a Japanese TTS paragraph: 3 lines, 9
  sentences said, lowered once for the session.
- **Rflow 2.5.0** (2026-10-08, the owner's "put it"): **phase 40, Text Transform** (#151). (1) The owner's rule: the
  selected text, or with nothing selected **all the text in the box** (Ctrl+A in the field; refused over
  `BOX_LIMIT` 2,000 characters, never in a terminal; the caret put back): the old search for the last dictation by
  caret steps failed in Teams (#152). **Proven in the owner's real Teams self-chat** (ms-teams.exe, 113 characters,
  selected again after Teams dropped the selection, replaced). (2) The cleanup guard allows fillers and repeats
  (24 of 87 cleanups were refused for "dropped words") but still refuses lost names, numbers, negations (#153).
  (3) New transforms (#154): translate in place ("translate to Japanese", 翻訳して), a Japanese Teams message, Japanese
  internal/external email (the writer line only from `signature_name`/`signature_company`), fix grammar; a
  Japanese-aware `TransformGuard`; the menu grouped Tone · Format · Language (T = Translate). Real tests with
  gemini-3.5-flash-lite: Teams and translation good; **email structure and fix grammar's tenses need a prompt round**
  (next). The research: `research_notes/Rflow Text Transform expansion/notes.md` (git-ignored). Per-app formats: later.
- **Rflow 2.4.1** (2026-10-08, the owner's "release it"): phase 39. **Phase 39: live translation fixes** (the owner's
  report of 2026-10-08, issue #142): found from the log and live
  tests on the owner's laptop. (1) Microphone mode dropped lines already in the target language (Microphone → into
  English while speaking English gave an empty bar): now shown with "already English" (#143). (2) The microphone was
  Windows' default *communications* mic with call-style echo cancellation and noise suppression, ignoring the mic
  chosen in Rflow and never raised: now the chosen mic, opened raw (`IAudioClient2::SetClientProperties`), a 90 Hz
  high-pass, `Leveler.for_microphone()`, a phase-safe stereo downmix (#144; measured -47 → -40 dBFS on the Realtek
  mic). (3) A call on a Bluetooth headset plays on its hands-free device (the default output *for calls*): `_Speakers`
  adds it by endpoint loopback when it differs and plays, and says "Nothing plays on X: the sound may be on Y" (#145;
  **still to confirm with the owner's headset in a Teams call**). (4) The empty bar now says why ("Listening…",
  "Hearing sound, but no speech yet", "Nothing heard…", "You're speaking English…"; `LiveSession.note`, #147).
  Not an Rflow bug (#146, closed): Gemini Transcribe's "HTTP 400 Thinking is not enabled" on 2026-10-07 came from
  Google's preview model (the same request worked on 10-08); the log also shows the free Gemini key's quota (429) on
  2, 5 and 6 October, each time typed with Parakeet instead.
- **Phase 38: a website that shows what Rflow does** (the owner's request, issue #121): the hero's **Rflow lens**
  (`site/lens.js`, `site/lens.css`; spoken words → the orb with a label per real step → clean text; four examples;
  check it with `site/lens-preview.html`), sections by example instead of a gallery of settings screens (dictation,
  live translation, rewrite and translate, setups with costs), a "See it work" showcase under the hero playing five
  real feature clips (`site/showcase.js`; made by `scripts/make_feature_clips_a.py` and `_b.py` into `site/media/`),
  "Watch the 40-second demo" in a lightbox, and the demo video (`scripts/make_demo_video.py`: `docs/media/demo.mp4`, the LinkedIn cut
  `demo-linkedin.mp4`, `demo.gif`; the site plays `site/media/demo.mp4`). Parts: #122 lens, #123 page, #106 video.
  The owner's LinkedIn post (one post: voice typing + live translation, the 4:5 video, the link in the first
  comment) is drafted in the session, not in git.
- **Rflow 2.4.0** (2026-10-07): live translation at any volume, and the voice speaks like an interpreter (phase 37,
  issue #108): the computer's sound captured as 32-bit float and brought up to a normal level, so turning the sound
  down no longer stops the translation (measured: an app at 0.03 % volume, nothing heard before, every word after);
  while the voice speaks the other apps are lowered (a slider, 30 % by default) and come back after each sentence;
  muted Windows sound is named on the bar. See **Live translation at any volume**.
- **Rflow 2.3.0** (2026-10-06, the owner's "release the version on the site"): open source (MIT), live translation
  spoken aloud (phase 29), never lose work (phase 30) and **phase 31, the owner's UI review** (issue #81, built by
  five agents in parallel and merged part by part: #82-#88): ten sections, a big Start/Stop for live translation, all
  API keys in one place, the speech model in two groups, ready-made setups with what each costs a month, Start over,
  and fixes for retired or failing AI models. See **Clearer sections, setups and costs**.
- **Open source first, a closed layer later** (the owner's decision of 2026-10-06): the desktop app, everything it
  does today, is free and open source under the **MIT License** (the owner's choice; issue #79, branch
  `chore/open-source`): `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, a feature-request
  template, complete notices (`packaging/NOTICES.txt`; the installer ships `LICENSE.txt`) and a "Source code" row in
  Settings → Advanced. The Rflow name and logo stay outside the licence. Accounts, sign-in with Google, sync, the web
  and mobile come later as a **closed-source** layer in a separate private repository; their plan stays under
  discussion (`reports/Rflow accounts and device sync.md`, not in git: Supabase, encrypted sync, users not in Japan
  so no Japan hosting). Before the public announcement (as 3.0.0): phases 32 (privacy), 33 (first run) and 35
  (updates and code signing; open-source projects can be signed free by the SignPath Foundation) are recommended;
  the owner hasn't decided. Checked: no key or private address in any of the 248 commits (tests use the placeholder
  10.0.0.5). **sherpa-onnx's library has espeak-ng's code (GPL-3.0) compiled in**, so the installer already ships
  GPL code: fine for the open app, but the closed version must not bundle it as is.
- **The user testing of 2026-10-06** (a testing agent, as a user, isolated from the owner's data): 75 flaws (2 blockers,
  22 major, 39 minor, 12 polish) and the capabilities a complete app lacks, in `reports/Rflow user testing
  2026-10-06.md` (not in git; its scripts, to re-run, in `reports/user-testing-2026-10-06/scripts/`). The owner chose
  to fix them in phases: 30 never lose work, then (after the owner's own UI review came first, as 31) 32 privacy
  and data, 33 first run and shortcuts, 34 layout, accessibility and speed, 35 updates, install and trust, 36
  networks and moving PCs. The owner's defaults for 32: recordings kept 30 days, live transcripts on with a switch.
- **Never lose work** (phase 30, PR #78), released in 2.3.0. See **Never lose work**.
- **Live translation spoken aloud** (phase 29, PR #76, released in 2.3.0): the owner's
  chosen voice, Piper's Danny (English), reads each sentence of the translation out on the laptop, never heard and
  translated again. See **Live translation spoken aloud**.
- **Rflow 2.2.0** (2026-10-05, the owner's "release it"): live translation, a section of its own (phase 28, PR #74):
  its own sidebar section, Ctrl+Alt+L, a bar to move, resize and scroll back through with a ✕, and the source:
  Computer, Microphone or Both (the owner's own lines marked "You"). Released before the owner tried the section, the
  bar and the microphone by hand. See **Live translation, a section of its own**.
- **Rflow 2.1.0** (2026-10-05, the owner's "merge + release 2.1.0"): live captions, part 1 (phase 27, PR #72): what the
  laptop plays (a meeting, a video) translated while people speak, in a caption bar left out of screen shares, with a
  transcript; Gemini 3.5 Live Translate with the profile's Gemini key; a pipeline of its own (`sst/live/`). Tried by
  the owner against Google and in the app. **Next: part 2, the owner's choice "both ways in one bar"**: your own
  speech (microphone) translated into Japanese in the same bar, with a switch to show the bar in a screen share so
  colleagues can read it. See **Live captions**.
- **Rflow 2.0.2** (2026-10-05, the owner's "release it"): microphones that follow you (phase 26, PR #70: a headset
  plugged in and chosen recorded silence; the list now comes from Windows, open microphones are reopened when the
  devices change, a quiet one is reopened, no-sound recordings aren't transcribed), and the checks on Windows on ARM
  (PR #68: CI tests and self-tests the x64 app there too; the log's first line names the machine). See
  **Microphones that follow you** and **Different computers**.
- **Rflow 2.0.1** (2026-10-05, the owner's "let this be the logo"): the owner's new logo, a blue-to-violet ribbon "R",
  everywhere: the app icon (window, taskbar, tray, Rflow.exe, installer), the sidebar and the first run, the installer's
  pictures, the website (header, footer, favicon, touch icon, social card). See **The logo**.
- **Rflow 2.0.0** (2026-10-04, the owner's "implement it and release v2.0"): the whole UI in the "Obsidian Signal"
  design approved in Figma (phase 25, PR #64; see **Rflow UI 2.0**): five sections (Home, Words, Tools, AI & models,
  Settings), a three-step first run, the pill and popups redone, Geist fonts shipped, and the website in the same look.- **Rflow 1.10.1** (2026-10-04, the owner's "correct it in 1.10.1"): Translate's popup and the Text Transform menu take
  clicks. Since they came (1.7.0, 1.8.0) they had the pill's click-through window style, so every click went to the
  app behind them; only the keyboard worked (Esc, and 1-4 in the menu).
- **Rflow 1.10.0** (2026-10-04, the owner's "publish + release 1.10.0"): the Translate popup redone (phase 24).
- **Rflow 1.9.0** (the same day): settings that change only on purpose (phase 23, PR #58) and the fix for a dictation
  typed twice when its last part replaced the one before (PR #57, since 1.7.0). The owner updated to it through the
  in-app banner on the ARM64 laptop: the first in-app update there, and it worked.
- **A new dev laptop:** a Surface Laptop 7, Snapdragon X Plus (ARM64), 16 GB. The installed x64 Rflow works there
  under Windows' emulation (the owner dictates with it, Gemini for speech). The dev `.venv` is x64 Python 3.12 too,
  because CTranslate2 (Whisper) has no ARM64 build: `uv venv --python cpython-3.12-windows-x86_64-none .venv`.
- 1.8.0 (2026-10-02): Snippets (phase 21: say "my email", your text is typed; PR #52) and Translate (phase 22: select
  text, Ctrl+C+C, a popup with the translation; PR #54).
- 1.7.0 (the same day): Text Transform (phase 20: say "make it concise" or double-tap Ctrl; PR #50), the fix for text never said
  (PR #48) and the new names (Rach_flow, rachflow.vercel.app; PR #47). 1.6.0 brought the owner's voice
  pipeline (phase 19: parts transcribed while speaking, dictionary, formatting, a guarded AI cleanup, an always-on
  microphone). 1.5.0 brought speech recognition as building blocks (phases 13-18) and a 90 MB installer. 1.4.0 contained accuracy phases 10-12.
- **On the owner's voice** (150 read sentences, laptop microphone):
  - word errors 9.2% → 7.1%
  - held-out sets 6.7% → **5.4%**, better than Parakeet's own benchmark average of 5.9%
  - names and terms 40% → 24.5%, when Your words holds the names
- The owner **used 1.4.0 for real dictation** on both laptops, and updates to 1.5.0 from the banner. A raw-mode set
  is still to be read.
- **The owner's plan (2026-10-01):** Rflow becomes building blocks. Speech recognition is chosen like the AI cleanup:
  - 13: the building block itself (merged)
  - 14: Whisper turbo on this computer (PR #35)
  - 15: "Scan my computer" (PR #37)
  - 16: cloud speech models (PR #39)
  - 17: your own server (PR #41)
  - 18: Parakeet downloaded on demand, and version 1.5.0 (PR #43)

  Phases 14-18 were merged and released as **v1.5.0** on 2026-10-01; the installer went from 571 MB to 90 MB. The
  correction work (sound-alike fixer, confidence-gated cleanup) comes next. See **Next steps**.

**Read in this order:**
1. This section and **Next steps** (the end of this file).
2. `CLAUDE.md`: the working rules (authorship, no AI attribution, issue → branch → PR, the owner merges).
3. `docs/accuracy.md`: the accuracy plan, the target pipeline, and all measured results.
4. Only when needed:
   - `docs/research/accuracy-report.md`: the full research, with sources
   - `docs/research/notes/`: capture, noise and VAD, ASR models and hotwords, other dictation apps, correction and
     evaluation

**How the owner likes to work** (learned while working together):
- Explain in plain words, big picture first, with a table of results. Say what the owner has to do next as numbered
  steps.
- **Measure first.** An accuracy change is kept only if `sst eval` shows it better on the held-out sets (C-E), with
  the 95% range. Tune on sets A-B only.
- Product and privacy defaults are the owner's decision (e.g. the warm microphone: 5 minutes). Ask with 2-4 clear
  options and a recommendation.
- The owner reviews and merges. When the owner explicitly says to merge or release ("you do", "yes release it"), do
  it: merge stacked PRs in order, retargeting the next one to `main` first. Close issues by hand if a retargeted PR
  didn't close them.
- Long jobs (scoring, sweeps) run in the background; tell the owner what is running and why.

**Not in git, on purpose** (it stays on the laptop that made it):
- `models/`: run `scripts/download_model.py parakeet`. It also fetches `bpe.vocab`, which hotwords need.
- **The owner's voice.** Reading tests are in `%LOCALAPPDATA%\sst\bench` (150 recordings, the baseline for every
  measurement) and dictations in `%LOCALAPPDATA%\sst\recordings`. Settings, Your words, history and stats are in
  `%APPDATA%\sst`. On 2026-10-01 they were packed into `OneDrive\Documents\Rflow-data-2026-10-01.zip` (17 MB, synced by the
  owner's OneDrive) on the first laptop, for the owner to carry over (OneDrive or USB, **never GitHub**). Unzip `bench\` into `%LOCALAPPDATA%\sst\bench\` and
  `settings\` into `%APPDATA%\sst\`. Without them, `sst eval` has nothing to score until the owner reads new sets.
  `asr_cache.json` in each test is only a cache; the key in `gateway.json` (none was set) wouldn't decrypt on another
  laptop anyway (DPAPI).
- The installed app: `Rflow-Setup.exe` from the latest release.

**Gotchas on the Windows dev laptop:**
- **Git Bash heredocs broke** on long Python snippets with quotes and backslashes: edits failed half-way, and once
  wrote backspace characters into a regex. Write the script to a file, or use the editor, instead.
- **The console is cp1252.** `sst eval` now prints `?` for characters it can't show, rather than failing.
- **The first HTTPS connection sometimes stalls** (GitHub, Hugging Face). Use retries: curl `--retry`, or the
  scripts' own retries.
- **Only one Rflow can dictate.** The developer copy (`uv run sst app`) and the installed `Rflow.exe` share the same
  mutex and data folders. Quit Wispr Flow too: it also listens to Ctrl+Win.
- **The developer copy's "Update now" only opens the release page** (by design: it's updated with `git pull`). That
  confused the owner; see Next steps.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Record and transcribe locally: CLI (`sst start`, `sst file`) and web page (`sst web`) | done, on `main` |
| 1 | Dictate into any app with a global hotkey (`sst dictate`) | done, on `main` (PR #2) |
| setup | GitHub CI (lint, tests, CodeQL), Dependabot, templates, CLAUDE.md, this file | done, on `main` (PR #4) |
| 2 | Windows installer (no Python needed), CI app build, release pipeline | done, on `main` (PR #6) |
| 3 | Wispr-style hotkeys: hold Ctrl+Win, Ctrl+Win+Space hands-free, optional Menu key | done, on `main` (PR #8) |
| 4 | A real app: tray icon, recording pill, settings, history, logs (Qt) | done, on `main` (PR #10) |
| 5 | AI text cleanup (model chosen in Settings), key encrypted; one-piece transcription up to 3 min | done, on `main` (PR #12) |
| 6 | **Rflow 1.0**: product name, generic endpoint / key / model, in-app updates, download website | done, on `main` (PR #14), released **v1.0.0** |
| fix | Update checks retry when the first connection stalls | done, on `main` (PR #16), released **v1.0.1** |
| 7 | **Reading test**: accuracy on the user's own voice, per cleanup model; misheard words → Your words | done, on `main` (PR #18), released **v1.1.0** |
| 8 | **The Rflow window**: a complete app like Wispr Flow (Home, Dictionary, Reading test, AI cleanup, Settings), first-run welcome, branded installer | done, on `main` (PR #20) |
| 9 | **AI providers and profiles**: OpenAI, Anthropic, Gemini, Groq, Ollama, vLLM; one setup per person; 1.3.0 | done, on `main` (PR #22), released **v1.3.0** |
| 10 | **Accuracy lab**: five sets of sentences, session notes, audio measurements, `sst eval` with 95% ranges | done, on `main` (PR #24); the owner read all five sets |
| fix | Fairer scoring (contractions, compounds, Ctrl), only names suggested, eval report printing | done, on `main` (PR #25) |
| 11 | **Capture**: WASAPI, warm microphone with lead-in and tail, raw mode, Bluetooth warning, peak to -1 dBFS, retry of empty results | done, on `main` (PR #27) |
| 12 | **Hotwords**: Parakeet listens for Your words (bpe.vocab from NVIDIA's archive, beam search, score 1.0, guard) | done, on `main` (PR #29); phases 10-12 released as **v1.4.0** |
| 13 | **Speech recognition as a building block**: a catalog of speech models, a per-profile choice, background switching, the Speech recognition page | done, on `main` (PR #33) |
| 14 | **Whisper large-v3 turbo on this computer** (faster-whisper), downloaded when chosen, language choice | done, on `main` (PR #35), released **v1.5.0** |
| 15 | **Scan my computer**: hardware, a benchmark, the downloaded models timed, a verdict per model | done, on `main` (PR #37), released **v1.5.0** |
| 16 | **Cloud speech models**: OpenAI, Groq, Google Gemini with the user's key, a warning, a Test, Parakeet as the fallback | done, on `main` (PR #39), released **v1.5.0** |
| 17 | **Your own server for speech**: vLLM, the company gateway, any OpenAI-compatible transcription server; Load models, Test | done, on `main` (PR #41), released **v1.5.0** |
| 18 | **Parakeet downloaded on demand**: a speech step in the welcome, the installer 90 MB instead of 571 MB; version 1.5.0 | done, on `main` (PR #43), released **v1.5.0** |
| logo | **The owner's logo** everywhere (`docs/brand/rflow-logo.webp` → `scripts/make_brand.py`): app icon in 10 sizes, the window's mark, installer pictures, website | done, on `main`, released **v2.0.1** |
| 25 | **Rflow UI 2.0, "Obsidian Signal"** (the owner's request): the Figma design implemented in Qt: soft depth painted by `sst/theme.py`, widgets in `sst/ui.py`, five sections, a three-step first run, Home with orb, stats strip and search, the pill and popups in the popup look (Translate: C copies, Enter replaces), a rail at the smallest size, Geist shipped; the website redone | done, on `main` (PR #64), released **v2.0.0** |
| fix | **Popups that take clicks**: Translate's popup and the Text Transform menu had the pill's click-through style (`WS_EX_TRANSPARENT`): `_no_activate(hwnd, click_through=False)` now; checked by hand with `scripts/check_popup_clicks.py` (15 of 15) | done, on `main`, released **v1.10.1** |
| 24 | **The Translate popup, redone** (the owner: "the UX literally sucks"): "Japanese → English" with one-click languages and More, as tall as the text (nothing overlaps) and always on the screen, errors in plain words with Try again, no model leads to AI cleanup, Copy says "Copied", a click outside closes it, never English into English | done, on `main`, released **v1.10.0** |
| 23 | **Settings that change only on purpose** (the owner's request): the wheel never changes a dropdown, long lists searchable, a Save per section with its state shown, API keys masked with a pen to change them, the model in use and the language at the top of Speech recognition | done, on `main` (PR #58), released **v1.9.0** |
| fix | A dictation typed twice when its last part replaced the one before: the result read the replacement too early | done, on `main` (PR #57), released **v1.9.0** |
| 22 | **Translate** (the owner's idea, like DeepL): select text, Ctrl+C+C, a popup at the pointer shows it translated by the AI model, language at the top, Copy or Replace | done, on `main` (PR #54), released **v1.8.0** |
| 21 | **Snippets** (the owner's idea): say "my email" and your email is typed; your own phrases and text (several lines), alone or inside a sentence, never sent to the AI | done, on `main` (PR #52), released **v1.8.0** |
| 20 | **Text Transform** (the owner's idea): say "make it concise" (or double-tap Ctrl for a menu) and the selected text or the last dictation becomes Concise, Professional, Bullet points or Action items, checked, with undo; the text is found again if focus moved | done, on `main` (PR #50, with #48), released **v1.7.0** |
| 19 | **The voice pipeline** (the owner's plan): always-on mic, chunks while speaking, parallel ASR, merge, dictionary, formatting, guarded LLM | done, on `main` (PR #46), released **v1.6.0** |
| 26 | **Microphones that follow you**: Windows' own device list, open microphones reopened when devices change, silent recordings caught | done, on `main` (PR #70), released **v2.0.2** |
| 30 | **Never lose work** (the user testing's blockers): a key brushed while dictating keeps the dictation; a damaged dictionary or settings file never stops Rflow (moved aside, restored from a copy); crash reports; a failed start says why; a failed paste keeps the text; updates wait for idle; the keyboard hook is renewed | done, on `main` (PR #78), released **v2.3.0** |
| 29 | **Live translation spoken aloud** (the owner's request): Piper's Danny on the laptop (sherpa-onnx, no new dependency), each sentence spoken as soon as it's whole, faster when behind; process loopback leaves Rflow's own voice out of what's captured; the microphone pauses while it plays through speakers | done, on `main` (PR #76), released **v2.3.0** |
| open source | **MIT License** and the files contributors need: CONTRIBUTING, SECURITY, code of conduct, a feature template, complete notices, a "Source code" row | done, on `main` (PR #80), released **v2.3.0** |
| 31 | **Clearer sections, setups and costs** (the owner's UI review): ten sections, Live's big Start/Stop, Your API keys, the speech model in two groups, setups with monthly costs, Start over, retired models fixed | done, on `main` (issues #82-#88), released **v2.3.0** |
| 28 | **Live translation, a section of its own** (the owner's redesign): a sidebar section, Ctrl+Alt+L, a movable, resizable, scrollable bar with ✕, and the source: Computer, Microphone or Both (your words marked "You") | done, on `main` (PR #74), released **v2.2.0** |
| 27 | **Live captions, part 1** (the owner's idea): what the laptop plays → WASAPI loopback → Gemini 3.5 Live Translate → a caption bar left out of screen shares, and a transcript; a separate pipeline (`sst/live/`) | done, on `main` (PR #72), released **v2.1.0** |

Released: v1.0.0, v1.0.1, v1.1.0, v1.3.0, v1.4.0, v1.5.0, v1.6.0, v1.7.0, v1.8.0, v1.9.0, v1.10.0, v1.10.1, v2.0.0, v2.0.1, v2.0.2, v2.1.0, v2.2.0 and v2.3.0 (GitHub Releases; there is no 1.2.0; the updater compares versions as numbers, so 1.10.0 is newer than 1.9.0). Website: https://rachflow.vercel.app (Vercel project `rach_darling_flow-site`, team karthi-labs; the address was
added on 2026-10-01, and the old https://rachdarlingflow-site.vercel.app stays assigned: installed apps up to 1.6.0
link there, so never remove it;
`site/`). The in-app update path is verified end to end: the owner's installed 1.0.0 showed the banner and updated
itself to 1.0.1.

## Continue on another device

```
git clone https://github.com/karthi-ai-engineer/rflow-ai.git
cd rflow-ai
git config user.name "Karthi27"
git config user.email "karthi.ai.engineer@gmail.com"
gh auth login                                     # as karthi-ai-engineer
uv sync
uv run python scripts/download_model.py parakeet  # ~630 MB + bpe.vocab (10 KB), not in git
uv run pytest                                      # 243 tests
uv run sst app                                     # the developer copy (quit the installed Rflow first)
uv run sst eval --no-cleanup                       # score the reading tests (needs the owner's recordings, above)
```

Then read `CLAUDE.md` (workflow and rules) and pick up at **Next steps** below. Ollama is installed on the first
laptop (useful for phase 13's local AI cleanup); check with `ollama list` on the new one.

## Decisions so far

- **Engine:** Parakeet 0.6B "unified" English, int8, on the CPU via sherpa-onnx 1.13.8. It loads in about 2 s, and a
  3 s clip takes about 0.3 s. `sherpa-onnx-core` has to be listed explicitly, or Windows loads the older
  `System32\onnxruntime.dll` and the model fails.
- **Hotkey Ctrl+Win (phase 3), as in Wispr Flow.** Phase 1 used Ctrl+Alt+D; the owner's muscle memory is Ctrl+Win, and
  Ctrl+Win+D opened new virtual desktops. Modifier-only keys need a low-level keyboard hook (`sst/hotkey.py`), not
  RegisterHotKey:
  - The hook thread only matches keys; the logic is the pure `Matcher` class, which is fully unit-tested.
  - sherpa-onnx releases the GIL while decoding (another Python thread waited at most 21 ms during a 3 s decode), so
    typing never lags during a transcription.
  - Keys sst sends itself carry `dwExtraInfo = OUR_INPUT`, and the hook ignores them. Keys injected by other tools
    (PowerToys remaps) are treated like real ones.
  - A vkE8 "mask" key is sent on press, so releasing Win doesn't open Start (the same trick AutoHotkey uses).
  - Another key during Ctrl+Win means a Windows shortcut such as Ctrl+Win+D or +←/→, so the recording is dropped
    ("interrupt"). Space means hands-free and is hidden from Windows. The key-state tracking is re-checked with
    GetAsyncKeyState, because the hook misses releases on the lock screen.
  - Only one dictation can run (mutex `SST-Dictation-dictate`), and there's a warning if Wispr Flow is running.
- **The dev laptop's PowerToys Keyboard Manager** remaps Menu (0x5D) to Ctrl+Win and the Copilot key (Win+Shift+F23)
  to Ctrl+Win+Space, set up for Wispr Flow. With the default hotkey, both keys therefore work for sst as they are.
  Don't use `--hotkey menu` there: which hook sees the key first depends on start order.
- **Tap = hands-free, hold = push-to-talk** on the same key, split at 0.4 s. Esc cancels, and Esc is only taken from
  other apps while recording.
- **Typing = clipboard + Ctrl+V.** The old clipboard is restored afterwards (every memory-block format) and dictated
  text is kept out of Win+V history. Pasting was chosen over simulated typing because it is instant and editors don't
  auto-close brackets partway through.
- **Long audio is split at pauses into pieces of up to 30 s.** Parakeet crashed in onnxruntime on a 514 s recording.
- **Recordings stop by themselves after 3 minutes** (the owner's choice). 3 minutes transcribes in about 23 s.
- **Recordings are kept** as `.wav` + `.txt` in `recordings/` (git-ignored) to compare engines later. `--no-save` turns
  this off.

## Packaging (phase 2)

- `build_installer.cmd` runs `scripts/build_installer.py` in its own environment, `build\venv`, so it works while the
  dev copy is running. The steps are PyInstaller (one folder, `packaging/sst.spec`), then the model into
  `dist/sst/models` (hard links), then a smoke test (`dist/sst/sst.exe file` must transcribe the model's test WAV),
  then Inno Setup (`packaging/installer.iss`). It takes about 95 s and produces a 500 MB `Setup.exe`
  (710 MB installed). Since phase 18 the model is taken out before Inno Setup: see **Parakeet downloaded on demand**.
- Installs per user into `%LOCALAPPDATA%\Programs\SST Dictation`, so no admin is needed. There is a Start menu
  shortcut, plus optional desktop and start-at-sign-in shortcuts. The model is inside, so it works offline.
- The installed app keeps recordings in `%LOCALAPPDATA%\sst\recordings` and its model next to `sst.exe`
  (see `sst/__init__.py`). Opening it with no arguments starts dictation (`packaging/sst_app.py`).
- sherpa-onnx's `onnxruntime.dll` must stay in `_internal/sherpa_onnx/lib/`. Verified: the installed app loads it
  from there, not the older copy in System32.
- The only DLL the bundle needs from Windows is `propsys.dll`, so no Microsoft C++ runtime install is needed.
- While running, the app holds the mutex `SST-Dictation-running`. Setup and uninstall use it to ask the user to close the app.
- Verified locally: silent install, install over an existing copy, the installed `sst.exe` transcribing, dictation
  mode, and uninstall (nothing left behind).
- The installer is not code-signed, so SmartScreen shows "Windows protected your PC" (click More info, then Run anyway).
  A certificate would fix that.
- Version: `__version__` in `sst/__init__.py` is the only place to change it. A tag `vX.Y.Z` must match it, or the
  Release workflow stops.
- CI: the `Build app (Windows)` job builds `sst.exe` on every PR and runs the smoke test (the model is cached in Actions).
  The Release workflow (tag `v*`, or started by hand) builds `Setup.exe` and publishes it as a GitHub Release (a manual
  run only uploads it as an artifact).
- Inno Setup on the dev laptop is a portable copy in `.tools/innosetup` (git-ignored).

## The tray app (phase 4)

- `sst/app.py` (Qt / PySide6-Essentials, LGPL, bundled as separate DLLs). It has a tray icon and menu (status,
  History, Settings, logs folder, Quit), and the icon turns red while recording.
- The pill is a frameless, topmost, click-through window with `WS_EX_NOACTIVATE`. Checked: a focused window stays
  focused through every pill state, so the paste still goes to the user's app.
- The model loads on a background thread, and the tray icon appears at once. The keyboard hook starts after the model
  has loaded. A 15 ms QTimer pumps the hook events into `Dictation`, and the worker thread reports back through a Qt
  signal.
- `sst/dictate.py` `Dictation` is the one dictation engine, driven by the tray app and by the console command
  `sst dictate`. Its tests feed events with explicit times, so they are exact and fast (no sleeping).
- Settings (`%APPDATA%\sst\settings.json`) are the hotkey, microphone by name, sounds and saving recordings. A damaged
  file falls back to the defaults. History is `%APPDATA%\sst\history.jsonl` (last 200). Logs are
  `%LOCALAPPDATA%\sst\logs\sst.log` (rotating).
- The microphone list is re-read before every recording (~45 ms), so a headset plugged in later, or a new Windows
  default, is picked up. The dev laptop's default mic was Bluetooth earbuds (OnePlus Nord Buds 3r). Bluetooth headset
  mics use a low-quality call mode, so pick the laptop mic in Settings for better accuracy.
- The installer has two programs: `SST Dictation.exe` (tray app, no console; the shortcuts point here) and `sst.exe`
  (CLI).
- "Start with Windows" is on by default. The installer task and the in-app setting share one HKCU Run value with
  `--startup`, which starts quietly. Uninstall removes that value even if it was switched on from the app (verified
  with a throwaway installer).
- Build checks: `SST Dictation.exe --self-test` builds every window off-screen and transcribes once, and the build
  script runs it. The dependency scan of all 92 bundled binaries finds nothing missing on a plain Windows (Qt brings
  its own C++ runtime).
- The installer is 520 MB (Qt added 20 MB), 762 MB installed.
- The owner installed phase 4 on 2026-09-30 and dictated with it (the tray app, pill and History work).

## Accuracy on the owner's voice (first check, 2026-09-30)

- The owner read the "What I need from you" list aloud: 76.5 s, one hands-free dictation in the installed app.
  Compared with that text (148 words):

  | Decoding | Word errors |
  |---|---|
  | as dictated (greedy, cut into 30 s pieces) | 28% |
  | one piece | 24% |
  | beam search | no better |

  Leaving out the owner's own reading changes (skipped or added words), about 13–17% of words are wrong. This model
  scores about 6% on standard English benchmarks.
- Typical errors are tech words and names: commit → clot, "the five PRs" → "a file PR", Claude → cloud, Tamil →
  dropped / "Tamar", polishing → publishing, "tech words" → "that was". One whole sentence was dropped when cut at
  30 s, and decoded correctly as one piece.
- **Next fix:** raise `MAX_PIECE_SECONDS` in `sst/engines/parakeet.py`. 257 s in one piece works; it crashed at 514 s.
- Microphone: none is chosen, so the Windows default is used, which is the Bluetooth earbuds when they are connected.
  2 of that day's 6 recordings were phone quality (no sound above 4 kHz). The long reading was full band, so most
  errors are the model on this voice, not the mic.
- The analysis scripts were ad hoc. Phase 7 replaced them with the reading test (see below).

## Company AI gateway (tested 2026-09-30)

- The owner's company runs an internal, OpenAI-compatible AI gateway (models, chat/completions, responses,
  completions, embeddings, images/generations, audio/transcriptions). Its address is in the owner's own settings
  (`%APPDATA%\sst\gateway.json`), deliberately not in this public repository.
- Auth is the owner's gateway API key, sent as `Authorization: Bearer <key>` or `X-API-Key: <key>`. The key lives only
  in `%APPDATA%\sst\gateway.json`, outside the repository and encrypted with DPAPI (see phase 5). **Never commit it.**
- Task:
  - list the models
  - test only the models hosted on the company's own GPU server, **one at a time** (a small server, with cold starts)
  - measure transcription accuracy and speed on the owner's recordings, and text polish quality and speed
  - pick the best and fastest, then use it in sst
- **Results (2026-09-30).** 24 models: 17 cloud (OpenAI, skipped as asked) and 7 local (`is_cloud: false`), tested one
  at a time:

  | Local model | Speed | First word | Polish (word error 23.6% before) | Transcription |
  |---|---|---|---|---|
  | Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 | **43 tok/s** | 0.14–0.24 s | 22.9% (fixes "commit"; rewords a bit) | not supported |
  | unsloth/Qwen3.8-27B-NVFP4 | 28 tok/s | 0.24–0.35 s | 23.6%; short sentence 20% → **10%** (most faithful) | not listed |
  | Qwen/Qwen3.6-35B-A3B-FP8 | 20 tok/s | 1.3 s | 24.3% | HTTP 500 |
  | Qwen/Qwen-AgentWorld-35B-A3B | 26 tok/s | 0.26 s | 24.3%, identical output: same model as 3.6 | HTTP 500 |
  | Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf | 17.6 tok/s | 0.29 s | 25.0% (kept "clot") | not listed |
  | qwen3, sensenova-u1.5 | – | – | "Backend unavailable" (switched off; retried after 90 s) | – |

- **No local model transcribes speech.** Only the cloud `whisper-1` does. Keep Parakeet on the laptop for
  speech-to-text. vLLM can serve Whisper, so IT could add `whisper-large-v3-turbo` locally; worth asking.
- **Polish:**
  - With the owner's word list in the prompt, the LLMs fix names that sound like a vocabulary word ("STD" → "SST",
    "hashtag 2" → "#2", "clot" → "commit").
  - They can't bring back words the recognizer dropped, and they don't fix "cloud" → "Claude" when "cloud" also
    makes sense.
  - A typical one-sentence dictation polishes in 0.5–0.7 s; the 76 s reading takes 4–5 s.
- **Connection quirk:** from Python (not curl) the first connection to the gateway sometimes stalls. Use a 4 s
  connect timeout with retries, keep the connection alive, and warm it up at start. `trust_env=False`, since it's
  internal.
- The test scripts were ad hoc (httpx, one model per run). The reading test (phase 7) now compares the cleanup models
  on the owner's own recordings, one model at a time.

## Text cleanup (phase 5)

- (Phase 6 made this generic: see below. The owner uses these two models as model and backup.)
- The owner's decision: in Settings, choose **1. `unsloth/Qwen3.8-27B-NVFP4`** (best quality) or **2.
  `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`** (fastest), or Off. Saving activates it from the next dictation. Only the
  model changes; everything else stays the same.
- `sst/gateway.py`, standard library only (http.client, no proxy):
  - `Polisher.polish(text)` never raises. Each dictation gets 2 s plus 0.04 s per word to answer.
  - The other model is tried after an HTTP error. A timeout gives the heard text; there's no second wait.
  - An unreachable gateway (e.g. at home) is skipped for 60 s.
  - An implausible answer (much shorter or longer) is not used.
  - `<think>` tags and wrapping quotes are removed.
  - Connects with a 1.5 s timeout and 3 tries (the first-connect stall), keeps the connection alive, and
    `prepare()` connects when recording starts.
- `Dictation.cleanup` is swapped by the app on Save. The state `typed_raw` shows an amber pill, "Typed as heard",
  plus at most one tray notification per 10 minutes. History stores `heard` next to `text`.
- The key lives in `%APPDATA%\sst\gateway.json` (Settings edits it; masked field; "Test" button). It is never in git
  or the logs (`GatewayConfig.__repr__` masks it).
- On disk it is `api_key_protected`, encrypted with Windows DPAPI for the signed-in user: only that user on this laptop
  can decrypt it. CodeQL flagged the first version, which stored it as plain text. A plain `"api_key"` typed into the
  file by hand is encrypted on first load. A key that can't be decrypted (copied from another laptop) is dropped,
  and the user is asked to enter it again.
- Settings has a Settings button in the History window too; a tray click opens History, and the owner looked
  for the options there. `sst.exe` without arguments starts the tray app, because an old taskbar pin to `sst.exe`
  opened a console that said "already running".
- The system prompt: fix recognition errors using the user's vocabulary, add punctuation and capitals, remove
  fillers, keep the wording, don't answer. Settings' "Your words" feed the vocabulary.
- Against the real gateway:

  | | First cleanup | Next dictation | Connect |
  |---|---|---|---|
  | Model 1 | 1.12 s | 0.64 s | 1.6 s, while speaking (the stall happened and was hidden) |
  | Model 2 | 0.67 s | 0.35 s | 0.05 s |
  | Fallback (switched-off `qwen3` chosen) | 1.17 s via model 1 | | |

  "STD dictation" → "SST Dictation", "pr's" → "PRs", and fillers were removed.
- `MAX_PIECE_SECONDS` is 180: a whole dictation (max 3 min) is transcribed in one piece.
- Tests: 94. `tests/test_gateway.py` runs a fake gateway on localhost (good, error, slow, unreachable, implausible
  answers, keep-alive, fallback).

## Rflow 1.0 (phase 6)

- **Name.** Rflow is the tray app `Rflow.exe`, and the command-line tool is `rflow-cli.exe` (Windows ignores case, so
  the two names must differ). The installer is `Rflow-Setup-<ver>.exe`. The installer keeps the same AppId, so SST
  Dictation installs upgrade in place: their folder and their Settings > Apps entry stay. `[InstallDelete]` removes the
  old `SST Dictation.exe` / `sst.exe` and the old shortcuts, and `[Registry]` deletes the old "SST Dictation" Run value.
  The mutexes keep their old names (`SST-Dictation-running`, `SST-Dictation-dictate`), so new installers still detect a
  running old version. Data folders stay `%APPDATA%\sst` and `%LOCALAPPDATA%\sst`.
- **Generic cleanup.** The code has no company defaults. Settings > "Text cleanup with an AI model" has:
  - an on/off switch
  - Endpoint and API key (optional, e.g. for Ollama; no empty Bearer header is sent)
  - "Load models" (`Polisher.models()`: GET /models; models marked `is_cloud: false` are listed first)
  - Model and Backup model (editable combos: pick one or type a name), and Test

  `Settings.cleanup` / `cleanup_model` / `cleanup_fallback`: an old settings file that has a model but no `cleanup`
  key loads with cleanup on. The owner's backup model has to be picked once in Settings, since the fixed pair of models
  is gone.
- **In-app updates** (`sst/updates.py`):
  - The installed app checks `api.github.com/.../releases/latest` 20 s after start and then every 6 h; the tray menu
    has "Check for updates". A newer tag with the assets `Rflow-Setup.exe` and `Rflow-Setup.exe.sha256` shows a blue
    banner in the window (What's new / Update now), a tray notification (clicking it opens the window), and a menu item.
  - The installer is downloaded to `%TEMP%\Rflow-update`. Only GitHub hosts are accepted after redirects, and the
    SHA-256 must match, or the file is deleted.
  - Then the app releases its `SST-Dictation-running` mutex (otherwise setup stops at AppMutex) and starts setup with
    `/SILENT /SUPPRESSMSGBOXES /NORESTART /UPDATE=1`, then quits. setup's `[Run]` entry with `Check: IsInAppUpdate`
    starts the new Rflow.
  - A source checkout doesn't check for updates; "Update now" there only opens the release page.
- **Release workflow:** it copies the build to `dist/release/Rflow-Setup.exe`, writes `Rflow-Setup.exe.sha256`
  (sha256sum), and publishes both to the GitHub Release "Rflow vX.Y.Z" (the tag must equal `sst.__version__`).
  `releases/latest/download/Rflow-Setup.exe` is the stable download link.
- **Website:** `site/index.html` is a single static page with screenshots in `site/img` (rendered from the real
  widgets, generic example settings) and `favicon.ico`. It works in dark and light mode, and at phone width (checked:
  no horizontal overflow at 390 px). The download button links to the stable link, and a script shows the latest
  version, size and date from the GitHub API. Vercel: import the repo with **Root Directory `site`**, Framework
  "Other", and no build command.
- **Public repo:** the company gateway's hostname and IP were removed from the code and the docs. Older commits in the
  git history still contain the hostname; removing it would mean rewriting published history. That is the owner's
  decision, and so is whether the repo stays public.
- The Settings screenshot exposed a layout bug: a long word-wrapped QLabel inside a QFormLayout overlapped and got
  clipped. The text is now one short line, with the details in a tooltip.

## Reading test (phase 7)

- Tray menu → "Reading test..." opens `ReadingTest` (`sst/app.py`). It shows one of the 30 sentences in `sst/bench.py`
  (everyday dictation with the owner's kind of words: GitHub, pull request, merge commit, branch, Rflow, Parakeet,
  Vercel, Tamil, Japanese, Teams, CodeQL, Karthi, numbers). It has Record / Stop (Space too), Redo, Back / Next and a
  live level. It records with the microphone chosen in Settings, and rejects recordings under 0.5 s.
- Each recording is saved as `NN.wav` + `NN.txt` (the sentence) in `%LOCALAPPDATA%\sst\bench\<date_time>`. Closing the
  window halfway loses nothing: the next "Reading test..." continues the newest folder while it has sentences left
  (`bench.unfinished()`).
- **Score** (`bench.score()`, on a thread):
  - Transcribes every recording with Parakeet first, then cleans up all the texts with the cleanup model, then with the
    backup model, one request at a time and with no fallback, so each model is measured on its own.
  - Word error rate = (substituted + missing + extra words) / words in the sentence. `bench.words()` makes the
    comparison fair: case, punctuation and hyphens don't count, "70" equals "seventy", ok = okay.
  - The alignment is Levenshtein; among equally short alignments it pairs words that look alike, so "Tamil → Tamar" is
    listed, not "Tamil → please".
  - Writes `results.json` and `report.md` next to the recordings.
- The results page shows a table per setup (errors, wrong / total, seconds per sentence; the best in bold) and the most
  misheard words. Suggested words (misheard, not common words, spelled as in the sentence) can be ticked and added to
  Your words, which activates them at once, and then **Score again** shows the difference.
- `sst bench <folder> [--model <name>]...` (`rflow-cli bench` when installed) re-scores a folder from the command line,
  by default with the models in Settings.
- `ParakeetEngine.transcribe` has a lock now: dictating while the test scores must not use one recognizer from two
  threads.
- The installed app deletes `%TEMP%\Rflow-update` (the downloaded installer, 500 MB) 60 s after it starts.
- A dry run on the model's test WAV against the gateway (both models, one at a time): 0% errors for every setup;
  Qwen3.8-27B 2.75 s (cold), Qwen3-30B 0.90 s per sentence. The owner's real run is still to come.
- Not done in this phase: trimming silence (VAD) and a Whisper comparison. The owner's reading test results should
  decide whether they're worth it.

## The Rflow window (phase 8)

- The owner's request: Rflow should open as a complete application, like Wispr Flow, so that people who download it
  from the website or a link can set it up and change everything inside the app.
- `sst/window.py`, `MainWindow`: a sidebar (Home, Dictionary, Reading test, AI cleanup, Settings; the status and the
  version at the bottom) and a page stack. It is native Qt with one stylesheet (`stylesheet(theme)`, the website's
  colours) that follows Windows' light/dark mode (`styleHints().colorSchemeChanged`). An embedded browser
  (QtWebEngine) would have added ~150 MB and a slower start for the same look. The icons come from Windows' own icon
  font (Segoe Fluent Icons, or MDL2 Assets on Windows 10). The checkbox tick and the dropdown arrows are small PNGs in
  `sst/static/ui`, drawn by `scripts/make_ui_images.py`, because a Qt stylesheet can only show images from files.
- The window keeps no state. It calls the app: `TrayApp` in `sst/app.py` (`apply_settings`, `save_cleanup`,
  `add_words`, `remove_word`, `score_reading`, `finish_welcome`, `window_closed`, updates), or `PreviewApp` in the
  self-test, the tests and `scripts/make_site_screenshots.py`. Settings apply at once; AI cleanup has a Save button,
  because a half-typed endpoint or key shouldn't be used.
- **Home:**
  - a greeting and how to dictate with the chosen key
  - the stats: words this week and in total, words per minute, day streak. `Stats` in `sst/settings.py`, file
    `%APPDATA%\sst\stats.json`, updated on every dictation; `Dictation.on_result` now also gives the recording's
    length. The first time, it starts from the history, which has no lengths, so words per minute is shown after half
    a minute of new dictation.
  - the recent dictations grouped by day, with copy buttons
- **Dictionary:** Your words, with add (comma-separated) and remove. It says so when AI cleanup is off.
- **Reading test:** the phase 7 test, now a page (`ReadingTestPage`), with a "New test" button. Leaving the page stops a
  recording. Space records only while the test has the focus (`WidgetWithChildrenShortcut`).
- **Settings:** key, microphone with a live level (`MicrophoneBox` + `audio.LevelMeter`), beeps, keep recordings, start
  with Windows, check for updates, logs, website, report a problem.
- **Welcome** (first run, `Settings.welcomed` false): step 1 the microphone with its level; step 2 try a dictation in a
  text box (the real dictation pastes into it); step 3 optional AI cleanup. Existing users see it once too, since
  their settings file has no `welcomed` yet.
- **Opening Rflow again** (Start menu, desktop) shows the running window: the running app listens on the local pipe
  `Rflow-window-<user>` (`QLocalServer`), and a second start (`show_running_window()`) asks it to open and exits.
  `--startup` (sign-in) starts with only the tray icon. Closing the window hides it; the first time, a notification
  says that Rflow keeps running (`Settings.told_about_tray`).
- **Audio fix:** `Recorder.start()` re-reads the device list by restarting PortAudio, which closes every open stream. With
  a level meter open, a dictation starting would have pulled the rug from under it (and the reading test's recorder
  had the same risk). `audio._open_streams` now counts open streams, and the list is only re-read when none is open.
- **Installer:** Rflow-branded wizard pictures in `packaging/images` (`scripts/make_installer_images.py`, BMP at
  100–200% scaling), the welcome page switched on, and texts that say what Rflow does and that it opens after Finish.
  Only options every Inno Setup 6 knows are used: the CI build machine's copy is preinstalled, and its version isn't
  pinned.
- **Website:** `site/img/app.png` (Home), `settings.png` (AI cleanup) and the new `welcome.png` are rendered from the
  real window by `scripts/make_site_screenshots.py`, with generic example data. There is a new section, "A real app,
  not just a tray icon".
- Tests: 156. `tests/test_window.py` builds every page with `PreviewApp`. `tests/test_app.py` runs the real `TrayApp`
  with the settings in memory, a fake model and a fake keyboard hook: loading, a dictation shown on Home, a key
  change applied at once, words added, close to tray, and a second start showing the window.
- Checked visually: every page rendered with Windows fonts, light and dark, at 1000×700 and at the minimum 780×540. That
  caught history cards drawn over the page (removed widgets are only deleted later, so `clear()` hides them first),
  checkboxes without a box, and native-looking dropdown arrows.

## AI providers and profiles (phase 9)

- The owner's request: people use different AI services (OpenAI, Anthropic, Gemini, Groq, Ollama, vLLM...), so the AI
  cleanup offers them by name. Also profiles (e.g. Karthi, Rahul), each person with their own setup.
- **Providers** (`sst/gateway.py`, `PROVIDERS`): OpenAI, Anthropic, Google Gemini, Groq, Ollama (on this computer), and
  vLLM or another OpenAI-compatible server. Each entry says its usual address, which API it speaks, whether it needs a
  key, whether it's the user's own server (then the address can be edited), where to get a key, and a model hint. Each
  provider gets the request it understands (`Polisher._body`):
  - **Anthropic:** `/messages` with `x-api-key` and `anthropic-version`, the system prompt as `system`, and the answer
    from the `content` blocks. Its model list is `/models?limit=1000`.
  - **OpenAI:** `max_completion_tokens`, since its newest models refuse `max_tokens`. Reasoning models (o-series,
    gpt-5) get no temperature and room to think; they are usually too slow for dictation anyway. No
    `chat_template_kwargs`: OpenAI refuses fields it doesn't know.
  - **Gemini** (its OpenAI-compatible endpoint): no token limit, because its "thinking" counts against it and would
    cut the answer short. Its model ids lose their `models/` prefix.
  - **Groq:** the plain OpenAI format.
  - **Ollama and vLLM:** also `chat_template_kwargs: {enable_thinking: false}` (Qwen3 on the company gateway).
  - Error messages are read from `error.message` for every provider.
  - "Load models" leaves out models that don't write text (whisper, tts, embeddings, images, moderation...).
- **GatewayConfig** has `provider` and `others` (the other providers' address and key, encrypted too), so switching
  back and forth loses nothing. An old file with only an address gets its provider from the address
  (`provider_for`): the owner's company gateway becomes "vLLM or another OpenAI-compatible server", with the same
  address, key and model. That was checked against the real files, and a real request through the new code worked
  (Qwen3.8-27B, 2.3 s).
- The **AI cleanup page** has a provider dropdown. The address row appears only for Ollama and vLLM, and "Get a key"
  only for the cloud providers. A new user starts with OpenAI selected.
- **Profiles** (`sst/settings.py`, `Profiles` in `%APPDATA%\sst\profiles.json`): each profile has its own
  `settings.json` (key, microphone, words, cleanup model, welcome...), `gateway.json` (provider and keys),
  `history.jsonl`, `stats.json` and reading tests. The first profile (`default`) keeps the files where they were
  before profiles: nothing moves, and an older Rflow still reads them. The others live in
  `%APPDATA%\sst\profiles\<id>` and `%LOCALAPPDATA%\sst\bench\profiles\<id>`. `bench.unfinished()` only counts
  date-named test folders, so the `profiles` folder doesn't confuse it.
- **In the window:** a profile button under the logo (a menu to switch, "New profile...", "Manage profiles") and a
  *Profiles* page (rename in place, switch, delete with a question, create). The Home greeting uses the name, and the
  welcome asks for it. Switching (`TrayApp._activate_profile`) loads the other files, applies the key, microphone
  and cleanup at once, and builds the window again, since every page shows the profile's own data. The profile in
  use and the first profile can't be deleted.
- `sst bench` uses the profile in use.
- Tests: 191. Every provider's request format against the local fake, Anthropic's errors and models, the model filter,
  `provider_for`, the encrypted `others`. Profiles: files, ids, round trip, removal, damaged files. The window: each
  provider's fields, switching providers keeping keys, the profile button, the greeting, the Profiles page, the
  welcome's name. The real TrayApp: creating Rahul (welcome, empty words, own key) and switching back to the first
  profile's words and key.
- **Not tried for real:** OpenAI, Anthropic, Gemini and Groq with real keys (only against the fakes, which check the
  request format). The owner, or anyone with a key, should press Test once for each.

## Accuracy lab (phase 10)

- The owner's decision: work on accuracy before phases 10/11 of the old roadmap, and first on everything before the
  speech recogniser, so that a better model later gains even more. The research (papers, sherpa-onnx / NeMo sources,
  Handy, OpenWhispr, VoiceInk and others, measurements on the dev laptop) is summarised in `docs/accuracy.md`, with the
  target pipeline and phases 11-13. Main findings:
  - The 24% is mostly names and terms, plus the capture path, not the model.
  - Neural denoising makes modern models worse.
  - sherpa-onnx 1.13.8 can bias Parakeet towards Your words (hotwords), but it needs a `bpe.vocab`, which the model
    download doesn't include.
- This phase is only the measuring stick; dictation is unchanged. With 30 sentences (~360 words) the noise is about
  ±2.3 points, too much to see a 2-3 point change.
- **Sets** (`sst/bench.py`): `BLOCKS` A-E, 30 sentences each. A is the old list (old tests stay comparable, and old
  folders without notes count as A). A-B are for tuning, C-E the held-out test. Some sentences use cloud, clot and
  publishing literally. `TERMS` are the names and tech terms, scored apart.
- **Sessions:** each test folder gets `session.json` (set, microphone chosen, device and host API from
  `Recorder.describe()`, rate, version) with its first recording. A new test reads the set read completely the fewest
  times (`next_block`). Fillers (um, uh) no longer count as errors.
- **Audio measurements** (`audio.measure`): the speech level (loudest 5% of 20 ms frames), noise (quietest 10%), peak,
  clipped share, and the 4-7 kHz level against 0.3-3 kHz on speech.
  - Flags: narrowband below -45 dB, quiet below -45 dBFS, noisy below 15 dB SNR, clipped over 0.1%.
  - On the model's test WAV the high band is -30 dB, and -106 dB after a 300-3400 Hz filter.
- **`sst eval`** (`sst/evaluate.py`; `sst bench` is an alias; `rflow-cli eval` when installed):
  - Replays the tests (all of the profile's by default) through `Pipeline`s: recognition alone, `--degrade narrowband`
    / `gain:<dB>`, and each cleanup model.
  - Reports: word and character errors, names-and-terms errors, other-word errors, and terms put in where something else
    was said.
  - A paired bootstrap (1000 draws within each session, fixed seed) gives each setup a 95% range and a verdict against
    the first: better or worse only when the whole range is on one side of zero.
  - The tuning and test sets and each microphone are reported apart, with time p50/p95.
  - Recognition is cached in each test's `asr_cache.json`, keyed by `engine.signature` (model and decoding), the
    degradation and the WAV's hash. Phase 12 must change the signature when the decoding changes.
  - Several tests: the report goes to `bench\summary`.
- **The page:** "Set C · sentence 3 of 30", Score (this test) and Score all tests. The results show the 95% range, the
  verdict, names and terms, and plain microphone warnings. The buttons sit in two rows so the page fits the window's
  smallest size (checked in both themes).
- Checked with the real model: `sst eval --degrade narrowband --degrade gain:-30` on the model's test WAV gave 0% for
  all three, and the second run came from the cache. That is one clean clip, so nothing about the owner's voice yet.
- Tests: 211.

## The owner's baseline (after phase 10)

- All five sets read on 2026-09-30 with the laptop microphone (Intel Smart Sound array, Windows default). Set A was read
  with the old reading test (no session.json: counted as set A, microphone "unknown"). Set B was read while Bluetooth
  headphones were connected; the laptop microphone still recorded.
- With the fairer scoring: **9.2%** word errors (7.1-11.6%); test sets C-E 6.7%; names and terms 40% (41 of 102); other
  words 7.5%. Narrowband (phone quality, simulated) costs +3.4 points (+1.4 to +5.5), and names go to 50%.
- Set B lost 5 first words and 2 sentences decoded to nothing, although the speech is in them (raised 4 times, or cut
  in halves, they decode). The other sets had neither.
- About a quarter of every recording is exact digital zeros, and a quiet room measured -91 dBFS: Windows or the Intel
  driver gates and suppresses noise on this microphone.

## Capture (phase 11)

- `Recorder` (`sst/audio.py`):
  - Records through WASAPI at the device's own rate (48 kHz on the laptop; the engine resamples), mono via
    `auto_convert`. It falls back to the old MME way if WASAPI fails. Names saved by MME (cut at 31 characters) match
    the start of the WASAPI name.
  - `warm_seconds`: stays open after a recording (the app: 300 s, the owner's choice) and keeps the last
    `PREROLL_SECONDS` (0.4) before `start()`. `tick()` closes it when idle. A call-quality microphone
    (`call_quality()`: WASAPI rate of 16 kHz or less, or "Hands-Free" in the name) is never kept open.
  - `stop_later()` returns a `Take` at once; the audio thread adds the `tail` (0.3 s) and `Take.audio()` waits for it.
    `Take.seconds` leaves out the lead-in, so a quick accidental tap is still ignored.
  - `raw=True` sets `AUDCLNT_STREAMOPTIONS_RAW` through sounddevice's private `_streaminfo`, and falls back to
    Windows mode if the driver refuses. On the laptop: cold open 287 ms; warm start instant; a quiet room -91 dBFS
    processed, -60 dBFS raw.
- `audio.condition()`: DC offset removed and the peak raised to -1 dBFS before the engine. `ParakeetEngine` also
  decodes audible audio that came back empty again in two halves. Its signature says so (`|peak-1|retry`), so the
  bench doesn't reuse older cached text.
- Dictation hands the take to the worker (the hotkey thread never waits) and ticks the recorder. The reading test
  records the same way (its Stop doesn't freeze the window), closes the microphone when the page is left, and notes
  the mode. `sst eval` reports raw recordings as "<device> (raw)".
- Settings: "Keep the microphone ready for 5 minutes after dictating" (on), "Turn off Windows' voice effects for this
  microphone" (off), and an amber warning under the microphone choice for a Bluetooth headset.
- Result: the owner's 150 sentences went from 9.2% to **8.2%**, with no empty results.
- Not measured yet: whether the warm microphone and the tail remove the lost first and last words (needs new
  recordings), and raw against Windows mode.
- Tests: 235.

## Hotwords (phase 12)

- **The vocabulary:** sherpa-onnx needs the model's word pieces with scores (`bpe.vocab`) to spell Your words in them,
  and its model download doesn't have it. NVIDIA's `parakeet-unified-en-0.6b.nemo` on Hugging Face is a 2.5 GB tar;
  `scripts/download_model.py` reads its tar headers with HTTP range requests and fetches only
  `*_tokenizer.vocab` (10 KB, sentencepiece `piece<TAB>score`). It keeps it as `bpe.vocab` only if its pieces equal
  `tokens.txt` in order (1024 pieces + `<blk>`: they do). The installer takes every file of the model folder, so
  it ships. CI's model cache key is the script's hash, so it refetches.
- **`ParakeetEngine`** (`sst/engines/parakeet.py`):
  - With a matching `bpe.vocab`: `modified_beam_search`, 4 paths, `modeling_unit="bpe"`, `hotwords_score=1.0`.
    `engine.words` (Your words, set by `TrayApp._apply_cleanup` and by `sst eval`) go to each recording as
    `create_stream(hotwords="A/B")`, so a new word works at once.
  - Without it, greedy as before, with a warning in the log.
  - `runaway()`: a boosted word twice in a row means boosting went wrong, and that recording is decoded again without
    the words.
  - `signature` is a property: model, decoding, a hash of the words, and the preparation.
- **The owner's reading test** (150 sentences), words = 31 names and terms from `bench.TERMS`:
  - errors 8.2% → **7.1%**
  - test sets 6.0% → **5.4%**
  - names 41% → **24.5%**
  - other words 6.4% → 6.1%
  - names put in wrongly: 2 (1 "Claude")

  With the owner's actual Your words (Parakeet, Vercel): 8.1%. **The gain comes from the words in Your words.**
- **Strength sweep** (tuning / test / names put in wrongly):
  - 0.5: 9.8% / 5.7% / 2
  - **1.0: 9.3% / 5.4% / 2**
  - 1.5: 10.0% / 5.2% / 13
  - 2.0: 14.0% / 6.9% / 58
  - 2.5 repeats "Claude Claude Claude" and 4.0 falls apart. The guard (then comparing with greedy) made no difference
    at 1.0.
- **Speed:** 161 s of speech in 14.8 s (greedy 13.7 s); the sample WAV in 0.46 s (0.34 s). Beam search without words:
  8.5% against greedy's 8.2% (noise).
- The Dictionary page says that speech recognition listens for Your words, and to add names and terms, not everyday
  words.
- `sst eval --words "A,B"` tries other words, and `--no-words` uses none.
- Tests: 243.

## Speech recognition as a building block (phase 13)

- **The owner's plan (2026-10-01):** Rflow should be building blocks. Parakeet is the base, but people want their own
  speech model, just as they choose their own AI cleanup model. The four steps:

  | Phase | Step | Notes |
  |---|---|---|
  | 13 | The building block itself | this phase |
  | 14 | Whisper large-v3 turbo on this computer | downloaded when chosen; Rflow picks the best way to run it (sherpa-onnx or faster-whisper: measure both) |
  | 15 | "Scan my computer" | disk, memory, processor, graphics card, and a short speed test per model; then a suggestion per model and one-click download |
  | 16 | Cloud and server speech models | OpenAI, Gemini, Groq, vLLM, the company gateway (`whisper-1` is on it) |

  The owner's decisions:
  - Local Parakeet stays the default; cloud speech is opt-in, with a warning that the voice leaves the computer. The
    company gateway counts as the owner's own AI: no warning needed there.
  - Start with only Parakeet and Whisper turbo locally.
  - Parakeet stays inside the installer for now (on demand later).
- **The catalog** (`sst/engines/__init__.py`): `SPEECH_MODELS` lists each model's where (`WHERE`: on this computer,
  cloud, own server), languages, size and what it's good at. `ready` marks what this version can load; the others
  show as "Coming soon". `usable(key)` turns a setting into a loadable model (the default for unknown or not-ready
  keys, e.g. a setting from a newer version).
- **An engine has:**
  - `name` (its catalog key) and `title` (what reports call it: "Parakeet alone")
  - `signature`
  - an optional `words` list, and `transcribe()`
- **Per profile:** `Settings.speech_model` (default `parakeet`; old files get it).
- **Switching** (`TrayApp._load_speech` / `_on_loaded` / `_on_failed`):
  - The chosen model loads on a thread, and the status says "Loading <model>...".
  - Dictation keeps the model it has until the new one is ready, then gets it with Your words.
  - A model that's no longer wanted when it finishes loading is dropped.
  - A failed load keeps the old model and says so.
  - Switching profiles loads the other profile's model.
- **The Speech recognition page** (`SpeechPage`) sits in the sidebar between Dictionary and AI cleanup. Its tabs are
  On this computer / Cloud / Your own server. Each model has a card with an "In use" or "Use this model" button. The
  "Scan my computer" card and the Cloud and server tabs said "Coming in the next update" (filled in by phases
  15-17). The tray menu has "Speech recognition" too.
- `sst eval` (and the reading test) name the model in use ("Parakeet alone"). By default, `sst eval` uses the
  profile's own speech model; `--engine` picks another.
- Tests: 254. Among them, the real TrayApp switching to a second model in the background (Your words carried over),
  a failed switch, a model this version can't load, and profiles with different models.

## Whisper on this computer (phase 14)

- **Measured first** on the owner's 60 recordings on the second laptop:
  - the laptop: i5-1334U, 32 GB, Intel Iris Xe, no NVIDIA card
  - the sets: set A from the old capture (4 narrowband recordings) and set B from 1.4.0
  - the method: `sst.evaluate`, every model bare (no Your words)

  | Speech model | All 60 | Set B | Set A | Names | Per sentence |
  |---|---|---|---|---|---|
  | Parakeet | 20.2% | 12.5% | 27.3% | 46% | 1.2 s |
  | Whisper turbo, sherpa-onnx (greedy) | 22.5% | 13.3% | 31.1% | 29% | 5.9 s (p95 18 s) |
  | Whisper turbo, faster-whisper, beam 5, 4 threads | 18.1% | 10.7% | 24.9% | 24% | 16.3 s |
  | Whisper turbo, faster-whisper, greedy, 8 threads | **17.5%** | **10.0%** | 24.5% | 29% | 10.7 s |

  The ranges overlap, with only 60 sentences. Other programs kept the processor about 50% busy, so the times vary
  (one 7 s clip later took 26 s).
- **The owner's decision:** faster-whisper. It's the most accurate, can use an NVIDIA card, and adds about 100 MB of
  program files. Parakeet stays the default for English: Whisper always processes 30 s windows, which makes short
  dictations slow on a processor.
- **`sst/engines/whisper.py`, `WhisperEngine`:**
  - CTranslate2 int8 on the processor (`cpu_threads` up to 8); float16 on an NVIDIA card when
    `ctranslate2.get_cuda_device_count()` finds one and NVIDIA's cuBLAS/cuDNN load, else the processor
  - greedy decoding, no VAD, no timestamps; Your words as `hotwords`
  - `language`: "" = detected, or a code from `LANGUAGES`, from `Settings.speech_language`, changed without reloading
  - audio goes through `audio.condition()`, then `audio.resample()` to 16 kHz (FFT method, numpy only)
  - `signature` includes the model revision, device, language and words
- **`sst/downloads.py`:**
  - pinned files (`whisper.MODEL`: `dropbox-dash/faster-whisper-large-v3-turbo` at `0a363e9`, 5 files, 1.62 GB,
    each with its SHA-256) into `%LOCALAPPDATA%\sst\models` (`DOWNLOADS_DIR`, shared by the installed app and the
    source checkout)
  - a `.part` file resumed with HTTP Range after a broken connection; 4 attempts against the first-connection stall
  - only Hugging Face hosts after redirects (`huggingface.co`, `*.hf.co`)
  - cancel keeps the part; `complete.json` is written last
  - checked for real: a download broken off at 1.4 GB was resumed from Hugging Face and verified
  - The repo was renamed from `mobiuslabsgmbh/...`; Hugging Face redirects the old name.
- **The app:** `TrayApp.download_speech_model` / `cancel_download` / `remove_speech_model` / `set_speech_language`,
  on a thread with progress signals, one download at a time.
  - When the download finishes, the model is chosen and loads in the background (phase 13's switch).
  - A model in use or chosen can't be removed.
  - `usable()` falls back to Parakeet when the chosen model isn't downloaded (removed, or another laptop).
- **The page:** each model is a `_ModelCard`:
  - Download and use (1.6 GB), then the progress (percent and GB), Cancel, Use this model, Remove download
  - the language row ("Language it listens for") once Whisper is downloaded
  - the summary says it takes several seconds per sentence without an NVIDIA card
- **Packaging:**
  - faster-whisper, CTranslate2 (its DLLs collected), tokenizers, huggingface_hub and PyAV are bundled
  - faster-whisper's `onnxruntime` dependency (only for its VAD) is excluded, so sherpa-onnx's stays the only
    `onnxruntime.dll`
  - the self-test imports `ctranslate2` and `faster_whisper`
  - NOTICES.txt lists the new licences (FFmpeg in PyAV is LGPL, as separate DLLs)
- **Notes:**
  - faster-whisper's own `decode_audio` breaks with PyAV 15+ (`metadata_errors`); Rflow doesn't use it.
  - `huggingface_hub` stalled for 23 minutes on the first-connection stall during the experiments, which is why
    Rflow downloads with its own code.
- **Tests:** 271, and `tests/conftest.py` gives every test an empty download folder (`whisper_downloaded` fakes an
  installed model). They cover:
  - the downloader against a local fake server: download and check, resume, a damaged file, cancel and resume,
    HTTP errors not retried, allowed hosts, remove
  - the engine with a fake faster-whisper: 16 kHz, language, words, no NVIDIA libraries, signature
  - the app's download flow: done and used, cancelled, failed, remove rules
  - the card's states

## Scan my computer (phase 15)

- **`sst/scan.py`** reads the computer through Windows itself (no new dependency, no PowerShell):
  - memory (`GlobalMemoryStatusEx`)
  - free disk where models are downloaded
  - the processor's name (registry) and physical cores (`GetLogicalProcessorInformationEx`)
  - the display adapters (the display class key in the registry)
  - whether Whisper can use an NVIDIA card: `ctranslate2.get_cuda_device_count()` and `cublas64_12.dll` loading
- **A one-second benchmark:** GFLOPS of a float32 768x768 matrix multiply on all cores.
- **The reference laptop** (i5-1334U, measured 2026-10-01; `REFERENCE_SCORE`, `SECONDS`, `MEMORY_GB`):

  | | Benchmark | Seconds for the 7.4 s sample sentence | Memory once loaded |
  |---|---|---|---|
  | The laptop | 197-247 GFLOPS (others were using the processor); 228 used | | |
  | Parakeet | | 0.9-1.0 s | 1.0 GB |
  | Whisper turbo (int8, processor) | | 9-20 s; 10 used | 3.3 GB |

- **`judge()`:**
  - Measured seconds win; otherwise the reference seconds are scaled by the benchmark. Whisper with usable CUDA is
    estimated at 1 s.
  - Levels:
    - "no" when memory is below the model's need plus 2.5 GB, or the disk can't take a download
    - fast: up to 2 s
    - usable: up to 5 s
    - slow: beyond that
  - The quickest fast model is "recommended", with Parakeet first among equals.
  - Too little free memory adds "close some programs first".
- **The app:** `TrayApp.scan_computer()` runs on a thread, with progress in the card. It times each downloaded model
  on Parakeet's own test recording (`test_wavs/0.wav`):
  - the model in use as it is
  - others loaded for it if free memory allows

  The result is kept in `%APPDATA%\sst\scan.json` (the computer's, not a profile's) and shown on the next start.
- **The page:** the scan card shows "Scan my computer" / "Scan again" and "Last scan: ...". Then "This computer:
  ..." and a line per model: an icon, the level's words, the speed, and the hint "Choose it for other languages, or
  on a computer with an NVIDIA card" for a slow Whisper.
- **On the second laptop:** Parakeet recommended (0.9 s, measured), Whisper slow (19.8 s, measured).
- **Tests: 283.** They cover the rules (reference laptop, faster and slower processors, an NVIDIA card, memory and
  disk, free memory), reading this computer, the benchmark, save and load, the card's states, and the real TrayApp
  scanning with the model in use timed.

## Cloud speech models (phase 16)

- **`sst/engines/cloud.py`:** `CLOUD` lists the providers and their models (the first is the default; the box takes
  any other name):

  | Provider | Request | Models |
  |---|---|---|
  | OpenAI | multipart `POST /v1/audio/transcriptions`, `Authorization: Bearer` | gpt-4o-mini-transcribe, gpt-4o-transcribe, whisper-1 |
  | Groq | the same, at `api.groq.com/openai/v1` | whisper-large-v3-turbo, whisper-large-v3 |
  | Google Gemini | `POST /v1beta/models/<model>:generateContent`, `x-goog-api-key`, the WAV inline (base64) with an instruction to write down exactly what is said | gemini-flash-lite-latest, gemini-flash-latest, gemini-3.5-flash-lite, gemini-3.6-flash |

  - The recording goes as a 16 kHz 16-bit WAV (32 KB a second; 3 minutes is 5.8 MB, under every provider's limit).
  - Hints: the chosen language (OpenAI's `language`, or named in Gemini's instruction) and Your words (OpenAI's
    `prompt`, or in Gemini's instruction). Gemini's thinking parts are left out of the text.
  - The connection opens while the user speaks (`prepare()`), with a 1.5 s connect timeout and 3 tries (the
    first-connection stall), kept alive.
- **Never losing a dictation:** when the provider fails (HTTP error, no answer in time, unreachable), Parakeet
  transcribes the same recording on this computer, and `last_error` says why. The dictation then reports
  `typed_local`: the pill says "Typed with Parakeet (cloud unavailable)", and a notification comes at most every 10
  minutes. An unreachable provider is skipped for a minute, so being offline costs one wait. Parakeet is the one
  already in memory when the user switched from it; otherwise it loads at the first failure (a cloud-only user stays
  light).
- **Keys:** one per provider, shared with AI cleanup and encrypted with DPAPI in the profile's `gateway.json`
  (`GatewayConfig.key_for`, `with_key`, `entries`). The AI cleanup page now builds its result from the saved keys, so
  saving it never drops a key saved on the speech page, and each page shows a key changed on the other (unless one is
  being typed).
- **Settings:** `speech_model` is `openai`, `groq` or `gemini`; `speech_cloud_models` keeps each provider's model.
  `usable()` falls back to Parakeet when a cloud model has no key, so a lost key never stops Rflow.
- **The page:** the Cloud tab has a card per provider:
  - what it is, "Nothing to download", and the warning "Your voice is sent to <provider> each time you dictate"
  - the API key with "Get a key", the model, the language
  - Test: Parakeet's sample sentence through the provider, with the time and the text
  - "Use this model" asks first (the voice leaves the computer); while in use, "Save" appears when the key or model
    changes, and applies without a reload

  The status line says "speech: <provider> <model>" when the model isn't Parakeet.
- **Scoring:** `sst eval --engine openai` (or groq, gemini) and the Reading test score through the provider, without
  the Parakeet fallback, so a failure shows. A rate limit (HTTP 429) is waited out (10, 20, 30, 60 s), and the cache
  is now saved even when a run stops halfway. `sst eval --engine whisper-turbo` no longer fails on `engine.biased`.
- **Not yet tried against the real providers:** only against a local fake speaking both formats. The owner presses
  Test with their own key. (Google's OpenAI-compatible address, `.../v1beta/openai`, has no `/audio/transcriptions`:
  it answers 404, which is why Gemini gets its native `generateContent` request.)
- **Tests: 309.** They cover each provider's request (fields, headers, the WAV, the hints), errors, the fallback
  (loaded once, Your words passed on), the one-minute skip, a slow answer, the rate-limit wait, the kept connection,
  Test, keys never in `repr`, the signature, the catalog, key sharing and saving, the dictation's `typed_local`, the
  cards (the question, Save, Test), and the real TrayApp switching to a cloud model and back.

## Your own server for speech (phase 17)

- **The engine** is the cloud one (`CloudEngine`) with the provider `SERVER` (`sst/engines/cloud.py`): OpenAI's
  multipart `/audio/transcriptions` at the user's address, with `Authorization: Bearer` only when a key is set (vLLM
  often needs none). `REMOTE` = the cloud providers plus the server: everything that falls back on Parakeet.
  - `models()` (GET `/models`) lists the speech models first (`SPEECH`: whisper, transcribe, asr, parakeet, canary,
    voxtral...), then the rest, since a gateway lists its chat models too.
  - `set_url()` takes a new address from the next request on, so the window never waits for a dictation.
  - The signature includes a hash of the address: another server may give other text for the same model name.
- **Storage:** the address and key are kept under `SPEECH_SERVER` (`"speech-server"`) in the profile's
  `gateway.json`, encrypted like the others, **apart from AI cleanup's own server**. The two may differ (e.g. a
  company gateway for cleanup and Speaches on this computer for speech), and changing one never breaks the other.
  The model is `Settings.speech_server_model`. `usable()` needs an address.
- **The page:** the "Your own server" tab has one card:
  - Address, API key (optional), Load models, Model, Test, Language
  - a new card starts from AI cleanup's own server ("vLLM or another OpenAI-compatible server", e.g. the company
    gateway) and says so
  - "Use this model" asks nothing (the owner counts their own and their company's server as their own); the card
    says that the voice goes to the server, and Parakeet takes over when it can't be reached
  - while in use, "Save" applies a changed address, key or model without a reload
- **The company gateway:** its `whisper-1` transcribed on 2026-09-30 (see **Company AI gateway**). The owner tries it
  from the card: Load models → whisper-1 → Test.
- **Also:** the chosen segment tab (On this computer / Cloud / Your own server) is no longer bold, which clipped
  "Your own server".
- `sst eval --engine server` scores through the server, without the fallback.
- **Tests: 317.** They cover the request without and with a key, the address required, Load models' order, a new
  address used from the next request, the catalog, the separate storage, the TrayApp switching address without a
  reload and back to Parakeet when the server is removed, and the card (prefilled from AI cleanup's server, Load
  models, Test, Use, Save).

## Parakeet downloaded on demand (phase 18)

- **Where Parakeet comes from:** `csukuangfj2/sherpa-onnx-nemo-parakeet-unified-en-0.6b-int8-non-streaming` on Hugging
  Face (the sherpa-onnx author's copy), revision `8c3a10fb`, pinned in `sst/engines/parakeet.py` (`MODEL`):
  - the encoder, decoder, joiner and tokens.txt, plus NVIDIA's model card (bias, explainability, privacy, safety)
  - 663 MB, every file's SHA-256 checked
  - byte for byte the files Rflow 1.4 installed: the same hashes, and the same folder name, so the eval cache still
    applies
  - Downloaded through `sst.downloads` like Whisper (resume, cancel, retries), into
    `%LOCALAPPDATA%\sst\models`. Tried for real on 2026-10-01: 663 MB in 370 s, then the sample transcribed with
    hotwords on.
- **Comes with Rflow:**
  - `sst/static/parakeet/bpe.vocab`: 10 KB, cut from NVIDIA's 2.5 GB `.nemo`; the hotwords need it, and the
    Hugging Face copy lacks it. `.gitattributes` keeps it byte-exact (`*.vocab -text`), because sherpa-onnx parses it.
  - `sst/static/sample.wav`: the sample sentence for the scan and the cloud and server Tests; a LibriVox reading, see
    NOTICES.
- **Found first next to the program** (`parakeet.find_model()`): `MODEL_DIR` is where Rflow 1.4 and older installed
  it, or the source checkout's `models/`.
  - An update keeps it: the installer's `[InstallDelete]` only removes `_internal`, so `{app}\models` stays.
  - `SpeechModel.found` counts it as installed, and "Remove download" only shows for a real download.
- **No speech model yet** (a new install that hasn't chosen one): `usable()` returns `""`.
  - The app starts without dictation, and the status says "Choose a speech model to start dictating".
  - A welcomed user is told once.
  - Dictation starts as soon as Parakeet is downloaded (`_on_download_done` also calls `_load_speech`, since
    `parakeet` is the default setting) or a cloud or server model is chosen.
- **The welcome** has a new step 2, "How Rflow recognises your speech":
  - "Download Parakeet (663 MB)", with progress and Cancel ("Meanwhile, choose your microphone")
  - or "Use a cloud model or your own server instead", which finishes the welcome on the Speech recognition page

  Step 4 ("Try it") waits for it.
- **Without Parakeet:**
  - The cloud and server cards say "Download Parakeet too ... to have it type when ... can't be reached", instead
    of promising it.
  - A failed transcription keeps the recording (`recordings`), and the error says so.
- **The installer:** `build_installer.py` links the model in for the smoke test (rflow-cli transcribes the sample;
  `Rflow.exe --self-test` transcribes it when it finds the model), then takes it out before Inno Setup. Solid LZMA2
  now (many small files).

  | | Installer | Installed |
  |---|---|---|
  | 1.4.0 | 571 MB | ~900 MB |
  | 1.5.0 | 90 MB | 288 MB, plus 663 MB once Parakeet is downloaded |

  CI still downloads the model with `scripts/download_model.py` for the smoke test.
- **The website and the installer's text** no longer say the model is included or that no internet is needed. They
  describe the choice, Whisper's languages, and what goes where (FAQ "Is my voice sent anywhere?").
- **Version 1.5.0** (`sst/__init__.py`): phases 14-18.
- **Not removed on uninstall:** downloaded models stay in `%LOCALAPPDATA%\sst\models` (shared with the source
  checkout), like the settings.
- **Tests: 330.** They cover the pinned download, Parakeet found next to the program or downloaded, no model at all,
  the bundled vocabulary (no CR) and sample, a new install starting without a model then downloading Parakeet, a
  cloud-only start, the welcome's step, the pages without Parakeet, the kept recording, and the fallback's message.

## The voice pipeline (phase 19, the owner's plan of 2026-10-01)

The owner wrote the plan in `appropriate_plan.txt` (repository root, not in git): dictation as separate stages with
contracts. It is built as written, with the deviations below, each backed by a measurement or a reason.

```
mic (always on, 2 s pre-roll in RAM) -> session -> VAD -> chunker (1.2 s pause / 20 s max, 1 s overlap at the max)
-> bounded parallel ASR (retry, sanity check, Parakeet fallback) -> ordered results -> overlap merge
-> dictionary -> formatting -> LLM polish -> guard -> FinalText -> the existing paste
```

| Stage | Where | What it does |
|---|---|---|
| Contracts, config | `sst/pipeline/contracts.py` | the objects passed between stages; every tunable value (`VoiceConfig`) |
| Capture | `sst/audio.py` `Recorder` | always-on (`keep_open`, `warm_seconds = inf`), `preroll_seconds` 2 s for dictation, `current_take` for live feeding |
| Ring, VAD, chunker | `sst/pipeline/audio_stream.py` | `RingBuffer` (monotonic index), energy VAD with adaptive floor and hysteresis, `Chunker`, `trim_preroll` |
| ASR | `sst/pipeline/asr.py` | `EngineBackend` (any engine), `ASRScheduler` (bounded pool, per-session ordering, retry by error class with backoff and jitter, `ASRValidator` suspicion score, fallback), fail closed (`SessionFailed`) |
| Word times | `sst/engines/cloud.py`, `parakeet.py`, `whisper.py` | `transcribe_chunk()` -> `RawTranscript` with words: Gemini 3.5 Transcribe (`audioTranscriptionConfig`, verbatim, `wordTimestamp`), whisper-1 / Groq `verbose_json`, Parakeet token times, faster-whisper words; a connection pool for parallel requests; the Files API for audio over 14 MB |
| Merge | `sst/pipeline/merge.py` | boundary-only: timestamps, exact, normalized, fuzzy; keeps both when unsure; partial words at a forced cut |
| Dictionary | `sst/pipeline/dictionary.py` | SQLite per profile (`dictionary.db`): terms, sound-alikes, AUTOMATIC / CAREFUL / HINT_ONLY, phrase and fuzzy matching with an ordinary-word guard (`sst/static/common_words.txt`); Your words mirrored in |
| Learning | `sst/pipeline/learning.py` | correction events from Home's ✎; a suggestion after the same correction twice; Add / Dismiss on the Dictionary page |
| Formatting | `sst/pipeline/formatting.py` | numbers, ordinals, dates, times, money, percentages, units, versions, emails, URLs; prose stays ("two options") |
| LLM | `sst/pipeline/polish.py` | `POLISH_PROMPT` (strict); `GatewayLLM` wraps the AI cleanup provider; only the terms present are listed |
| Guard | `sst/pipeline/guard.py` | protected entities, negation, speech act, substitutions, added/removed words; the formatted text is typed when it rejects |
| Orchestrator | `sst/pipeline/session.py` | `Session` (feed / finish / result, state machine, metrics), `VoicePipeline`, `LazyBackend`, the debug folder |
| Dictation, app | `sst/dictate.py`, `sst/app.py` | a session per key press fed by `_Feeder`; `retry_last()`; the pipeline rebuilt when the model or cleanup changes |

- **Deviations from the plan:**
  - Audio stays at the microphone's rate inside a session and is resampled to 16 kHz per chunk. A streaming
    resampler would smear chunk boundaries; the result is the same.
  - **No overlap after a pause cut.** The cut sits 0.3 s after the last word, so a 1 s overlap reached back into it.
    On the owner's recordings Parakeet then decoded the next part to nothing. The overlap stays at the 20 s maximum,
    where words can be cut.
  - **A pause ends a part only after 0.4 s of its own speech** (`min_cut_speech_ms`). A breath or a word's tail alone
    made Parakeet hear "Yeah."; it now stays with its neighbour.
  - **Before failing closed, Dictation transcribes the whole recording in one piece** (as before the pipeline) and
    runs the text stages on it. That text is complete, so it isn't a text with a hole in it. Only if that fails too
    is nothing typed; the recording is kept, and the tray's "Retry the last dictation" tries again.
  - After a pause cut the session merges without an overlap, even when an engine gives no word times.
- **Gemini:** `gemini-3.5-transcribe` is the default Gemini speech model. It runs VERBATIM, and `timestamp_mode`
  "timestamps" asks for word times; "vocabulary" sends Your words instead, since Google says the two can't be
  combined. Google doesn't fully document the REST response, so the parser accepts several shapes; it is tested
  only against fakes so far.
- **The microphone is on while Rflow runs (default, the plan's locked spec).** Only the last 2 s are kept, in RAM.
  Windows shows the microphone icon. It is never kept open for Bluetooth headsets. Settings can turn it off.
- **Settings:**
  - "Keep the microphone on while Rflow runs"
  - "Write numbers, dates, times and money as such"
  - "Voice pipeline" (off: the classic whole-recording path)
  - "Keep each dictation's steps for troubleshooting" (`%LOCALAPPDATA%\sst\debug\<session>`)
- **The Dictionary page:** "When Rflow writes ... write instead ..." adds a sound-alike, which always applies. Each
  word shows what it is also heard as, and suggestions from corrections appear at the top.
- **Measured on the owner's voice** (12 reading-test sentences from this laptop joined into one 100 s dictation:
  every third with a real pause, the others run together; local Parakeet, one part at a time):

  | | Words wrong | Parts |
  |---|---|---|
  | The whole recording at once (before) | 23 of 166 (13.9%) | 1 |
  | The pipeline | **20 of 166 (12.0%)** | 9 (2 cut at the maximum, 6 at pauses, the end) |

  The overlap of the two forced cuts was merged without duplicates. Script: `real_pipeline.py` in the session's
  scratchpad (not in git); it reads the reading tests in `%LOCALAPPDATA%\sst\bench`.
- **Known limitations:**
  - A quiet phrase after a mid-sentence pause ("...credit card | tomorrow morning") was still dropped. The energy
    VAD heard too little of it to cut there, and Parakeet drops a quiet phrase before louder speech. A model VAD
    (Silero via sherpa-onnx) is the likely fix.
  - At a forced cut the merge keeps the earlier part's version of the overlap words, which had less context
    ("went up against" where the next part heard "went up again"). Preferring the next part's words in the second
    half of the overlap is worth trying.
  - The fuzzy dictionary is English (ASCII), and so are the guard's question and correction rules. Rejections only
    mean the formatted text is typed.
  - Learning only sees corrections made with Home's ✎, not edits in the target app.
- **Tests: 1525.** Every stage has its own file (`tests/test_pipeline_*.py`, `tests/test_engine_words.py`), plus
  end-to-end sessions with synthetic speech and fake engines (`tests/test_pipeline_session.py`).

## Translate (phase 22, the owner's idea of 2026-10-02)

Like DeepL: select text in any app, press Ctrl+C twice, read it translated. The owner removed DeepL from this laptop so
Ctrl+C+C is free; M4 Translator still runs there and is left alone (Rflow reads the copy the moment it lands, before a
clipboard tool can rewrite it).

- **The shortcut** (`Settings.translate_shortcut`, "ctrl+c+c"; Ctrl+Alt+L or off on the page):
  `parse_hotkey("ctrl+c+c")` is a double press (`Matcher._double_press`: the key twice with exactly the modifiers held,
  within `PRESS_GAP` 0.5 s; Ctrl may stay held; repeats, other keys and slow presses don't count). Nothing is hidden:
  the app copies as usual. Another shortcut makes Rflow copy the selection itself (Ctrl+Insert, never in a terminal).
- **The text** (`TranslateController._read`): the clipboard as soon as its sequence moves after the shortcut (at most
  `COPY_WAIT` 0.4 s), with `textaccess.clipboard_text()`; up to `MAX_CHARS` (5,000).
- **The popup** (`TranslatePopup`): at the pointer, dark, never takes focus (`_no_activate`), so the app keeps its
  selection. The language list at the top translates again into exactly the language picked and remembers it; the
  translation box fits its text; the status line shows the time and model, or a warning. Esc (taken from the hook
  while it is open), x, or going to another app closes it (Rflow's own windows, like the list, don't count:
  `textaccess.window_process`). **Copy** leaves the translation on the clipboard; **Replace** puts it in place of the
  text after `transformui.place` (now shared with Text Transform) brought its window back and found the text, else
  the clipboard and "press Ctrl+V".
- **The translation** (`sst/translate.py`): the AI cleanup's model and backup (`Polisher.complete`) under a translator
  prompt (only the translation; questions and requests translated, never answered; names, numbers, dates, links and
  code kept; layout kept; same tone). Text already in the chosen language goes to the second language
  (`Settings.translate_second`; `already_in`: by script, kana for Japanese, small words for English). `check` names
  numbers (by their digits, 2万5000 = 25,000), emails and links the translation doesn't show, as a warning only.
- **The page:** Translate (sidebar, after Text Transform): the shortcut, the language, the second language, the model,
  Try it.
- **Tests: 1980.** `tests/test_translate.py`, `tests/test_translateui.py` (the flow with fakes: the copy read,
  nothing to translate, a second language, picking a language, an older answer dropped, Copy, Replace and its
  clipboard fallback, Esc, another app, a failing model, another shortcut, terminals), `tests/test_hotkey.py`
  (ctrl+c+c), the page, the app, settings, `tests/test_textaccess.py` (clipboard_text).
- **Not yet tried:** real apps and the owner's model.

## Snippets (phase 21, the owner's idea of 2026-10-02)

Like Wispr Flow's snippets: say a short phrase, get your own text typed ("my email" → xyz@gmail.com, "my signature" →
a signature of several lines). The owner asked whether to do it in the AI prompt or after the text; neither alone is
reliable (the AI may skip, reword or leak it, and the guard would refuse a new email address; after the AI, the
phrase may be reworded), so it is both ends:

- **Found in the words heard** (`sst/snippets.py`), before the dictionary, formatting and the AI cleanup:
  - **Alone** (the default): the whole dictation is the cue, a filler or "please" around it allowed; small
    mishearings count (`alone`: compact letters, so "my e-mail" is "my email"; fuzzy 0.9 for cues of 8+ letters, never
    a word short). The text is typed as is: no stage, no AI. A snippet wins over a Text Transform voice command.
  - **Anywhere** (per snippet, opt-in): the cue inside a sentence (whole words) becomes a placeholder `RFSNIP1`
    (`protect`), which the AI is given as a protected term and the guard protects as code. The snippet's text goes in
    after the guard (`expand`), exactly as entered, line breaks included; if the placeholder doesn't come back exactly
    once, the text from before the AI is used, and last the words as heard. Never the default: "I checked my email
    this morning" must stay as said.
- **Where:** the pipeline (`Stages.snippets`, in `_text_stages`) and the classic way (`Dictation.snippets`), both
  from `TrayApp.snippets()` (read from the profile at each dictation: a change counts at once).
- **Stored** in the profile's settings (`Settings.snippets`: cue, text, anywhere; `Settings.load` checks the items are
  objects). The text stays on this laptop; the AI never sees it.
- **Line breaks:** `paste.paste_text` now gives the clipboard "\r\n" (a classic edit box shows a bare "\n" as nothing).
- **The page:** Snippets (sidebar, after Dictionary): add (what you say, the text, "Also inside a sentence"), edit,
  remove, and Try it (type what you'd say, see what would be typed).
- **Tests: 1925.** `tests/test_snippets.py` (matching, placeholders lost, doubled or glued, longest cue first,
  loading), the pipeline and classic paths (`tests/test_pipeline_session.py`, `tests/test_dictate.py`: an AI that keeps
  or drops the placeholder), settings, the page, the app, `tests/test_paste.py`.

## Text Transform (phase 20, the owner's idea of 2026-10-02)

The owner's `post_text_transformation_idea.md` (repository root, not in git): "Speak normally first. Transform the text
afterwards only when you need to."

The owner's feedback on the first version (a shortcut, then a menu): the shortcut was hard to press, it took three or
four steps, and a transform was lost when the pointer left the menu. The second version, agreed on 2026-10-02:

- **Two ways in:**
  1. **Say it** (the main way): hold the dictation key and say only the command: "make it concise", "make it
     professional", "bullet points", "action items", "rewrite it", "undo that"... (`sst/commands.py`,
     `DEFAULT_PHRASES`). A whole dictation that is one command phrase (fillers and "please" around it allowed, at most
     8 words, a slightly misheard phrase counts) is a command: it isn't typed and doesn't go into the history. Anything
     more is dictation ("make it concise and send it to John" is typed). One-word phrases aren't defaults
     ("professional" alone is a word people dictate), except "undo". Each user can edit the phrases per command on
     the Text Transform page (`Settings.command_phrases`: only the commands they changed; an empty box turns a
     command off) or turn voice commands off (`Settings.voice_commands`). Checked on the words heard, before the
     dictionary, formatting and the LLM: `Stages.command` in the pipeline, `Dictation.command` in the classic way;
     both hand it to `TrayApp.voice_command`, then `TransformController.run_command` on the app's thread. Without an
     AI model a transform's phrase is typed as said; "undo" is always a command.
  2. **The menu:** double-tap Ctrl by default (`Settings.transform_shortcut`; Ctrl+Alt+T, F8 or off on the page).
     `parse_hotkey("double ctrl")`: two taps, each held under 0.35 s with no other key, the second within 0.4 s;
     Ctrl still reaches the apps. Two quick Ctrl+clicks aren't one (the pointer moved between them:
     `TAP_POINTER_PX`); the keyboard hook can't see clicks, and a mouse hook would slow every mouse move.
- **Flow** (`sst/transformui.py`, `TransformController`):
  1. **The text** (`_find`): `textaccess.copy_selection()` presses **Ctrl+Insert** (not Ctrl+C: see below) and puts
     the user's clipboard back. With nothing selected, the last dictation or transform typed in that window
     (`note_typed`, within 15 min) is selected with Shift+Left (`select_last`) and checked by a copy; a mismatch
     collapses the selection and nothing happens. **Never in a terminal** (`TERMINALS`: Ctrl+C there stops the
     running program). Each step is logged ("selected again in Chrome_WidgetWin_1", "other text is selected now"...).
  2. **The menu** (`TransformMenu`, the shortcut only): at the pointer. It takes no focus, so the app keeps its
     selection. Its keys (1-9, numpad, Up/Down, Enter, Esc, U) are taken from the hook while it is open
     (`HotkeyListener.capture`); clicks work too. A voice command skips it.
  3. **The transform** (`sst/transform.py`): the AI cleanup's model and backup (`Polisher.complete`, no cleanup
     length check, 8 s + 0.05 s a word) under `system_prompt()`: a writing tool, never an assistant; keep every
     value, name, term, negation, condition, choice, uncertainty and question; apply self-corrections; never invent.
  4. **The check** (`TransformGuard`), independent of the model. It rejects:
     - lost or new values (numbers in words or digits, dates, times, money, percentages, units, emails, URLs, code,
       terms) — a corrected value may go
     - new names
     - dropped or added uncertainty or negation
     - lost conditions, qualifiers and choices, and swapped opposites
     - a question lost or answered
     - added sentences, assistant replies
     - Concise or Action items longer than the input

     One repair request names the problems. If it is still rejected, the user's text stays and the pill says why.
     Action items answers NO_ACTIONS when the text has none.
  5. **Bring it back** (`_place`), right before the paste: if another window is in front, the text's window is
     brought back (`textaccess.activate`: restore if minimised, AttachThreadInput + SetForegroundWindow). Then the
     text must still be the selection (a copy, compared), or, with nothing selected (a click in the same box), be
     found again right before the caret (`select_last`). If any of that fails, nothing is pasted: the result goes on
     the clipboard (`textaccess.set_clipboard`, kept there for the user) and the pill says "press Ctrl+V". It is on
     Home too.
  6. **The replacement** (`textaccess.paste_rich`): plain text plus CF_HTML, so rich editors (Teams, Outlook,
     Word, Slack, browsers) get bold headings and real bullet lists, while plain editors get "- " lines. The text's
     own surrounding spaces are kept.
  7. **Undo:** say "undo that", or open the menu again right after a transform and press U ("restore the
     original"): the original is pasted over the transformed text (brought back the same way). Ctrl+Z in the app
     works too. Each transform is in Home's history with its original as the tooltip.
- **Transforms** (`TRANSFORMS`): Concise, Professional, Bullet points, Action items (the default menu,
  `Settings.transforms`), and Rewrite.
- **The page:** Text Transform (sidebar, after AI cleanup):
  - Say it: voice commands on or off, and a box of phrases per command (comma-separated; "Use the default phrases";
    a phrase in two commands is pointed out)
  - the menu: its shortcut and its transforms
  - the model used
  - Try it: a box with a button per transform, showing the result as it would be pasted
- **Found in the owner's first test (2026-10-02), fixed:**
  - **NumLock:** keys were sent without the extended-key flag, so Left was the number pad's; with NumLock on, Windows
    lifts Shift around it and Shift+Left moved the caret instead of selecting. "Make it concise" on the last dictation
    always said "Select some text first". `hotkey.key_input` now flags arrows, Home/End, Page Up/Down, Insert, Delete
    (`EXTENDED_KEYS`) and adds scan codes.
  - **A translator on Ctrl+C:** the owner's laptop runs M4 Translator and DeepL; two Ctrl+C in a second made one put a
    Tamil translation on the clipboard, which Rflow then read as the selection. Rflow copies with Ctrl+Insert (Ctrl+C
    only as a fallback where text is known to be selected: `copy_selection(fallback=True)`).
  - **Pasting right after a copy:** an app that has just copied (Qt) answered an immediate Ctrl+V with its own copy.
    `paste_rich` waits `SETTLE_DELAY` (0.15 s) between the clipboard and Ctrl+V.
  - **Spoken corrections:** the check rejected a correct transform of "five servers wait make that 35 servers",
    "$25,000 not the 20,000" and "five times sorry three times actually five". The dictation guard
    (`sst/pipeline/guard.py`) now reads "wait/no/sorry + make that / I mean" as one cue, "Actually no," as a cue, and
    finds the corrected value with its noun before the cue; the transform check reads "X not (the) Y" (same kind of
    value) as telling X from Y, so Y may go once X is kept.
  - Checked end to end on real windows (a test text box in its own process, the real `textaccess`, the model replaced
    by a fixed answer): short and long last dictations, a selection, undo, a switched window brought back. The script
    is not in the repository; it types into the window in front, so run such checks only with the owner away from
    the keyboard.
- **Tests: 1889.**
  - `tests/test_transform.py`: every good/bad example in the idea file, prompts, the repair, `render`
  - `tests/test_commands.py`: what is a command and what is dictation, the user's phrases
  - `tests/test_textaccess.py`: clipboard kept, selection check, CF_HTML offsets, activate, set_clipboard, Ctrl+Insert
    and its Ctrl+C fallback, the pause before Ctrl+V
  - `tests/test_hotkey.py`: menu key capture, double taps (slow, long, mixed with shortcuts, Ctrl+clicks)
  - `tests/test_transformui.py`: the whole flow with fakes: voice commands, undo, terminals, switched windows
    brought back, the clipboard fallback, failures
  - `tests/test_dictate.py`, `tests/test_pipeline_session.py`: a command is handed over, never typed
  - window and app tests
- **Not yet tried:** against real apps and the owner's model. The checker's rules are heuristics and may reject a
  good transform now and then: the text is then kept, never damaged. Known gaps: a single swapped noun ("staging"
  for "production") or a fact restated wrongly with the original's own words passes. VS Code's terminal can't be
  told from its editor (Ctrl+C there with nothing selected interrupts the program). A dictation that is only
  "Action items." or "Bullet points." (a heading in notes) is a command: the user can remove those phrases.

## Text never said (fixed 2026-10-02)

The owner saw words they hadn't said, at the end of dictations ("…version installer. Commit tray app pill. Let's move
stand meeting Thursday morning.") and once a rewritten opening ("Under some protocol, that's not a big deal."). Speech
model: Gemini `gemini-flash-lite-latest`, with the 20 Your words in its instruction.

- **Evidence** (40 pipeline dictations, each replayed locally through Parakeet; nothing sent anywhere):
  - Gemini invented the words; they weren't in the audio. In 7 of 40 dictations its text had list words that Parakeet
    didn't hear; the AI cleanup only passed them on.
  - The leaked text is the Your-words list itself, in its saved order.
  - The trigger is a last part with little speech: 5 of 5 with 1.4 s or less of their own speech got the list, 0 of 5
    with 2.1 s or more. Those parts held mostly the repeated overlap, the release click and Rflow's own stop beep.
  - "Under some protocol…" was the owner's own speech ("And some of the words are not being…"): a 1.8 s mid-sentence
    pause cut a 2.5 s fragment, which Gemini rewrote without its context.
  - The 2 s pre-roll did not leak. Rflow's 880 Hz start beep was in 39 of 40 recordings, louder than the voice.
- **Fixes:**
  - Only names and terms go to speech models as hints, and to the guard as protected terms (`speech_hints`): 16 of
    the 20 words were everyday words. The Dictionary page marks everyday words.
  - The Gemini instruction calls the hints a spelling reference, never to be written unless spoken.
  - A transcript with three or more hint words in a row, in the list's order (`hint_echo`), is transcribed again
    without hints; that transcript is used when the words disappear.
  - A last part with under 2 s of its own speech goes again with the part before it, as one chunk that replaces it
    (`min_alone_speech_ms`, `AudioChunk.replaces`).
  - A pause cuts only after 4 s of a part's own speech (`min_cut_speech_ms`, was 0.4 s).
  - The stop beep plays after the tail is recorded; the start beep is filtered out (`_ToneNotch`, 880 Hz, 0.6 s).
  - The fallback, Retry and saved recording leave out the pre-roll the session trimmed. The tail is no longer copied
    into the next pre-roll. The classic path keeps its 0.4 s pre-roll. The guard accepts either apostrophe and a
    capital at a sentence start.
- **Measured:**
  - On the 45 recordings, short parts (< 4 s) sent alone in multi-part dictations went from 9 to 1, and that one has
    2.1 s of speech. The owner's example now has 2 parts (0-18.3 s, 17.3-27.2 s) instead of 3.
  - The 12-sentence real-voice check (local Parakeet): 22 of 166 words wrong (13.3%), against 20 before and 23 for
    the whole recording at once. A 3 s pause threshold gave the same.
- **Owner's choice, recommended:** Speech recognition → Google Gemini → model `gemini-3.5-transcribe`. It is Google's
  dedicated speech model, takes no free-text prompt, and is the more accurate one. The `-latest` alias may now point
  to a Flash-Lite reported to transcribe dictation worse. Also remove the everyday words from the Dictionary.

## This laptop's first reading test (2026-10-01, Rflow 1.4.0)

- **The setup:** set B, the laptop microphone (Realtek, WASAPI, 48 kHz), Windows mode, warm microphone.
  - Speech level -37 dBFS, signal to noise 23 dB, full band.
  - **No first words lost (0 of 30):** the warm microphone works. On the other laptop set B lost 5.
  - No last word was cut off.
- **Results** (30 sentences, so about ±4 points):

  | Setup | Word errors | Names and terms |
  |---|---|---|
  | Parakeet, no words | 12.5% | 44% |
  | Parakeet, this laptop's Your words | 12.0% | 44% |
  | Parakeet, the 37 names and terms (`bench.TERMS`) | 11.8% | 36% |
  | **Parakeet + the company's Qwen3.8-27B cleanup** | **8.7%** | 24% |

  The cleanup is clearly better: -3.3 points, range -5.7 to -1.3.
- **This laptop's Your words** held mostly everyday words, from the old reading test's suggestions (version,
  installer, tray, app, move, stand, meeting, morning, after...). The owner was told to keep only names and terms.
- **Names still wrong with the names list:** Parakeet → parquit, Ollama → olama, Qwen → current, Groq → grok,
  PowerToys → power toys, Vercel → vessel, commit → comet, PyInstaller → py install. That is the correction fixer's
  ground.

## Correction groundwork, for after the building blocks (experiments on 2026-10-01, nothing merged)

These were scratch scripts, not in git. The findings:

- **Word confidence exists.** With beam search, `stream.result` has `tokens`, `timestamps` and `ys_log_probs` (one per
  word piece; a piece starting with a space starts a word). A word's confidence = the lowest probability of its
  pieces. On the owner's 150 sentences (with 31 names as hotwords, 1912 words heard, 6.1% wrong):

  | Confidence | Words | Wrong |
  |---|---|---|
  | 0.97 or more | 1113 (58%) | 1.8% |
  | 0.90-0.97 | 210 | 9.5% |
  | 0.70-0.90 | 259 | 10.0% |
  | 0.50-0.70 | 202 | 12.9% |
  | 0.30-0.50 | 105 | 21.0% |
  | below 0.30 | 23 | 8.7% (hotword-boosted pieces can have a low model probability and still be right) |

  - Good for skipping correction when every word is at least 0.97.
  - Weak as a detector of wrong words: flagging below 0.7 catches 43% of the errors, and 85% of the flagged words are
    right.
- **Errors left after hotwords** (135):
  - 70 ordinary substitutions: the/a, test/tests, note/notes, and some reading variations
  - 27 names: Ollama → olama ×2, onnx → onix ×2, Parakeet → parkit / parkeet / parketh / rakit, Qwen → quinn /
    quin / coin, Groq → grog / grok / how, Vercel → versal, PowerToys → powertise, Haiku → high
  - 19 dropped words and 19 extra words
- **Sound-alike fixer prototype:**
  - Spans of 1-3 low-confidence words, compared with Your words by letter edit distance, halved when a rough sound
    key matches.
  - Results:
    - similarity 0.8, confidence below 0.9: 7.1% → 7.0%, names 24.5% → **14.7%**, names put in wrongly 2 → 7
    - similarity 0.9: 6.9%, names 19.6%, 3 put in wrongly
    - 0.7 or looser: much worse ("in the" → "Inno", "a real" → "Rahul")
  - Its fixes: Olama → Ollama, Parkit / Parkeet / Parketh → Parakeet, PowerTise → PowerToys, Sherpa Onix →
    sherpa-onnx, Grok → Groq, "i installer" → PyInstaller.
  - Its harm: "installer" → "PyInstaller" (×3), "open a" → "OpenAI", "the cloud version" → "Claude", "Anthropic key"
    → "Anthropic".
  - **To build it properly:**
    - never change an ordinary English word; this needs a word list, e.g. SCOWL (permissive licence), or Windows'
      spell-checker API
    - never join a span that holds a word already in Your words
    - never join function words (a, the, in, on...)
    - tune on sets A-B, judge on C-E
- **AI cleanup only when unsure** (not started): send a sentence to the LLM only when a word is below about 0.9. Mark
  the uncertain words, and give only the words from Your words that sound like them. Reject answers that change too
  much. Research: `docs/research/notes/post_asr_and_evaluation.md` (whole-transcript LLM correction doubled word
  errors on Parakeet output in one study).

## Settings that change only on purpose (phase 23, the owner's request of 2026-10-04)

- **What the owner hit:** scrolling the Speech recognition page with the pointer over a dropdown changed the model or
  the language (the language was saved at once); long lists had to be scrolled through; nothing said whether a
  change was saved. The owner's direction: user experience first, the look later.
- **The rules now** (`sst/window.py`, "form controls"):
  - every dropdown is a `Choice`: the wheel never changes a closed one, it scrolls the page (a test checks that the
    window has no other kind)
  - a long list (the speech language, Translate's languages) opens with a search box: type, Enter picks the first
    match; a model box filters its list by what is typed (anywhere in the name), and any name can still be typed
  - a form has one `SaveBar`: Save is greyed out until something changes, then "Unsaved changes" with Cancel, then
    "Saved"; leaving the page with changes asks (Stay / Discard changes)
  - an API key is a `KeyField`: saved, it shows dots and its last four characters, with a pen; being changed, it's
    hidden, with Show, Paste and a cross back to the saved key
- **Speech recognition** starts with the model in use ("In use: Google Gemini · gemini-3.5-transcribe") and the
  language you speak, set once there (it was repeated on every card, though one setting). The cards keep their key,
  model, Test and "Use this model" (or Save for the one in use).
- **AI cleanup**: the Save is in the card, with Cancel; switching providers and back is no change.
- Toggles on Settings, Text Transform and Translate still apply at once (one click, one change); they got the wheel
  fix only. The UX review of the other pages is in **Next steps**.

## The Translate popup, redone (phase 24, the owner's request of 2026-10-04)

- **What was wrong** (found in the code, the log and the popup rendered in every state):
  - a long translation ran under Copy and Replace: the window was sized before its word-wrapped labels were
    measured, and the box measured its text at Qt's default width (632 px) instead of its own
  - it was placed while small, then grew downwards off the screen
  - the language dropdown re-translated and saved a new language when the wheel went over it
  - with no second language, English text went "into English"
  - no word of the text's language; "1.6 s · model" shown; raw provider errors, no Try again
  - without an AI model, Ctrl+C+C only flashed a notice in the pill
  - Copy closed it at once; a click elsewhere in the same app left it open
- **Now** (`sst/translateui.py`, `TranslatePopup`):
  - the header says "Japanese → English" (`translate.detect`: only when the letters make it certain, so
    Devanagari, Cyrillic and most Latin-alphabet text get no name rather than a guess)
  - the languages used most are one-click buttons beside it (never the text's own); More shows all 25 in a grid
  - every part is as tall as its text, measured; the translation up to 360 px or 40% of the screen, then it scrolls.
    The popup opens upwards when there's little room below, and is placed again whenever it grows
  - errors are said in plain words (`explain`), with the provider's own words small below and Try again; a refused key
    or no connection also offers "Set up AI cleanup", as does having no model at all (the popup opens and says so)
  - Copy says "✓ Copied" and the popup stays; a new click outside it closes it (`textaccess.mouse_down`); Esc in the
    language list goes back
  - text already in the chosen language goes to the second language, else to Windows' own language, else English
    (`translate.fallback_second`, `system_language`): the Translate page calls this "Automatic"
- The empty gap above "Try it" (Translate, Text Transform, the welcome) and "Add a snippet": the text boxes kept their
  growing size policy despite a fixed height, so their card grew and the heading took the room (`fixed_height`).
- **A popup with buttons must not be click-through** (1.10.1): `sst.app._no_activate` adds `WS_EX_TRANSPARENT` for the
  pill only; Translate's popup and the Text Transform menu pass `click_through=False`. The tests can't see this (Qt
  runs off the screen there), so after changing a popup run `uv run python scripts/check_popup_clicks.py` on Windows:
  it hit-tests every button the way Windows routes a real click and clicks it in Rflow's own window only.

## Rflow UI 2.0 (phase 25, the owner's request of 2026-10-04)

- **The design**: "Obsidian Signal", made in Figma first ("Rflow UI 2.0",
  https://www.figma.com/design/PTXZYTNbOrcCIR2eGiXdXF) from `docs/design/rflow-ui.html` (14 frames; `render.ps1`
  renders them with headless Edge), after a written UI audit and a critic's review. The owner approved it ("perfect
  work") and asked for the implementation and v2.0.
- **Colours** (`sst/theme.py`, `TOKENS`): Obsidian (dark, the default) and Porcelain (light), chosen by Windows' mode.
  One job per colour: Iris = can be pressed, Violet = the AI, Coral = listening, Mint = ready, Amber = attention, Rose
  = error. Text is 4.5:1 and control edges 3:1 on every surface of its theme (checked for the design). Popups over
  other apps (the pill, Translate, the Text Transform menu) are always Obsidian (`POPUP`): dark reads on any app.
- **The soft depth Qt can't do**: style sheets have no shadows, so `theme.paint_surface` draws them: a raised surface
  has a light and a dark soft shadow and a 145° gradient, a pressed one (a well, a field, the chosen nav item) inner
  shadows. Shadows are blurred once per radius, blur and colour (numpy) and drawn nine-sliced, snapped to device pixels
  (no hairlines at 125%/150%); a circle or a pill's round end uses the whole tile, scaled.
- **Surfaces are painted by their host**: `surface(widget, "key")` gives a widget a surface; its nearest host (a page
  body, a card, the sidebar, a popup's panel; `make_host` + `paint_hosted` in its paintEvent) paints it beneath it,
  with its shadow reaching outside the widget. The widgets themselves stay ordinary Qt widgets with transparent
  backgrounds, so text, focus and keys work as always. A watcher repaints a surface on hover, press, focus, move and
  show/hide. Pages keep 32 px around their cards: at 24 px the light theme's white highlight showed as a line where
  the page's edge cut it.
- **Fonts**: Geist and Geist Mono, the official static TTFs (Regular/Medium/SemiBold/Bold, Mono Regular/Medium/
  SemiBold; 0.9 MB, `sst/static/fonts`, SIL OFL with `OFL.txt`; from vercel/geist-font v1.7.2, with the owner's OK),
  registered at start (`theme.load_fonts`). `theme.font(px, weight, mono, spacing, tabular)`. **A style sheet's
  font-size overrides setFont**: don't set font sizes on `QWidget` in the sheet; labels get theirs from `role`
  properties, others from `setFont`. Japanese falls back to Yu Gothic, which is taller: give one-line labels room.
- **Icons**: the design's line icons (`theme.ICONS`, 24-unit SVG paths) drawn with Qt SVG in any colour
  (`icon_pixmap`). The Segoe Fluent glyphs are gone.
- **Widgets** (`sst/ui.py`): `Button` (key, primary, ghost, quiet, danger, nav, chip, segment; a keycap `hint` drawn
  inside, `suffix` for a tab's count, `badge` on nav), `IconButton`, `Toggle` (a QCheckBox drawn as a switch),
  `Lamp` (always with a word), `KeyCap`/`keys()`, `Orb` (ready, live, ai, loading with %, done, off), `Segmented`,
  `Meter`, `WaveProgress`, `Mark` (the logo; it was a keycap with a waveform until 2.0.1), `Toast`.
- **The sections** (`window.py`): `NAV` has five; `SECTION` maps every page to its section and `OPENS` a section to
  its first page (Words → `dictionary`). The old page keys all stay (`dictionary`, `snippets`, `speech`, `cleanup`,
  `transform`, `translate`, `reading`, `profiles`), one level down with a "‹ back" link; new pages are `tools` and
  `models` (overviews). `current_section()`; the welcome hides the sidebar. Below 900 px the sidebar becomes an 84 px
  rail (icons over names, "AI" for AI & models, the status as a lamp over "Ready").
- **Home**: the hero shows Ready (orb breathing, keycaps), or what Rflow waits for: Parakeet downloading (orb arc with
  %, an amber waveform, Use a cloud model instead, Pause, which keeps the partial file: downloads resume), loading,
  or no model (Download Parakeet / Use a cloud model). Stats strip; the list (30, then "Show all"), search box with
  Ctrl+K, Correct and Copy on hover.
- **AI & models** (`ModelsPage`): the speech model and the microphone (moved here from Settings, with the Bluetooth
  warning), the AI connection with a masked key and Test (`app.check_ai`), the cleanup switch (saves at once through
  `save_cleanup`). **Settings**: "Microphone stays ready" is one choice (while Rflow runs / 5 minutes after dictating /
  only while dictating) for `always_on_mic` and `warm_mic`; Advanced holds the voice pipeline, the debug steps, raw
  audio, Reading test, Profiles, logs, website, issues.
- **First run** (`WelcomePage`, three steps): how Rflow hears you (On this PC downloads Parakeet and goes on while it
  downloads; In the cloud leads to the speech page's cloud tab), try it (the orb lights up when the box gets text), and
  connect an AI: provider tiles, a key, then `app.ai_models` + `window.fast_model` (the newest Flash-Lite, 4.1-mini or
  4o-mini, Haiku, llama-3.1-8b-instant; never a preview, embedding, TTS or transcription model) + `app.check_ai`, then
  `save_cleanup(True, model)`. The name question is gone (Profiles renames).
- **Popups**: drawn in a 24 px transparent margin for their shadow (a tight "float" shadow that fades out inside it,
  else its edge showed as a box over light apps). `TranslatePopup.contains_global` is the panel without the margin
  (a click outside the panel closes it). Translate takes C (Copy) and Enter (Replace / Try again / Connect an AI)
  while those buttons are there (`keys()`, `keys_changed`); `scripts/check_popup_clicks.py`: 14 of 14 clickable.
  `_fit` resizes twice: the panel's own layout settles after the window's first pass (it came out 150 px too tall).
- **Gotchas met**: in PySide6 don't set attributes on `self` before `super().__init__()`; a QVBoxLayout with a
  max-width child centres the whole column (the page titles moved right); a word-wrapped label added with an alignment
  gets one line.
- **Screenshots and website**: `scripts/make_site_screenshots.py` renders Home (dark and light), AI & models, Tools,
  Words, the first run, the popups and the pill from the real widgets (never shown), as if Parakeet were downloaded.
  `site/index.html` is redone in the same look (Geist from Google Fonts, Obsidian by default, Porcelain in light
  mode; no horizontal overflow at 390 px).

## The logo (2.0.1, the owner's logo of 2026-10-05)

- The artwork is `docs/brand/rflow-logo.webp` (1254 px, the ribbon "R" above the word "Rflow", on white).
  `uv run python scripts/make_brand.py` makes everything from it: `sst/static/sst.ico` (16-256 px, PNG inside the ICO;
  the window, the taskbar, the tray, `Rflow.exe` via `packaging/sst.spec`, the installer via `installer.iss`),
  `sst/static/brand/rflow-mark-{64,128,256,512}.png` (the window's `ui.Mark`: the sidebar and the first run), and the
  website's `favicon.ico`, `img/logo.png`, `img/touch-icon.png` and `img/og.png` (the social card, 1200 x 630). Then
  `scripts/make_installer_images.py` (the wizard pictures, Obsidian) and `scripts/make_site_screenshots.py`.
- The mark is cut out of the white: a pixel at least as coloured as the ribbon's palest part is solid, a paler one is
  an edge, made transparent in proportion and given back its own colour, so it has no white fringe on Obsidian.
- The ribbon is only ~390 px in the artwork: the 512 px files are slightly upscaled. A vector (SVG) of the logo would
  make the website's and any larger use sharper; ask the owner for one if it exists.
- The words next to the mark stay live text in Geist (the app and the website), close to the artwork's wordmark.

## Different computers: x64 and ARM64 (the owner's choice of 2026-10-05)

The owner develops on a Surface Laptop 7 (Snapdragon X Plus, **ARM64**) and also uses an Intel laptop (i5-1334U,
**x64**), and mostly uses cloud speech models. Decision: **one x64 program for every Windows laptop** (option A);
no native ARM64 build for now, and the Mac later.

- **How it works:** Intel and AMD laptops run Rflow natively; ARM laptops run the same x64 program through Windows
  11's x64 emulation. The installer allows both (`ArchitecturesAllowed=x64compatible` in `packaging/installer.iss`).
  The dev `.venv` on the ARM laptop is x64 Python too (`uv venv --python cpython-3.12-windows-x86_64-none .venv`), so
  what is developed there is what ships.
- **Checked on both:** CI runs every test on `windows-latest` (x64) and on `windows-11-arm` with x64 Python
  (`UV_PYTHON`), and checks the latter really is "x64 on ARM64 (emulated)"; it also builds the app on both and runs
  `Rflow.exe --self-test` (every window, the fonts, the logo, Whisper's runtime, a Parakeet transcription).
- **Which computer a log came from:** the first line says it, "Rflow <version> starting (x64)" or "(x64 on ARM64
  (emulated))" (`scan.machine()`: the program's kind from how Python was built, `sysconfig.get_platform()`, and the computer's
  from `IsWow64Process2`. Not `platform.machine()`: since Python 3.12 it reports the processor, "ARM64" even inside
  the emulation; CI's ARM job caught that).
  Scan my computer shows it after the processor, and the bug report form asks.
- **Cost of emulation:** only heavy local work (Parakeet, Whisper on the processor) is slower; cloud speech and the
  AI cleanup are network-bound. Scan my computer measures the real speed, so its verdicts already include it.
- **A native ARM64 build later** (faster local speech on Snapdragon): every native library has a `win_arm64` wheel
  (numpy, sherpa-onnx and sherpa-onnx-core, PySide6, sounddevice; checked on PyPI 2026-10-05) except **CTranslate2**,
  Whisper's runtime, so Whisper would be missing there. It would need a second installer (`Rflow-Setup-arm64.exe`)
  and the updater picking it by `scan.machine()`; `Rflow-Setup.exe` must stay the x64 file, since installed copies
  download it by that name.
- **What differs between computers more than the processor** (each caused a real bug): NumLock (arrows must be sent
  as extended keys), clipboard and translator tools, click-through popup windows, screen scaling, Japanese fonts,
  Bluetooth microphones, admin windows. Test on real Windows what the off-screen tests can't see
  (`scripts/check_popup_clicks.py`).
## Microphones that follow you (phase 26, the owner's report of 2026-10-05)

- **What the owner hit:** connecting a headset (or another microphone) and choosing it gave no audio at all.
- **Why** (`sst/audio.py` before): PortAudio knows only the devices it found when it started, and restarting it
  (`sd._terminate`/`_initialize`) closes every open stream, so Rflow re-read the list only while no stream was open.
  With "Microphone stays ready: while Rflow runs" the dictation stream is always open, and the page's level meter is a
  second one, so the list was never re-read: a headset connected later wasn't in it, choosing it by name resolved to
  nothing, and `_resolve` quietly opened the old default instead. Many laptops switch the built-in microphone off
  when a headset is plugged in: that stream then delivered silence, with no error and no check.
- **Now, in loosely coupled parts:**
  - `sst/devices.py`: Windows' own list (Core Audio's MMDevice API through ctypes, read-only): `snapshot()` = the
    microphones that can record and the default one, ~1 ms (the first call ~200 ms, COM starting); `all_capture()`
    with unplugged and disabled ones. Its names are PortAudio's WASAPI names.
  - `audio.refresh_devices()`: when Windows' list differs from the one PortAudio last read, the open recorders and
    meters (`_live`) are paused, PortAudio restarted (~65 ms) and they are opened again, each on the device it should
    have now. Never while one is recording (`busy`); a `LevelMeter` is never busy. Without Core Audio, the old rule.
  - `Recorder`: `start()` re-reads first if the devices changed (so "Windows default" follows a headset plugged in);
    `tick()` every `WATCH_SECONDS` (2 s) does the same while idle; `healthy` is False when no audio block came for
    `STALL_SECONDS` (1.5 s): such a stream is opened again (at the key press, or by `tick()`). A microphone chosen by
    name and not connected records on Windows' default, and `notice` says so (shown once per change, as a Windows
    notification, through `Dictation.on_notice`); when it's back, it's used again.
  - `Dictation._no_sound`: a recording whose loudest sample is under `SILENT` (-80 dBFS) isn't transcribed; the error
    names the microphone and says where to choose another (both the pipeline and the classic way).
  - `MicrophoneBox` (AI & models, and the first run): "Windows default (now: …)", the list asked again every 2 s while
    shown (`source`), the choice kept, "Not connected now: Rflow uses … until it is." for an unplugged choice, and
    "Hearing: …" for the microphone the meter opened.
- **Checked on this laptop's real microphone** (Realtek, 48 kHz): the always-on recorder and a meter open, a forced
  re-read closes and reopens both, audio flows after, and a recording right after works. Windows remembers eight
  headsets here (AirPods Pro, Razer Barracuda X, OnePlus Nord Buds 3r, Beats Fit Pro, Airdopes 141, EarPods), most
  of them Bluetooth "Headset" (hands-free) microphones: call quality, and the warning on the page says so.
- **Owner to try:** with Rflow running, connect a headset → AI & models shows it within 2 s → dictate with "Windows
  default" and with the headset chosen → unplug it and dictate (Rflow uses the laptop's microphone and says so).

## Live captions (phase 27, the owner's idea of 2026-10-05)

- **The owner's rules:** "do not merge with the current pipeline": live translation flows separately. First (1) what
  you *hear* (mostly Japanese → English in meetings), then (2) your own speech (English → Japanese). This is (1).
- **Why not the dictation pipeline:** it cuts at pauses or at 20 s, then cleans up once at the end: fine for text you
  type, far too slow for captions. Live translation needs 100 ms audio slices, partial results and a translation that
  starts before the sentence ends. Japanese puts the verb last, so even human interpreters trail by ~2-5 s. The
  research (not in git: `reports/Live speech translation.md`, `research_notes/Live speech translation/`) compared
  cascades (streaming recognition + translation) with end-to-end models; Gemini 3.5 Live Translate (preview) does
  both in one connection, with the Gemini key the AI connection already keeps: option A, built first.
- **The flow** (`sst/live/`, never importing `sst.pipeline`, `sst.dictate` or `sst.audio`; a test checks):
  - `wasapi.py`: the default output device through WASAPI loopback (ctypes COM, read-only), the mix format (48 kHz
    stereo float here) → mono → 16 kHz (box filter + interpolation, seamless across packets) → 100 ms PCM16 frames.
    Windows sends nothing while nothing plays, so silence is filled in by the clock; a new default output (headphones)
    is noticed within 2 s and opened. Checked here: a 440 Hz tone played was captured as 440 Hz.
  - `gemini.py`: `GeminiLiveTranslate`, one websocket (`websockets`, sync client) to
    `BidiGenerateContent` with `translationConfig.targetLanguageCode`; sends the frames, receives the words heard
    (`inputTranscription`) and the translation (`outputTranscription`), joins pieces into lines (a pause of 1.5 s or a
    sentence end), reconnects before Google's ~10-minute limit and on `goAway`, gives up only on a refused key, and
    says problems in plain words (the key never in a message). Its spoken translation (audio) is ignored.
  - `session.py` (capture → engine → events; logs how far translations trailed the words: median and maximum),
    `transcript.py` (each session a two-language text file in the profile's `live captions` folder, made with the
    first line), `captions.py` (the caption bar: the words heard small, the translation large, finished lines dimmer;
    never takes focus, clicks go through, `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)` so Teams/Zoom shares and
    recordings don't show it, checked here; `LiveCaptions` moves the engine's events to Qt's thread).
  - The app: Tools has a **Live captions** card (Start/Stop, the language, hide from sharing, the transcripts), the tray
    menu a **Live captions** check item. Settings: `live_target`, `live_hide_from_share`, `live_told`. The first start
    asks once: the audio goes to Google, about $2.20 an hour with a paid key, a free key's audio may be used by Google.
  - `sst live [--to ja] [--seconds 60]`: the same pipeline in a console, each line printed with its lag.
- **The owner's runs against Google (2026-10-05, `sst live`, a Japanese conversation video → English):**
  1. Refused: Google's Live Translate guide puts `inputAudioTranscription`/`outputAudioTranscription` inside
     `generationConfig`; the server closes with 1007 "Unknown name". The API reference has them on the setup: fixed.
     Errors that retrying can't fix (refused request, key, model) now stop the captions with the reason.
  2. Working, good translations. Fixed after it: a long line cut at the translation's full stop split the Japanese
     mid-word (the translation runs ahead of the transcript); the "s behind" number compared Google's two texts
     (0.2 s, meaningless); Japanese pieces had spaces between them.
  3. Lines whole, but matching sentence counts drifted once one Japanese sentence became two English ones. Now a long
     line ends when both texts are at a sentence end; a slip stays in one line.
  - The lag shown is "complete X s after the voice paused", from the loudness of the frames sent, and only when the
    voice paused (a fast conversation rarely does: few numbers, none of them wrong). First measured: 1.0 s.
  - **Every start waited 10 s and failed once:** the first of the eight addresses `generativelanguage.googleapis.com`
    gives never answers from this laptop's network (5 s timeout there, 0.03 s for the other seven), and Python waits
    out the whole timeout on it. `gemini._socket` gives each address 2 s: the first connection takes ~2.2 s. This is
    likely also the "first connection stalls" in Known limitations (GitHub, the gateway): not checked yet.
  4. Clean: started at once, lines whole; one more fix ("parents'house": Google drops the space after a plural
     possessive). The English for the last sentence or two can still arrive after the Japanese and open the next
     line: Google sends no timings to align them by; the slip doesn't build up.
- **The owner tried the caption bar in the app** (Tools → Live captions → Start, over the video): "it worked well".
- **Next:** part 2 (the microphone, English → Japanese, without captioning the meeting's own audio twice); option B
  engines (local streaming recognition, e.g. Nemotron in sherpa-onnx or Soniox, plus clause-by-clause translation)
  if Gemini's preview is too slow, too costly or goes away.

## Live translation, a section of its own (phase 28, the owner's redesign of 2026-10-05)

- **How it came about:** part 2 first added "Translate my speech too" to the Tools card (the owner's choice "both
  ways in one bar"). The owner then asked for live translation as a section of its own, more visible and easier to
  use: a shortcut, a bar to move, resize and scroll back through with a ✕, and two pipelines to choose between:
  the computer, or the microphone in real time. Their choices: the microphone translates into **one** language;
  **Both** is kept as a third source for online meetings; the shortcut is **Ctrl+Alt+L**.
- **The section** (`LivePage` in `window.py`, between Tools and AI & models; its icon is `live` in `theme.ICONS`):
  Start/Stop with the shortcut drawn as keys; "Listen to" Computer / Microphone / Both with a line on each and only
  the languages that source needs ("Computer sound into", "Microphone into" or "Your speech into"); the shortcut
  (Ctrl+Alt+L, Ctrl+Shift+L, Win+Alt+L or off); "Hide the bar from screen sharing" (applied at once); the past
  sessions (date, lines, Open; "Open the folder"); the cost. The live card left Tools.
- **The shortcut** (`TrayApp._start_live_shortcut`, a `HotkeyListener` polled by `live_keys`): a press toggles live
  translation. Translate also offers Ctrl+Alt+L: when two tools have the same keys, live translation steps aside
  (`live_shortcut_clash`) and the section says which tool has them. The tray item shows the keys.
- **The bar** (`CaptionBar` in `captions.py`, rewritten): a header (what's translated into what, the status) with a
  ✕ that stops live translation; a `QTextBrowser` with the whole session (up to 2,000 lines; the transcript has them
  all), each line the words heard (small) above the translation; only the lines in progress are redrawn
  (`_tail_at`), so reading back isn't disturbed, and it follows new lines only while at the bottom. Drag anywhere to
  move it, 8 px from an edge to resize (`dragged()`, done in Qt so tests can drive it with QTest); the geometry is
  saved (`live_bar`) and used at the next start if it's still on a screen. Not click-through any more (it has to take
  the mouse), but still never activated (`WS_EX_NOACTIVATE`): checked on a real window, the app in front kept focus.
- **The pipelines:** `LiveConfig.source` → `SOURCES` → the ways (`SYSTEM`, `MIC`); the session starts each, a way that
  can't start is said on the bar while the others go on, and the source can change while it runs
  (`LiveCaptions.set_source`). The microphone is Windows' default communications microphone (`Capture.microphone()`,
  checked here: 48 kHz, room sound at -46 dBFS). Its lines that come back untranslated are dropped (speech already
  in that language; with Both, the meeting heard through the speakers). Only with Both are its lines marked "You"
  (on the bar and in the transcript); with the microphone alone it hears everyone in the room.
- **Settings:** `live_source`, `live_mic_target` (Japanese), `live_shortcut`, `live_bar`; the folder and file names
  stay `live captions` (2.1.0's), so earlier sessions are listed too.
- **`sst live --source computer|microphone|both [--to en] [--mic-to ja]`.**
- **Owner to try:** the section (Start, the shortcut from another app, moving, resizing and scrolling the bar, ✕);
  Microphone with someone speaking Japanese or English near the laptop; Both with headphones in a meeting; a Teams
  share with "Hide the bar…" off.

## Live translation spoken aloud (phase 29, the owner's request of 2026-10-05)

- **The owner's request:** "the transcript and then lively talk also, by enabling the option", with Piper's
  `en_US-danny-low` (they pasted the rhasspy/piper-voices links). The Piper program itself moved to
  `OHF-Voice/piper1-gpl` (GPL); Rflow doesn't need it: sherpa-onnx, already in Rflow for Parakeet, runs Piper voices.
- **Measured first** (scratch scripts, not in git):
  - sherpa-onnx reads a Piper voice's settings from the model's metadata; rhasspy's keep them in the .onnx.json. An
    ONNX model is a protobuf ModelProto, and protobuf merges repeated fields that come later in the bytes, so the
    metadata entries (field 14) are appended to a copy: no ONNX library. Loaded in 1.1 s; speaks ~20x faster than
    real time (2.8 s of speech in 0.13 s).
  - espeak-ng-data is 355 files (17 MB) for every language; English needs 6 (795 KB) for byte-identical speech
    (phontab, phonindex, phondata, intonations, en_dict, lang/gmw/en-US).
  - Windows' process loopback (`ActivateAudioInterfaceAsync` on `VAD\Process_Loopback`, excluding Rflow's process
    tree, a COM completion handler made in ctypes) records what the laptop plays without Rflow's own sound, already as
    16 kHz mono: another program's 440 Hz tone 93.4, Rflow's player's 1000 Hz tone 0.2.
- **The pieces** (`sst/live/`):
  - `voice.py`: `DANNY` (rhasspy/piper-voices at `c10ece1a`, 63 MB, MIT; espeak-ng-data from the sherpa-onnx author's
    copy at `9b9d4d57`), downloaded through `sst.downloads` (pinned, checked; it now makes subfolders), then
    `prepare()`: model.onnx with metadata, tokens.txt, the original removed. `PiperVoice.synthesize(text, speed)`.
    Lives in `DOWNLOADS_DIR/piper-en_US-danny-low` and `espeak-ng-data-en`.
  - `speaker.py`: `Speaker.hear(event)` takes TRANSLATION/LINE events of the ways it speaks; a sentence is said once
    the text goes on past it, when its line ends, or 0.6 s after a full stop with nothing new ("3." waits for "3.5";
    "e.g.", "Mr." don't end one). Behind: faster by 0.15 per sentence waiting (up to 1.6x); a sentence waiting over
    10 s while newer ones queue is skipped (on screen and in the transcript). `speaking` lasts 0.4 s after the audio.
    Measured: a sentence was heard 0.12 s after the translation went on past it.
  - `wasapi.py`: `Player` (WASAPI render, 16-bit mono with AUTOCONVERTPCM; follows the default output; `private` from
    the endpoint's form factor: headphones/headset/handset), `_ProcessLoopback` (tried first by `Capture.speakers()`;
    older Windows fall back to the whole mix and `hears_self` is True).
  - `session.py`: `set_speaker()`; each way's frames become silence while the voice speaks where that way hears it:
    the microphone unless the output is private, the computer only when `hears_self`. Echo lines never reach the
    speaker (dropped first).
  - What's spoken (`LiveConfig.spoken_lanes`): what's translated into the voice's language; with Both only the
    others' words. Danny speaks English: translating into Japanese shows only text (the section and the button say so).
- **The app:** "Speak the translation" and Speed (Normal, a little faster, faster) on the Live translation section;
  the bar's speaker button does the same; the first time it's switched on the voice downloads in the background, the
  bar's title shows the percent, and it starts speaking when done. Settings: `live_speak`, `live_speak_speed`,
  `live_voice`. `sst live --speak`.
- **Owner to try:** a Japanese video, Computer into English, speaking on: the voice should follow the captions, and
  the captions must not pick up the voice. Microphone into English through the laptop's speakers: the mic pauses
  while it speaks.
- **Later, if wanted:** Gemini Live Translate already returns the translation as speech in any language (Rflow
  ignores it): that would speak Japanese too, with no download.

## Never lose work (phase 30, after the user testing of 2026-10-06)

| Flaw | Fix |
|---|---|
| **B-01** a dictation thrown away when any other key was touched while Ctrl+Win was held | The matcher keeps the hotkey held after another key ("interrupt"), so the release still comes. `Dictation`: another key within `SHORTCUT_SECONDS` (0.6 s) is a Windows shortcut (Ctrl+Win+D) and dropped quietly as before; later, the dictation goes on. The tester's own script (30 s, Shift brushed) now types and saves it |
| **B-02 / N-11** a damaged `dictionary.db`, or a settings folder that can't be written, stopped every start | `DictionaryStore.open()`: a damaged file is moved aside (`dictionary.damaged-<time>.db`, with its -wal/-shm) and a new one started (Your words come back from the settings); an unwritable folder gives a store in memory. The user is told (`TrayApp._tell`, after the tray icon is up). History writes can't fail a dictation either |
| **M-14** crashes left no trace | `install_crash_handlers`: `sys.excepthook` and `threading.excepthook` log uncaught errors; `faulthandler` writes native crashes to `logs/crash.log`; a `running` marker (removed at quit and when Windows signs out) tells a crash from a shutdown. The next start keeps the report as `crash-<time>.log` and says so. A start that fails shows why (`start()`) instead of vanishing |
| **N-10** a damaged settings file reset everything | Every save keeps the previous file as `settings.json.bak`; a damaged one is kept aside and the copy used; the user is told (`settings.LOAD_PROBLEMS`). A save that fails is said once |
| **N-33** history trimmed in place | A new file, then a swap |
| **M-19** a failed paste lost the dictation | `Dictation._type`: the text goes to the history and the recording is saved either way; the pill says "Couldn't type it here (…). It's on Home, ready to copy." |
| **M-22** Win+Ctrl+V instead of a paste | `paste._wait_until_keys_released`: waits while the keys are held (also for the next dictation, up to `HELD_WAIT`), and never presses Ctrl+V with them held |
| **M-15** "Update now" cut dictations and live translation | `TrayApp.busy()`; a ready update waits ("it installs when this dictation is done / live translation stops") and installs when idle |
| **M-21** (suspected) Windows dropping a slow hook | `HotkeyListener` puts in a fresh hook every `REHOOK_SECONDS` (30 s) while no key is held, the new one before the old one goes |
| **The owner's report, 2026-10-06:** "make it concise" after a dictation said "Select some text first" | The log showed why: the last dictation was found but the app had changed it a little ("11 expected, 9 copied"), and the exact match failed. `textaccess.select_last` now returns the text it selected: exact first; else it looks `LOOK_FURTHER` (12) more steps back, finds the closest text ending at the caret (`CLOSE_ENOUGH` 0.85, difflib, never starting on a space) and selects exactly that. The message now tells "couldn't find your last dictation here" apart from "nothing to rewrite", and the log names the window class |
| crash.log written while nothing crashed | On Windows faulthandler also writes exceptions Windows raises and handles itself (`0x8001010d`, COM, seen on the owner's laptop): `crashed()` counts only real ones (access violation, stack overflow, a fatal Python error) |

## Clearer sections, setups and costs (phase 31, the owner's UI review of 2026-10-06)

The owner went through every page and asked for it before the remaining bugs. Built by five agents in parallel, each
in its own worktree with a strict scope, then merged one pull request per part (the owner's rule 8 in CLAUDE.md):

| Part | Issue | What |
|---|---|---|
| Live translation | #82 | A big green Start (`ui.PowerButton`); while running a red Stop, a blinking "Live" light (`BlinkLamp`, only while shown) and what it translates into what (`TrayApp.live_languages`); Speak the translation right under it |
| AI & models | #83 | Your API keys at the top (`TrayApp.save_key` / `save_server` never change the cleanup or speech model: M-04); the speech model in two groups (On this PC, Cloud tiles: OpenAI, Groq, Gemini, Your own server); long microphone names elided (M-03) |
| Sections | #84 | Home, Live translation, Words, Snippets, Text Transform, Translate, Formatting, AI & models, Settings, Report a problem; Tools removed (`go_to("tools")` opens Text Transform); before/after examples checked by the real `TransformGuard`; Formatting's examples run through the real formatter; Start over (`TrayApp.start_over`, `erase_data`: only folders named `sst`; Rflow restarts with `--after=<pid>`) |
| Models | #85 | From the price research: Groq's `llama-3.1-8b-instant` is gone (`openai/gpt-oss-20b`, reasoning low); Claude 4.7+ get no temperature; gpt-6 is a reasoning model; Gemini 3 Flash thinks at minimal/low; `gpt-transcribe` added (the old OpenAI speech models shut down 2027-02-26, `cloud.retirement`); a cut-off answer is an error, never typed with a hole (`sst/modelrules.py`) |
| Setups | #86 | `sst/setups.py` and `sst/costs.py`: Recommended (Parakeet + `gemini-3.5-flash-lite`, about $1.05 a month), Fastest (Groq, $0.70), Multilingual (Gemini Transcribe, $3.58), Local (free; AI on the PC "coming soon"), Custom; the first run starts with them; every model list shows its monthly cost, amber $2-5, red over $5 or 10x the cheapest |
| Tests | #87 | The `tray_app` fixture no longer touches the real log folder (it removed a running Rflow's crash marker) or the real data folders |
| Docs, site | #88 | README's ten sections, the website (open source, setups with prices, live translation, new screenshots), version 2.3.0 |

**Still to check with real keys** (the code follows the providers' pages; tests use fakes): Groq gpt-oss-20b with
`reasoning_effort: low` and `include_reasoning: false`; Gemini's `reasoning_effort` minimal/low through the
OpenAI-compatible endpoint and `thinkingConfig.thinkingLevel` for Flash speech; OpenAI `reasoning_effort` none on
gpt-6 and `languages[]` on gpt-transcribe (Tamil, Japanese); Claude Sonnet/Opus 5.5 with no temperature. The cost
estimates assume 20 minutes of dictation a day (150 dictations), 22 days a month.

**Decisions made for the owner** (to confirm): Start over keeps the downloaded models by default; the setups sit at
the top of AI & models (not a sidebar entry); Formatting is a section of its own; Your own server's address shows in
plain text in Your API keys (only keys are masked). **Open:** the owner said "the sound is gone" about Settings:
ask whether the start/stop beep stopped. Not done: Home's own long microphone label (the other half of M-03).

## Live translation at any volume (phase 37, the owner's report of 2026-10-07)

- **The owner's report:** with the voice on, "if I lower the system's audio nothing is able to hear"; asked, they said
  the translation itself stops, and chose interpreter style for the voice: the original lowered while it speaks,
  back between sentences, how low set by a slider (issue #108).
- **Measured first** (scratch scripts, not in git):
  - Gemini isn't the limit: the same English clip (Windows' Zira voice) sent at falling levels was translated in full
    at -20, -40, -50, -60 and -70 dBFS (speech RMS), and not at all at -80.
  - Rflow's capture was: process loopback was asked for 16-bit. The playing app's own volume (a player's slider, the
    volume mixer) lowers what loopback hears on every PC (here: 50/25/10/5/2 % arrived 4/7/13/21/30 dB lower; this
    Surface's audio effects compress it). Windows' own volume doesn't reach the loopback on this Surface (read only:
    -21 dB set, the capture as loud as ever), but on PCs whose driver applies it first, a low volume does the same.
    End to end, an app at 0.03 % (another process playing the clip): before, frames at -81 dBFS and Gemini heard
    nothing; after, -20 dBFS and all 48 words translated (at 0.3 % both worked, but before misheard "Osaka").
  - Process loopback gives 32-bit float (16 kHz mono) when asked. Lowering another process's session volume works
    from Rflow, and Rflow's own voice stays out of it.
- **The pieces** (`sst/live/`):
  - `wasapi.py`: `_ProcessLoopback` asks for float first (16-bit where refused); `Capture.speakers()` has `level=True`:
    each packet goes through a `Leveler` before the 16-bit frames are made. With nothing heard for 3 s and Windows'
    output muted (read only, `IAudioEndpointVolume`), `on_problem` says "Windows' sound is muted…" (the session shows
    it on the bar like an engine's problem). The log names the format ("32-bit float").
  - `level.py`: `Leveler`: the loudest 20 ms of the last 3 s heard below -20 dBFS is raised to it (up to +60 dB); at a
    normal volume the sound is left exactly as it is; down at once, up 20 dB a second at most, a soft limit above 0.9.
  - `ducking.py`: `Ducker(depth)`, started and stopped by the `Speaker`, watches `Speaker.speaking` every 20 ms: the
    other apps playing on every active output (session volumes, `IAudioSessionManager2` / `ISimpleAudioVolume`;
    Rflow's own process and system sounds left out) go to their own volume x depth in 0.15 s, and back 0.2 s after
    `speaking` ends (itself 0.4 s after the audio) over 0.6 s. An app that starts during a sentence is found within
    0.5 s; an app whose volume the user changes meanwhile keeps the user's. Windows remembers mixer volumes, so what's
    lowered is written to `%LOCALAPPDATA%\sst\lowered-apps.json` first and `put_back_left()` (at Rflow's start, and
    when a Ducker starts) puts it back after a crash; apps not running then are kept 7 days. Tried on this laptop on
    a test tone's own process: 1.0 -> 0.25 while "speaking", back after, the record gone.
- **The app:** Live translation → Speak the translation → "Other sounds while it runs": a slider (`ui.Slider`, new:
  the switch's inset track, Iris fill, round knob) from 10 % to 100 % ("Unchanged"), 30 % by default, saved when let
  go (`live_duck`), applied at once while it speaks (`LiveCaptions.set_duck`). `sst live --speak` uses it too.
- **Owner to try:** a video, Computer into English, speaking on: the video gets quieter while Danny speaks and comes
  back after; turn the video's own volume (or Windows') right down: captions go on. Mute Windows' sound on the other
  PC: the bar says so.

## Known limitations
- Apps running as administrator don't receive the text, because Windows blocks input from normal programs into them.
- Ctrl+Win isn't sent through the real hook in automated tests (Wispr Flow on the dev laptop would react). The hook
  plumbing is tested with the Menu key, Esc and Ctrl+Alt+X, and Ctrl+Win by the Matcher unit tests. The owner
  confirmed that Ctrl+Win dictation works in the installed app.
- English only. The cleanup can't bring back words the recognizer dropped, and it leaves real words that are wrong
  in context ("charted" for "chatting", "cloud" for "Claude").
- The console command (`sst dictate`) has no cleanup; the tray app does.
- The installer isn't code-signed, so SmartScreen warns on the first install. In-app updates don't trigger it, because
  a file downloaded by the app isn't marked as coming from the internet.
- The Python client's first connection to the company gateway, and sometimes to GitHub, stalls on the dev laptop. The
  clients use short connect timeouts with retries (`sst/gateway.py`, `sst/updates.py`, `sst/engines/cloud.py`).
- A cloud speech model sends the voice to the provider; the window says so and asks before one is used.

## Next steps

1. **Owner: update to 2.4.0 and try phase 37** on the PC where the translation stopped: a video, speaking on; the
   video should get quieter while Danny speaks and come back after; turn the video (or Windows) right down and the
   captions should go on. If the original should go lower or stay louder, the slider is under Speed.
1. **Owner: update to 2.3.0 and try phase 31**: the first run's setups (on another Windows account, or after Start
   over), AI & models (Your API keys, a setup, a model with its cost), Live translation's Start/Stop, the new
   sections. Then the real-key checks listed in **Clearer sections, setups and costs**.
2. **Then the user testing's phases 32-36** (privacy first), and announce Rflow as open source as 3.0.0. With the
   owner's OK: the repository's description and topics, private vulnerability reporting (SECURITY.md points to it)
   and Discussions. The accounts layer (closed source) after that.
1. **Owner: update to 2.0.0 from the banner and look around** (both themes: Windows Settings → Personalization →
   Colors):
   - Home: dictate, then search a word with Ctrl+K; hover a dictation: Correct and Copy
   - Words: type part of a word to find it; add one with Enter; remove one with its ✕ and Undo
   - Tools: try Concise and Translate in the box; select text in another app, Ctrl+C twice, then press C (copied) or
     Enter (replaced)
   - AI & models: Test the AI connection; switch the cleanup off and on
   - make the window as small as it goes: the sidebar becomes a rail
   - on another Windows account (or after deleting `%APPDATA%\sst\settings.json` there), the three-step first run
1. **The UX review of the other pages** (found while doing phase 23; the owner chooses the order):
   - Text Transform: long phrase lists are cut off in one-line boxes
   - Settings, Text Transform and Translate apply each toggle at once with no word that it was saved
   - the speech language list has 17 languages plus Automatic; Whisper and the cloud models know about 99, and the
     list is searchable now
   - a long dictation with a cloud model waits ~6 s after the release: a short rest re-sends the whole part before it
   - the log (`sst.log`) holds every dictation's text at INFO, even with the debug folder off
2. **Owner: update to 1.5.0 and try it** (released 2026-10-01; 91 MB, SHA-256 `c05c7c81...` checked by the updater):
   - Open the installed 1.4.0: the banner offers 1.5.0 (or Settings → Check for updates) → Update now. Expected: no
     Parakeet download (it stays next to the program), the same settings, words and history.
   - Speech recognition → Whisper turbo → "Download and use (1.6 GB)"; dictate in English, and in Tamil with the
     language set; switch back to Parakeet; press "Scan my computer".
   - Cloud → a provider you have a key for (Groq has a free tier) → paste the key → Test → Use this model → dictate;
     then turn Wi-Fi off and dictate: Parakeet should type it, and say so.
   - Your own server → it is filled in from AI cleanup's server, or enter the company gateway → Load models →
     whisper-1 → Test → Use this model → dictate.
   - Later, the new-install path: on another Windows account, install from the website; the welcome asks how to
     recognise speech → Download Parakeet → dictate when it's done.

   The release was made with the stacked-merge recipe: #35 into `main`, then #37, #39, #41 and #43, each retargeted
   to `main` first; CI green on `main`; then the tag. This laptop's global git config signs tags
   (`tag.gpgsign`), so a tag needs `-m`.
2. **Owner: try Translate** (phase 22, released as 1.8.0: update from the banner): select a sentence in Edge or Teams → Ctrl+C twice → the popup shows it in
   English (set Japanese as the second language on the Translate page for English text); pick another language at
   its top; Replace; Esc.
3. **Owner: try Snippets** (phase 21, released as 1.8.0): Snippets page → add "my email" → your email; dictate "my email"; add
   "insert my signature" with two lines and "Also inside a sentence", then dictate "thanks, insert my signature".
4. **Owner: try Text Transform** (phase 20, released as 1.7.0: update from the banner): Text Transform page → Try
   it; then in Teams or Notepad:
   - dictate a paragraph, then hold Ctrl+Win and say "make it concise"; then say "undo that"
   - select a paragraph, hold Ctrl+Win and say "bullet points"
   - select text → double-tap Ctrl → 1-4; open the menu, click into another window, then press 1: the text's
     window comes back and the text is replaced (or the pill says "press Ctrl+V")
   - edit a phrase on the page ("trim it" for Concise) and say it
3. **Owner: try the voice pipeline** (phase 19, released as 1.6.0: update from the banner; the download is the
   ~90 MB installer, nothing else):
   - dictate a long paragraph with pauses: the text should arrive soon after you let go
   - say "twenty five percent" and "October first at three thirty pm"
   - add a sound-alike on the Dictionary page
   - correct a dictation twice with ✎ on Home and accept the suggestion
   - with AI cleanup on, try a sentence with a number; turn "Keep each dictation's steps" on for a while and look at
     `%LOCALAPPDATA%\sst\debug`

   Then tune with the debug folder (plan §108: audio, VAD, chunk boundaries, ASR, merge, dictionary, formatting, LLM,
   guard, in that order).
3. **The building-block plan is built** (phases 13-18; see **Speech recognition as a building block**). Possible
   later: Whisper's runtime on demand too (installer ~50 MB), and the website's screenshots of the new pages.
3. **The owner, meanwhile:**
   - Keep only names and terms in Your words.
   - Read a set with Settings → "Turn off Windows' voice effects" on (raw mode). Set B in Windows mode was read on the
     second laptop on 2026-10-01.
   - On the second laptop, the other laptop's 150 recordings are missing: copy the data zip (see **Start here**).
4. **Correction, after the building blocks, chosen by what goes wrong in real dictation** (see **Correction
   groundwork**):
   - names still wrong → the sound-alike fixer with an ordinary-word guard
   - small words wrong → AI cleanup only when unsure (Ollama locally, or the owner's provider)
   - both mediocre → compare a stronger model (Qwen3-ASR or Canary via sherpa-onnx) on the same recordings
   - later: learn from the owner's edits after pasting
5. **A small fix:** in the developer copy, the update banner should say "you run Rflow from its source: update with
   git pull; download the installer for the app", not offer "Update now" (`TrayApp.start_update`, `sst/app.py`).
6. **Then the features paused for accuracy:**
   - text without the AI endpoint: fillers, "new line", spacing and capitals from the text before the cursor,
     snippets
   - command mode ("make this formal" on selected text) and a style per app
7. **Later:**
   - code signing (removes the SmartScreen warning)
   - a "paste last transcript" hotkey
   - the bench: resampling whole sessions, and substitutions / deletions / insertions counted apart
8. With a key for OpenAI, Anthropic, Gemini or Groq: press Test once on the AI cleanup page, and for OpenAI, Gemini
   or Groq on the Speech recognition page's Cloud tab (they were only tested against the local fakes).
9. Waiting on the owner:
   - the apps used most
   - code-signing budget
   - the git history question above
