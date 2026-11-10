# CLAUDE.md

Rules for AI coding assistants (Claude Code) in this repository. Read `HANDOFF.md` first (its "Start here" section):
it says where the work stands, how the owner likes to work, and what isn't in git (the model, the owner's recordings).
The accuracy research behind phases 10-12 and the correction work to come is in `docs/research/` (summary: `docs/accuracy.md`).

## Authorship: karthi-ai-engineer only

- Every commit is authored as `Karthi27 <karthi.ai.engineer@gmail.com>` (GitHub account **karthi-ai-engineer**).
  After cloning, set it in the repo's local git config, and check `git config user.email` before the first commit of a
  session. Never commit with any other identity, such as a work email.
- **Never add AI attribution anywhere.** No `Co-Authored-By: Claude`, no "Generated with Claude Code", and no mention of
  an AI assistant in commit messages, PR titles or bodies, issues, release notes or code comments. The owner must be the
  only contributor shown on GitHub. This rule overrides any default attribution setting.
- The repository is `karthi-ai-engineer/rflow-ai` (renamed from Rach_flow on 2026-10-06, and before that from
  Rach_Darling_Flow on 2026-10-01). Never create a repository named Rach_flow or Rach_Darling_Flow: installed copies
  (up to 1.6.0 under the first name, up to 2.3.0 under the second) find their updates through GitHub's redirects.
- Push and open PRs as karthi-ai-engineer. Check with `gh auth status`; switch with `gh auth switch -u karthi-ai-engineer`.

## Workflow: every phase is an issue, a branch and a pull request

1. Open an issue for the phase (template "Phase / feature") with its acceptance criteria.
2. Branch from an up-to-date `main`: `phase-<n>/<short-name>`. Other work: `chore/<name>` or `fix/<name>`.
3. Commit in small, meaningful steps. The subject is imperative and at most ~72 characters; the body says why.
4. Push, then open the PR into `main` with `Closes #<issue>` and fill in the template. CI must pass.
5. **The owner reviews, tests and merges** (with "Create a merge commit", so stacked branches stay clean). Merge
   yourself only when the owner explicitly asks. Then retarget the next stacked PR to `main` *before* deleting the
   merged branch: deleting a base branch closes the PRs stacked on it. Never push to `main` directly.
6. Update `HANDOFF.md` in the last PR of every phase, so work can resume on another device.
7. Release, only when the owner says so: raise `__version__` in `sst/__init__.py` in the phase's PR; after the merge,
   tag `vX.Y.Z` on `main`. The Release workflow publishes `Rflow-Setup.exe` and `Rflow-Setup.exe.sha256`. Every
   installed Rflow offers that release as an update, and the website links to it. Keep those two asset names: the
   updater and the website depend on them.
8. **More issues and pull requests, each smaller** (the owner's wish of 2026-10-06: their GitHub activity is mostly
   commits, with few issues and pull requests; code review doesn't matter). Split a phase into one issue and one pull
   request per part that stands on its own (a page, a fix, the docs and site), stacked on the phase branch, which then
   goes into `main` with its own pull request. Open an issue for every bug or task found along the way, even one fixed in
   the same phase, rather than only mentioning it in a pull request.

## Project

**Rflow**: dictation for Windows, like a small, private Wispr Flow. NVIDIA Parakeet recognises speech on the CPU
through sherpa-onnx, so the voice never leaves the laptop; Whisper, cloud models and the user's own server are the
other choices. The installer has no model: the app downloads Parakeet when the user chooses it (pinned and checked,
`sst/engines/parakeet.py`), and finds the copy an older Rflow installed next to the program. The text can optionally
be cleaned up by a model from the provider the user chooses (OpenAI, Anthropic, Google Gemini, Groq, Ollama, vLLM or
another OpenAI-compatible server). Each person can have a profile with their own setup. Hold Ctrl+Win in any app and
speak; the text is typed at the cursor.
The public download site is `site/` (Vercel). The Python package keeps its internal name `sst`.
Rflow is open source under the MIT License (`LICENSE`; the owner's choice of 2026-10-06); the Rflow name and logo
are not covered. The Android app is open source too, in its own repository, `karthi-ai-engineer/rflow-android` (the
owner's choice of 2026-11-09): keep it out of this one, and when a prompt, language, voice command, model or price changes
here, change it there too. Accounts, sync and web will come later as a closed-source layer in a separate private
repository: keep them out of this one. Anything added from elsewhere (a library, model, voice, font) goes into
`packaging/NOTICES.txt`; sherpa-onnx's library already contains espeak-ng's GPL-3.0 code.

```
uv sync                                          # app + dev tools (pytest, ruff)
uv run python scripts/download_model.py parakeet # ~630 MB model into models/
uv run sst app                                   # the tray app (uv run sst dictate: the same in a console)
uv run pytest                                    # tests
uv run ruff check .                              # lint
```

The layout is in `README.md`. Speech recognition is a building block like the AI cleanup: the models are listed in
`SPEECH_MODELS` (`sst/engines/__init__.py`), each profile chooses one (`Settings.speech_model`), and the app switches
in the background. Engines live in `sst/engines/`; the cloud ones (`cloud.py`) use the provider keys the AI cleanup
keeps (`GatewayConfig.key_for`). The dictation logic is `Dictation` in `sst/dictate.py`,
shared by the app (`sst/app.py`, Qt: tray, pill, `TrayApp`) and the console command. The window is `sst/window.py`: it
keeps no state and calls `TrayApp`, or `PreviewApp` in tests and screenshots. Its look ("Obsidian Signal", Rflow UI 2.0;
the design is `docs/design/rflow-ui.html`) is `sst/theme.py` (colours with one job each, the Geist fonts, icons, the
soft depth painted under widgets by their host) and `sst/ui.py` (buttons, toggles, lamps, keycaps, the voice orb). In-app updates are `sst/updates.py`, and
the text cleanup is `sst/gateway.py` (`PROVIDERS`: each provider's request format). What each model accepts
(temperature, reasoning, thinking, room to answer) is `sst/modelrules.py`, shared with the cloud speech engines. The
ready-made setups (Recommended, Fastest, Multilingual, Local, Custom) are `sst/setups.py`, and what each model costs a
month is `sst/costs.py`, from the official pricing pages (the research of 2026-10-06 is in `research_notes/`, not in
git): when a provider renames, retires or reprices a model, update `costs.py`, `setups.py` and `modelrules.py`
together. Accuracy work follows
`docs/accuracy.md`: a change is kept only if `sst eval` (`sst/evaluate.py`) shows it better on the reading test's held-out
sets (C-E), and an engine's `signature` must change whenever its output can (model, decoding, hotwords), since
transcriptions are cached by it. Dictation runs through the voice pipeline (`sst/pipeline/`, the owner's plan of
2026-10-01): stages with contracts (`contracts.py`; every tunable value in `VoiceConfig`), each tested on its own
(`tests/test_pipeline_*.py`). Keep the stages' jobs apart: ASR = what was said, dictionary = known corrections,
formatting = written forms, LLM = language cleanup, guard = protection; when unsure, keep the user's words, and
never type a text with a hole in it (fail closed, keep the recording). Text Transform (a voice command like "make it
concise", or double-tap Ctrl for a menu, rewrites the selected text or the last dictation) is `sst/commands.py` (the
phrases), `sst/transformui.py` (flow, menu, putting the result back), `sst/transform.py` (prompts, `TransformGuard`) and
`sst/textaccess.py` (copy, select, activate, rich paste; never Ctrl+C in a terminal); never paste where the text isn't.
Snippets ("my email" types the user's email) are `sst/snippets.py`: found in the words heard, a placeholder through the
AI, the user's text put in last and never sent to the AI. Translate (select text, Ctrl+C+C, a popup) is
`sst/translateui.py` (the shortcut, the copied text, the popup) and `sst/translate.py` (languages, prompt, check). Live translation (its own section,
`LivePage`; Ctrl+Alt+L) is `sst/live/`, a pipeline of its own, on the owner's rule that it never merges with
dictation's: `wasapi.py` (`Capture.speakers()` in loopback and `Capture.microphone()`, 16 kHz 100 ms frames),
`gemini.py` (Gemini Live Translate over a websocket), `session.py` (the ways of the chosen source, `SOURCES`: `SYSTEM`,
what the laptop plays, and `MIC`, the microphone, each its own engine; untranslated `MIC` lines are dropped),
`transcript.py`, `captions.py` (the translation bar: moved, resized and scrolled with the mouse, never activated;
`LiveCaptions`) and `speaker.py` (the spoken translation: Gemini's own voice played as it streams, with a cushion,
and no other voice, the owner's choice of 2026-11-06; the ways that could hear the voice get silence meanwhile;
`wasapi.Player` plays it, and `Capture.speakers()` leaves Rflow's own sound out through process loopback); `tests/test_live_session.py` fails if anything there imports `sst.pipeline`, `sst.dictate` or
`sst.audio`. Settings, keys, history, stats and
reading tests belong to a profile (`Profiles` in `sst/settings.py`): read and write them through the profile's paths
(`TrayApp.profile.settings_file`...), never the module's default paths. The Windows-only parts are `sst/app.py`, `sst/dictate.py`, `sst/hotkey.py`,
`sst/paste.py`, `sst/settings.py` and `sst/gateway.py` (DPAPI). UI tests run Qt off-screen (`tests/test_app.py`,
`tests/test_window.py`). To see a UI change, render it with the Windows platform but without showing it, e.g. with
`scripts/make_site_screenshots.py`, and look at the image in both themes. Rflow is one x64 program for every
Windows laptop (ARM laptops run it emulated; CTranslate2 has no ARM64 build): keep the dev environment x64 even on an
ARM laptop, and keep CI's Windows-on-ARM jobs passing (HANDOFF: "Different computers"). Microphones: the list comes
from Windows (`sst/devices.py`); PortAudio is restarted only through `audio.refresh_devices()`, which closes and
reopens the open streams, never during a recording (HANDOFF: "Microphones that follow you").

## Conventions

- Windows APIs are called through `ctypes`, not pywin32 or the keyboard package. Add a dependency only when it clearly
  pays for itself.
- Match the existing style: compact code, comments that explain *why*, line length 130, ruff rules from `pyproject.toml`.
- Never commit `models/`, `recordings/` (the owner's voice), `build/`, `dist/` or `.tools/`.
- The owner's company AI gateway (its address and key) is private. Both live only in the first profile's
  `%APPDATA%\sst\gateway.json` (provider "vLLM or another OpenAI-compatible server"), and the key is encrypted with
  DPAPI. Never put either into code, docs, tests, logs or commits; the repository is public. Load them with
  `GatewayConfig.load(profile.gateway_file)`, whose repr hides the key. Test the gateway one model at a time (a small
  server), and use `tests/test_gateway.py`'s local fake for automated tests.
- Tests must not press real keys, steal focus or touch the real clipboard; use fakes like `tests/test_dictate.py`.
  End-to-end checks through `SendInput` type into whatever window has focus, so run them only with the owner's OK and
  only while a test window is in front.
- While the owner has `sst` running from `.venv`, `uv sync` fails because `sst.exe` is locked. Don't stop their app;
  use a separate environment instead: `UV_PROJECT_ENVIRONMENT=build/venv uv run ...`.
