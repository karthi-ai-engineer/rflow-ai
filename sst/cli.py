"""sst: record from the microphone and transcribe it locally.

  uv run sst app                   the tray app: hold Ctrl+Win in any app, speak, let go: the text is typed there
  uv run sst dictate               the same in this console window
  uv run sst start                 record, press Enter to stop, print the text (repeats until you quit)
  uv run sst file <audio.wav>      transcribe an existing WAV file
  uv run sst devices               list microphones (* = default)
  uv run sst eval [<folder>...]    score reading tests (all of them by default); --degrade, --model, --no-cleanup
  uv run sst web                   open a Record / Stop page in the browser
  uv run sst live [--to ja]        live translation of what the laptop plays (Gemini), until Ctrl+C;
                                   --source microphone (or both), --mic-to ja, --speak (aloud too)

Options: --engine parakeet (or whisper-turbo; openai, groq, gemini, server as set up in Rflow)
         --device <number from `sst devices`>
"""
import argparse
import logging
import sys
import time
from pathlib import Path

from sst import __version__
from sst.engines import DEFAULT_MODEL, ENGINES, load_engine, usable
from sst.engines.cloud import CLOUD, REMOTE


def _load(engine_name: str):
    print(f"Loading {engine_name} model...", end=" ", flush=True)
    t0 = time.perf_counter()
    engine = load_engine(engine_name, **_cloud_options(engine_name))
    print(f"ready ({time.perf_counter() - t0:.1f}s)")
    return engine


def _cloud_options(name: str) -> dict:
    """A cloud model's key and model, or an own server's address, key and model, and the language: from the profile in
    use in the app. Without Parakeet to fall back on: a provider's failure is reported, not hidden."""
    if name not in REMOTE:
        return {}
    from sst.gateway import GatewayConfig
    from sst.settings import Profiles, Settings

    profile = Profiles.load().current
    settings, gateway = Settings.load(profile.settings_file), GatewayConfig.load(profile.gateway_file)
    if name not in CLOUD:
        address, key = gateway.speech_server()
        if not address:
            raise SystemExit("No speech server: set it up on Rflow's Speech recognition page first.")
        return {"language": settings.speech_language, "api_key": key, "model": settings.speech_server_model, "url": address}
    if not (key := gateway.key_for(name)):
        raise SystemExit(f"No {CLOUD[name].name} key: enter it on Rflow's Speech recognition page first.")
    return {"language": settings.speech_language, "api_key": key, "model": settings.speech_cloud_models.get(name, "")}


def _transcribe_and_report(engine, audio, rate) -> str:
    duration = len(audio) / rate
    t0 = time.perf_counter()
    text = engine.transcribe(audio, rate)
    took = time.perf_counter() - t0
    print(f"\n  Text: {text or '(nothing recognised)'}")
    print(f"  [{duration:.1f}s of audio transcribed in {took:.2f}s by {engine.name}]")
    return text


def cmd_start(args) -> None:
    from sst.audio import record_until_enter, save_recording  # the audio modules start PortAudio: not for `sst app`

    engine = _load(args.engine)  # load before recording so there is no wait after you stop talking

    while True:
        print("\n● Recording... speak now, then press Enter to stop.")
        audio, rate = record_until_enter(args.device)

        if len(audio) < rate * 0.3:
            print("  Too short, nothing to transcribe.")
        else:
            peak = float(abs(audio).max())
            if peak < 0.01:
                print("  Warning: almost silent. Check that the right microphone is selected (`sst devices`).")
            text = _transcribe_and_report(engine, audio, rate)
            print(f"  Saved: recordings\\{save_recording(audio, rate, text)}.wav / .txt")

        if input("\nPress Enter to record again, or type q then Enter to quit: ").strip().lower() == "q":
            break


def cmd_file(args) -> None:
    from sst.audio import load_wav

    path = Path(args.path)
    audio, rate = load_wav(path)
    engine = _load(args.engine)
    _transcribe_and_report(engine, audio, rate)


def cmd_devices(args) -> None:
    from sst.audio import list_input_devices

    print("\n".join(list_input_devices()))


def cmd_dictate(args) -> None:
    from sst.dictate import run

    run(lambda: _load(args.engine), hotkey=args.hotkey, device=args.device, save=not args.no_save)


def cmd_live(args) -> None:
    """Live translation in this console: what the laptop plays, the microphone, or both (--source), line by line, with
    how far each line trailed the words. The same pipeline as the app's translation bar (sst.live), with the Gemini key
    of the profile in use."""
    import threading

    from sst.gateway import GatewayConfig
    from sst.live.contracts import LANGUAGES, MIC, Kind, LiveConfig, language_name
    from sst.live.gemini import GeminiLiveTranslate
    from sst.live.session import LiveSession
    from sst.live.transcript import Transcript, heading
    from sst.live.wasapi import Capture
    from sst.settings import Profiles, Settings

    profile = Profiles.load().current
    settings, gateway = Settings.load(profile.settings_file), GatewayConfig.load(profile.gateway_file)
    target, mic_target = args.to or settings.live_target, args.mic_to or settings.live_mic_target
    for code in (target, mic_target):
        if code not in LANGUAGES.values():
            raise ValueError(f"--to and --mic-to take a language code: {', '.join(sorted(LANGUAGES.values()))}")
    key = gateway.key_for("gemini")
    if not key:
        raise ValueError("Live captions need a Gemini key: add one in Rflow, AI & models.")
    config = LiveConfig(target=target, mic_target=mic_target, source=args.source or settings.live_source)
    stopped = threading.Event()

    def show(event) -> None:
        who = "You: " if event.lane == MIC and config.marks_mine else ""
        if event.kind is Kind.LINE:
            into = language_name(config.for_lane(event.lane).target)
            print(f"\n  {who}{event.source}\n  -> {event.text or '(already ' + into + ')'}"
                  + (f"   [complete {event.seconds:.1f} s after the voice paused]" if event.seconds else ""), flush=True)
        elif event.kind is Kind.ERROR:
            print(f"\n  ! {who}{event.text}", flush=True)
        elif event.kind is Kind.STATUS:
            print(f"  ({who}{event.text})", flush=True)
            if event.text == "Stopped":
                stopped.set()

    session = LiveSession(config, Transcript(profile.folder() / "live captions", config), show)
    print(f"Live translation ({config.source}): {heading(config)}. Ctrl+C stops.")
    for lane in config.lanes:
        capture = Capture.microphone() if lane == MIC else Capture.speakers()
        session.add(lane, capture, lambda emit, lane=lane: GeminiLiveTranslate(key, config.for_lane(lane), emit, lane=lane))
    if args.speak:
        from sst.live.ducking import Ducker
        from sst.live.speaker import Speaker
        from sst.live.voice import LOCAL, PiperVoice, install, local
        from sst.live.wasapi import Player
        if not LOCAL.ready():  # the voice on this laptop, for when Gemini's own voice doesn't come
            print(f"Downloading the voice {LOCAL.name} ({round(LOCAL.size / 1e6)} MB, once)...")
            install(LOCAL, progress=lambda done, total: print(f"\r  {done * 100 // total}%", end="", flush=True))
            print()
        voice = local()
        lanes = config.spoken_lanes(voice.language)
        if not lanes:
            print(f"  (Gemini speaks the translation; {voice.name} speaks only {language_name(voice.language)}, so "
                  f"can't stand in for it here.)")
        session.set_speaker(Speaker(lambda: PiperVoice(voice), Player(), lanes, voice.language, settings.live_speak_speed,
                                    on_problem=lambda message: print(f"\n  ! {message}", flush=True),
                                    ducker=Ducker(settings.live_duck), own=config.spoken_lanes()))
    try:
        stopped.wait(args.seconds or None)
    finally:
        session.stop()
        if session.transcript.path:
            print(f"Transcript: {session.transcript.path}")


def cmd_app(args) -> None:
    from sst.instance import launch  # a running Rflow is asked to show its window before the app is imported

    sys.exit(launch())


def cmd_eval(args) -> None:
    from sst import bench, evaluate
    from sst.gateway import GatewayConfig, Polisher
    from sst.settings import Profiles, Settings

    profile = Profiles.load().current  # the models, words and tests of the profile in use in the app
    folders = [Path(f) for f in args.folders] or bench.sessions(profile.folder(bench.BENCH_DIR))
    if not folders:
        raise SystemExit("No reading tests yet: record one in Rflow's Reading test page, or give test folders.")
    settings, gateway = Settings.load(profile.settings_file), GatewayConfig.load(profile.gateway_file)
    models = [] if args.no_cleanup else args.model or [m for m in (settings.cleanup_model, settings.cleanup_fallback) if m]
    polishers = {m.rsplit("/", 1)[-1]: Polisher(gateway, m, settings.vocabulary) for m in models} if gateway.address else {}
    evaluate.pipelines_for(polishers, args.degrade or [])  # a typo in --degrade is reported before the slow model load
    engine = _load(args.engine or usable(settings.speech_model, gateway))  # by default the profile's, as in the app
    pipelines = evaluate.pipelines_for(polishers, args.degrade or [], title=engine.title)
    if hasattr(engine, "language"):  # a model that knows many languages listens for the profile's choice
        engine.language = settings.speech_language
    if hasattr(engine, "words"):  # as in dictation: the recogniser listens for Your words
        chosen = [w.strip() for w in args.words.split(",")] if args.words else settings.vocabulary
        from sst.pipeline.dictionary import speech_hints
        engine.words = [] if args.no_words else speech_hints(w for w in chosen if w)  # as in dictation
        biased = getattr(engine, "biased", True)  # Parakeet listens for them only with its bpe.vocab
        print(f"  Your words used while recognising: {len(engine.words)}" + ("" if biased else " (no bpe.vocab)"))
    results = evaluate.run(folders, engine, pipelines, progress=lambda text: print(" ", text))
    out = Path(args.out) if args.out else evaluate.output_folder(folders)
    results.save(out)
    print()
    print(results.report())
    print(f"Saved report.md and results.json in {out}")


def cmd_web(args) -> None:
    from sst.web import serve

    serve(_load(args.engine), port=args.port, open_browser=not args.no_browser)


def main() -> None:
    # A Windows console that isn't UTF-8 can't print every character of a report (e.g. "→"): show "?" rather than fail.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(prog="sst", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"sst {__version__}")
    parser.add_argument("--engine", choices=ENGINES,
                        help="speech recognition model (default: parakeet; for eval, the one chosen in the app)")
    parser.add_argument("--device", type=int, default=None, help="microphone number from `sst devices`")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("app", help="the tray app with the recording indicator, settings and history")
    p_dictate = sub.add_parser("dictate", help="type what you say into any app, using a hotkey")
    p_dictate.add_argument("--hotkey", default="ctrl+win",
                           help="ctrl+win (default, like Wispr Flow), menu (the Menu key), or e.g. ctrl+alt+d")
    p_dictate.add_argument("--no-save", action="store_true", help="don't keep recordings in recordings/")
    sub.add_parser("start", help="record from the microphone and transcribe")
    p_file = sub.add_parser("file", help="transcribe a WAV file")
    p_file.add_argument("path")
    sub.add_parser("devices", help="list microphones")
    p_eval = sub.add_parser("eval", aliases=["bench"], help="score reading tests: recognition alone, degraded, with cleanup")
    p_eval.add_argument("folders", nargs="*", help="test folders (default: every test of the profile in use)")
    p_eval.add_argument("--model", action="append", help="cleanup model to compare (repeatable); default: the ones in Settings")
    p_eval.add_argument("--no-cleanup", action="store_true", help="recognition only, no cleanup models")
    p_eval.add_argument("--no-words", action="store_true", help="recognise without listening for Your words (hotwords)")
    p_eval.add_argument("--words", help='try other words instead of Your words, e.g. "Karthi,Vercel,CodeQL"')
    p_eval.add_argument("--degrade", action="append",
                        help="also score the audio made worse on purpose: narrowband (phone / Bluetooth call quality) "
                             "or gain:<dB>, e.g. gain:-20 (repeatable)")
    p_eval.add_argument("--out", help="where report.md and results.json go (default: the test's folder, or 'summary')")
    p_live = sub.add_parser("live", help="live captions of what the laptop plays, translated (Gemini Live Translate)")
    p_live.add_argument("--to", help="language code to translate into, e.g. en, ja (default: the one set in Rflow)")
    p_live.add_argument("--seconds", type=float, help="stop after this long (default: until Ctrl+C)")
    p_live.add_argument("--source", choices=["computer", "microphone", "both"],
                        help="what to translate: what the laptop plays, the microphone, or both (default: as set in Rflow)")
    p_live.add_argument("--mic-to", help="language code for what the microphone hears (default: as set in Rflow, ja)")
    p_live.add_argument("--speak", action="store_true",
                        help="speak the translation aloud too (an English voice on this laptop, downloaded the first time)")
    p_web = sub.add_parser("web", help="open the record/stop page in your browser")
    p_web.add_argument("--port", type=int, default=8765)
    p_web.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")

    args = parser.parse_args()
    if args.command not in ("eval", "bench"):  # eval takes the profile's own model when none is given
        args.engine = args.engine or DEFAULT_MODEL
    logging.basicConfig(level=logging.INFO, format="  %(message)s")  # e.g. the per-dictation timing line
    try:
        commands = {"app": cmd_app, "dictate": cmd_dictate, "start": cmd_start, "file": cmd_file,
                    "devices": cmd_devices, "eval": cmd_eval, "bench": cmd_eval, "web": cmd_web, "live": cmd_live}
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\nStopped.")
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
