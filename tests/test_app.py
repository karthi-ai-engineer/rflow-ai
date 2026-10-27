"""The tray app: the pill, what TrayApp does for the window, and "open Rflow again" (built off-screen: nothing appears
on the screen, nothing takes focus)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import dataclasses  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6.QtNetwork import QLocalServer  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from sst import app as sst_app  # noqa: E402
from sst import settings as settings_module  # noqa: E402
from sst.audio import Take  # noqa: E402
from sst.engines.cloud import CLOUD, REMOTE  # noqa: E402
from sst.gateway import GatewayConfig  # noqa: E402
from sst.settings import Profile, Profiles, Settings  # noqa: E402
from sst.wakeui import BLUETOOTH  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def qt():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize("state, message", [("recording", ""), ("transcribing", ""), ("typed", ""), ("typed_raw", ""),
                                            ("typed_local", ""), ("cancelled", ""),
                                            ("ignored", ""), ("warning", "Almost silent: check the microphone."),
                                            ("error", "Could not open the microphone")])
def test_pill_draws_every_state(state, message):
    pill = sst_app.Pill(level=lambda: 0.05)
    pill.show_state(state, message)
    assert pill.isVisible() and not pill.grab().isNull()
    pill.show_state("idle")
    assert not pill.isVisible()


def test_pill_grows_to_fit_long_messages():
    pill = sst_app.Pill()
    pill.show_state("typed")
    short = pill.width()
    pill.show_state("warning", "Almost silent: check the microphone.")
    assert pill.width() > short


def test_only_the_pill_lets_clicks_through_and_nothing_takes_the_focus(monkeypatch):
    import ctypes
    styles = {}
    user32 = SimpleNamespace(GetWindowLongPtrW=lambda hwnd, index: 0x20 | 0x100,  # click-through already, and another bit
                             SetWindowLongPtrW=lambda hwnd, index, style: styles.__setitem__(hwnd, style))
    monkeypatch.setattr(ctypes.windll, "user32", user32)
    sst_app._no_activate(1)  # the pill
    sst_app._no_activate(2, click_through=False)  # Translate's popup, the Text Transform menu
    NOACTIVATE, TRANSPARENT, TOOLWINDOW = 0x08000000, 0x20, 0x80
    assert styles[1] & NOACTIVATE and styles[1] & TRANSPARENT and styles[1] & TOOLWINDOW
    assert styles[2] & NOACTIVATE and not styles[2] & TRANSPARENT and styles[2] & 0x100  # other bits kept


def _fake_tray_app(settings: Settings):
    """TrayApp's methods on a stand-in: apply_settings only records, so nothing touches the real settings file."""
    from sst.pipeline.dictionary import DictionaryStore
    fake = SimpleNamespace(settings=settings, gateway=GatewayConfig(), dictation=None, applied=[], cleanups=0, loads=0,
                           profile=Profile("default"), dictionary=DictionaryStore())

    def apply(new):
        fake.settings = new
        fake.applied.append(new)
    fake.apply_settings = apply
    fake._apply_cleanup = lambda: setattr(fake, "cleanups", fake.cleanups + 1)
    fake._load_speech = lambda: setattr(fake, "loads", fake.loads + 1)
    return fake


def test_adding_words_skips_ones_already_there():
    fake = _fake_tray_app(Settings(vocabulary=["GitHub"]))
    assert sst_app.TrayApp.add_words(fake, ["github", "Tamil", "CodeQL", "Tamil"]) == 2
    assert fake.settings.vocabulary == ["GitHub", "Tamil", "CodeQL"]
    assert sst_app.TrayApp.add_words(fake, ["TAMIL"]) == 0 and len(fake.applied) == 1  # nothing new: nothing saved


def test_removing_a_word():
    fake = _fake_tray_app(Settings(vocabulary=["GitHub", "Tamil"]))
    sst_app.TrayApp.remove_word(fake, "GitHub")
    assert fake.settings.vocabulary == ["Tamil"]


def test_saving_the_cleanup_keeps_the_other_settings(monkeypatch):
    monkeypatch.setattr(GatewayConfig, "save", lambda self, path=None: None)
    fake = _fake_tray_app(Settings(hotkey="menu", vocabulary=["Tamil"]))
    sst_app.TrayApp.save_cleanup(fake, True, "model-a", "model-b", GatewayConfig("http://localhost:11434/v1", ""))
    s = fake.settings
    assert (s.cleanup, s.cleanup_model, s.cleanup_fallback, s.hotkey, s.vocabulary) == (True, "model-a", "model-b", "menu",
                                                                                         ["Tamil"])
    assert fake.gateway.base_url == "http://localhost:11434/v1"


def test_the_first_close_tells_once_that_rflow_keeps_running():
    told = []
    fake = _fake_tray_app(Settings())
    fake._notify = lambda title, message, *_: told.append(title)
    fake.hotkey_label = lambda: "Ctrl+Win"
    sst_app.TrayApp.window_closed(fake)
    sst_app.TrayApp.window_closed(fake)
    assert told == ["Rflow is still running"] and fake.settings.told_about_tray


class FakeListener:
    """Stands in for the keyboard hook: no real keys are watched."""

    def __init__(self, hotkey):
        import queue
        self.hotkey, self.events, self.recording, self.running = hotkey, queue.Queue(), False, False
        self.on_event = None

    def send(self, event, at=0.0):
        """As the keyboard hook does: queue the event, then wake the app up (it polls nothing)."""
        self.events.put((event, at))
        self.on_event()

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def capture(self, keys):  # Text Transform's menu takes keys while it is open
        self.captured = keys


class FakeEngine:
    """Named after the model it was loaded as, like the real engines."""

    def __init__(self, name="parakeet", language="", api_key="", model="", fallback=None, url=None):
        self.name, self.title, self.words = name, name.title(), []
        if name.startswith("whisper") or name in REMOTE:
            self.language = language  # like the real one: a model that knows many languages
        if name in REMOTE:  # like sst.engines.cloud.CloudEngine
            self.api_key, self.fallback, self.url = api_key, fallback, url
            self.model = model or (CLOUD[name].models[0] if name in CLOUD else "")

    def set_url(self, url):
        self.url = url

    def transcribe(self, audio, rate):
        return "hello"


@pytest.fixture
def tray_app(monkeypatch, tmp_path):
    """The real TrayApp with everything outside the process faked: settings in memory, no model, no hook."""
    from PySide6.QtTest import QTest

    from sst.settings import Stats
    saved, files = {}, {}  # files: each profile's settings by path, as if on disk
    monkeypatch.setattr(settings_module, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(sst_app.bench, "BENCH_DIR", tmp_path / "bench")
    # quit() marks a clean exit in the log folder: never the real one, where a running Rflow keeps its marker
    monkeypatch.setattr(sst_app, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(sst_app, "DEBUG_DIR", tmp_path / "debug")
    monkeypatch.setattr(sst_app, "DATA_DIRS", (tmp_path / "roaming" / "sst", tmp_path / "local" / "sst"))  # Start over
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls, path=None: files.get(path) or Settings(
        welcomed=path == tmp_path / "settings.json")))  # the first profile is set up already, a new one isn't

    def save_settings(self, path=None):
        saved["settings"] = files[path] = self
    monkeypatch.setattr(Settings, "save", save_settings)
    monkeypatch.setattr(Stats, "load", classmethod(lambda cls, path=None, history=None: Stats()))
    monkeypatch.setattr(Stats, "save", lambda self, path=None: saved.__setitem__("stats", self))
    monkeypatch.setattr(GatewayConfig, "load", classmethod(lambda cls, path=None: GatewayConfig()))
    monkeypatch.setattr(GatewayConfig, "save", lambda self, path=None: saved.__setitem__("gateway", self))
    monkeypatch.setattr(Profiles, "load", classmethod(lambda cls, path=None: Profiles()))
    monkeypatch.setattr(Profiles, "save", lambda self, path=None: saved.__setitem__("profiles", self))
    history = []
    monkeypatch.setattr(sst_app, "add_to_history", lambda text, heard=None, path=None: history.insert(0, {
        "time": "2026-09-30 10:15:00", "text": text}))
    monkeypatch.setattr(sst_app, "read_history", lambda path=None: history)
    loads = []
    monkeypatch.setattr(sst_app, "load_engine", lambda name, *options, **named: loads.append(name) or FakeEngine(
        name, *options, **named))
    monkeypatch.setattr(sst_app, "HotkeyListener", FakeListener)
    monkeypatch.setattr(sst_app.Recorder, "keep_open", lambda self: None)  # the always-on microphone stays closed
    monkeypatch.setattr(sst_app, "wispr_flow_running", lambda: False)
    monkeypatch.setattr(sst_app, "input_device_names", lambda refresh=True: ["Mic A"])
    monkeypatch.setattr(sst_app, "SERVER_NAME", f"Rflow-test-app-{os.getpid()}")
    app = sst_app.TrayApp(quiet_start=True)
    for _ in range(100):  # the model "loads" on its thread
        if app.dictation:
            break
        QTest.qWait(20)
    yield app, saved, history
    app.window.hide()
    app.quit()


def test_the_tray_app_and_its_window_work_together(tray_app):
    app, saved, history = tray_app
    assert app.dictation and app.listener.running and app.window.ready
    assert "Ready: hold Ctrl+Win" in app.window.status_message
    app.window.open("home")
    app._on_result("hello world", "Hello, world.", 2.0)  # what the dictation reports after typing
    assert history[0]["text"] == "Hello, world." and saved["stats"].words == 2
    assert "Hello, world." in [label.text() for label in app.window.pages["home"].findChildren(QLabel)]
    settings_page = app.window.pages["settings"]
    first_listener = app.listener
    settings_page.hotkey.setCurrentIndex(settings_page.hotkey.findData("menu"))  # applied at once
    assert saved["settings"].hotkey == "menu" and app.listener is not first_listener and not first_listener.running
    assert app.hotkey_label() == "Menu key"
    assert app.add_words(["Tamil"]) == 1 and saved["settings"].vocabulary == ["Tamil"]
    app.window.close()
    assert not app.window.isVisible() and saved["settings"].told_about_tray


def test_the_tray_app_shows_its_window_when_opened_again(tray_app):
    app, _, _ = tray_app
    assert not app.window.isVisible()  # started at sign-in: only the tray icon
    assert sst_app.show_running_window()
    from PySide6.QtTest import QTest
    for _ in range(50):
        if app.window.isVisible():
            break
        QTest.qWait(20)
    assert app.window.isVisible()


def test_opening_rflow_again_reaches_the_running_copy(monkeypatch):
    monkeypatch.setattr(sst_app, "SERVER_NAME", f"Rflow-test-{os.getpid()}")
    assert not sst_app.show_running_window()  # nothing is running: the caller starts normally
    server = QLocalServer()
    assert server.listen(sst_app.SERVER_NAME)
    assert sst_app.show_running_window()
    assert server.waitForNewConnection(2000) or server.hasPendingConnections()
    server.close()


def test_each_profile_has_its_own_setup(tray_app):
    app, saved, _ = tray_app
    first_listener = app.listener
    app.add_words(["Tamil"])
    app.window.open("home")
    app.create_profile("Rahul")  # switches to it
    assert app.profile.name == "Rahul" and app.settings.vocabulary == [] and not app.settings.welcomed
    assert app.window.isVisible() and app.window.current_page() == "welcome"  # a new profile starts with the welcome
    assert "Rahul" in app.window.profile_button.text()
    app.apply_settings(dataclasses.replace(app.settings, hotkey="menu", welcomed=True))
    assert app.listener is not first_listener and app.hotkey_label() == "Menu key"
    app.switch_profile("default")
    assert app.settings.vocabulary == ["Tamil"] and app.hotkey_label() == "Ctrl+Win"  # the first profile's own again
    assert saved["profiles"].active == "default"
    app.delete_profile("rahul")
    assert [p.id for p in app.profiles.items] == ["default"]


def test_each_profile_counts_what_gemini_costs_and_the_app_names_the_feature(tray_app, monkeypatch):
    from sst import usage
    app, _, _ = tray_app
    assert usage.active() is app.usage and app.usage.path == app.profile.usage_file

    class Counting:  # a Gemini model that counts each answer the way sst.gateway does
        def __init__(self, *args, **kwargs):
            pass

        def complete(self, message):
            usage.record("cleanup", "gemini-3.5-flash-lite", text_in=100)
            return message
    monkeypatch.setattr(sst_app, "Polisher", Counting)
    app.run_transform("so this is the text we keep for the test", "concise")
    app.run_translation("Hello there.", "Spanish")
    assert {key.split("|")[0] for day in app.usage.days.values() for key in day} == {"transform", "translate"}
    first = app.usage
    app.create_profile("Rahul")
    assert usage.active() is app.usage and app.usage is not first and app.usage.days == {}
    assert first.path.exists()  # the first profile's count was saved as the other took over


def _wait_for(condition, qt_wait):
    for _ in range(100):
        if condition():
            return True
        qt_wait(20)
    return condition()


def test_another_speech_model_loads_in_the_background_and_takes_over(tray_app, whisper_downloaded):
    from PySide6.QtTest import QTest

    app, saved, _ = tray_app
    first = app.dictation.engine
    assert first.name == "parakeet" and app.speech_in_use() == "parakeet"
    app.add_words(["Tamil"])
    app.choose_speech_model("whisper-turbo")
    assert saved["settings"].speech_model == "whisper-turbo"
    assert _wait_for(lambda: app.speech_in_use() == "whisper-turbo", QTest.qWait)
    assert app.dictation.engine is not first and app.dictation.engine.words == ["Tamil"]  # Your words go along
    assert not app.loading_speech and "Ready" in app.window.status_message


def test_a_model_not_downloaded_or_unknown_is_not_chosen(tray_app):
    app, saved, _ = tray_app
    app.choose_speech_model("whisper-turbo")  # not downloaded: the page offers the download instead
    app.choose_speech_model("no-such-model")
    assert app.settings.speech_model == "parakeet" and app.speech_in_use() == "parakeet"


def test_a_failed_switch_keeps_the_model_in_use(tray_app, monkeypatch, whisper_downloaded):
    from PySide6.QtTest import QTest

    app, _, _ = tray_app

    def broken(name, *options):
        raise OSError("not downloaded")
    monkeypatch.setattr(sst_app, "load_engine", broken)
    told = []
    monkeypatch.setattr(app, "_notify", lambda title, message, *_: told.append(message))
    app.choose_speech_model("whisper-turbo")
    assert _wait_for(lambda: not app.loading_speech, QTest.qWait)
    assert app.speech_in_use() == "parakeet" and "still using NVIDIA Parakeet" in told[-1]


def test_switching_profiles_switches_the_speech_model(tray_app, whisper_downloaded):
    from PySide6.QtTest import QTest

    app, _, _ = tray_app
    app.create_profile("Rahul")
    app.choose_speech_model("whisper-turbo")
    assert _wait_for(lambda: app.speech_in_use() == "whisper-turbo", QTest.qWait)
    app.switch_profile("default")  # Karthi's profile still uses Parakeet
    assert _wait_for(lambda: app.speech_in_use() == "parakeet", QTest.qWait)


def test_downloading_a_speech_model_then_using_it(tray_app, monkeypatch, whisper_downloaded):
    from PySide6.QtTest import QTest
    app, saved, _ = tray_app
    seen = []

    def fake_download(model, progress, cancelled):
        progress(model.size // 2, model.size)
        seen.append(app.downloading)
        progress(model.size, model.size)
    monkeypatch.setattr(sst_app.downloads, "download", fake_download)
    app.download_speech_model("whisper-turbo")
    assert _wait_for(lambda: app.speech_in_use() == "whisper-turbo", QTest.qWait)
    assert app.downloading is None and saved["settings"].speech_model == "whisper-turbo"
    app.set_speech_language("ta")
    assert app.dictation.engine.language == "ta"  # no reload needed


def test_a_cancelled_or_failed_download_changes_nothing(tray_app, monkeypatch):
    from PySide6.QtTest import QTest
    app, _, _ = tray_app
    told = []
    monkeypatch.setattr(app, "_notify", lambda title, message, *_: told.append(message))
    for error in (sst_app.downloads.Cancelled("cancelled"), sst_app.downloads.DownloadError("no connection")):
        def fake_download(model, progress, cancelled, error=error):
            raise error
        monkeypatch.setattr(sst_app.downloads, "download", fake_download)
        app.download_speech_model("whisper-turbo")
        assert _wait_for(lambda: app.downloading is None, QTest.qWait)
    assert app.speech_in_use() == "parakeet" and len(told) == 1 and "no connection" in told[0]


def test_only_a_model_not_in_use_can_be_removed(tray_app, monkeypatch, whisper_downloaded):
    removed = []
    monkeypatch.setattr(sst_app.downloads, "remove", lambda model: removed.append(model.folder))
    app, _, _ = tray_app
    app.remove_speech_model("parakeet")  # comes with Rflow: nothing to remove
    app.remove_speech_model("whisper-turbo")
    assert removed == ["faster-whisper-large-v3-turbo"]
    app.choose_speech_model("whisper-turbo")
    app.remove_speech_model("whisper-turbo")  # chosen now: stays
    assert removed == ["faster-whisper-large-v3-turbo"]


def test_scanning_the_computer_times_the_model_in_use(tray_app, monkeypatch, tmp_path):
    import numpy as np
    from PySide6.QtTest import QTest
    app, _, _ = tray_app
    pc = sst_app.scan.Computer(processor="Test CPU", cores=4, threads=8, memory_gb=16, free_memory_gb=8,
                               free_disk_gb=100, score=sst_app.scan.REFERENCE_SCORE)
    monkeypatch.setattr(sst_app.scan, "computer", lambda: pc)
    monkeypatch.setattr(sst_app.scan, "SCAN_FILE", tmp_path / "scan.json")
    monkeypatch.setattr(sst_app, "_sample_sentence", lambda: (np.zeros(16_000, dtype=np.float32), 16_000))
    app.scan_computer()
    assert _wait_for(lambda: not app.scanning, QTest.qWait)
    verdicts = {v["key"]: v for v in app.last_scan["verdicts"]}
    assert verdicts["parakeet"]["measured"] and verdicts["parakeet"]["level"] == "recommended"
    assert not verdicts["whisper-turbo"]["measured"]  # not downloaded: estimated
    assert (tmp_path / "scan.json").exists()


def test_a_cloud_model_needs_its_key_then_takes_over_with_parakeet_behind_it(tray_app):
    from PySide6.QtTest import QTest

    app, saved, _ = tray_app
    parakeet = app.dictation.engine
    app.choose_speech_model("gemini")  # no key yet: the page asks for one instead
    assert app.settings.speech_model == "parakeet"
    app.add_words(["Karthi"])
    app.use_cloud_speech("gemini", "AIza-test", "gemini-3.5-transcribe")
    assert saved["gateway"].key_for("gemini") == "AIza-test"  # encrypted with the others, shared with AI cleanup
    assert saved["settings"].speech_cloud_models == {"gemini": "gemini-3.5-transcribe"}
    assert _wait_for(lambda: app.speech_in_use() == "gemini", QTest.qWait)
    engine = app.dictation.engine
    assert (engine.api_key, engine.model, engine.words) == ("AIza-test", "gemini-3.5-transcribe", ["Karthi"])
    assert engine.fallback() is parakeet  # already in memory: nothing to load when the provider fails
    assert "speech: Gemini" in app.window.status_message
    app.use_cloud_speech("gemini", "AIza-new", "gemini-3.5-flash-lite")  # the card's Save: no reload
    assert app.dictation.engine is engine and (engine.api_key, engine.model) == ("AIza-new", "gemini-3.5-flash-lite")
    app.set_speech_language("ta")
    assert engine.language == "ta"
    app.save_cleanup(False, "", "", GatewayConfig())  # the key is gone: back to Parakeet, the one in memory
    assert _wait_for(lambda: app.speech_in_use() == "parakeet", QTest.qWait)
    assert app.dictation.engine is parakeet


def test_starting_with_a_cloud_model_loads_parakeet_only_when_needed(tray_app):
    from PySide6.QtTest import QTest

    app, _, _ = tray_app
    app.use_cloud_speech("gemini", "AIza-test", "")
    assert _wait_for(lambda: app.speech_in_use() == "gemini", QTest.qWait)
    app._local = None  # as after a start with Gemini chosen: Parakeet isn't in memory
    engine = app.dictation.engine
    assert engine.model == "gemini-3.5-transcribe"  # Gemini's first model
    first = engine.fallback()
    assert first.name == "parakeet" and engine.fallback() is first  # loaded once


def test_parakeet_typing_for_the_cloud_is_said_once_in_a_while(tray_app, monkeypatch):
    app, _, _ = tray_app
    told = []
    monkeypatch.setattr(app, "_notify", lambda title, message, *_: told.append((title, message)))
    app._on_state("typed_local", "OpenAI: cannot reach api.openai.com")
    app._on_state("typed_local", "OpenAI: cannot reach api.openai.com")
    assert told == [("Cloud speech unavailable", "Parakeet typed it on this computer (OpenAI: cannot reach "
                                                 "api.openai.com).")]
    assert app.pill.state == "typed_local"


def test_a_model_on_your_own_server_switches_its_address_without_a_reload(tray_app):
    from PySide6.QtTest import QTest

    from sst.gateway import SPEECH_SERVER
    app, saved, _ = tray_app
    app.gateway = GatewayConfig("http://gateway.example/v1", "cleanup-key", "vllm")
    app.choose_speech_model("server")  # nothing set up yet
    assert app.settings.speech_model == "parakeet"
    app.use_server_speech("http://gateway.example/v1", "cleanup-key", "whisper-1")
    assert saved["gateway"].speech_server() == ("http://gateway.example/v1", "cleanup-key")
    assert _wait_for(lambda: app.speech_in_use() == "server", QTest.qWait)
    engine = app.dictation.engine
    assert (engine.url, engine.api_key, engine.model) == ("http://gateway.example/v1", "cleanup-key", "whisper-1")
    assert engine.fallback is not None
    app.use_server_speech("http://localhost:8000/v1", "", "openai/whisper-large-v3-turbo")  # Save: no reload
    assert app.dictation.engine is engine and engine.url == "http://localhost:8000/v1" and engine.api_key == ""
    assert engine.model == "openai/whisper-large-v3-turbo"
    assert (app.gateway.base_url, app.gateway.api_key) == ("http://gateway.example/v1", "cleanup-key")  # AI cleanup's
    app.save_cleanup(False, "", "", app.gateway.with_entry(SPEECH_SERVER, "", ""))  # the server is gone: Parakeet
    assert _wait_for(lambda: app.speech_in_use() == "parakeet", QTest.qWait)


def test_a_key_added_on_ai_and_models_never_changes_the_cleanup_or_the_speech_model(tray_app, monkeypatch):
    # The user testing's M-04: a Gemini key added for live translation switched AI cleanup to Gemini, with no model.
    from sst.gateway import SPEECH_SERVER
    monkeypatch.setattr(sst_app.Polisher, "prepare", lambda self: None)  # no connection to the provider
    app, saved, _ = tray_app
    app.save_cleanup(True, "gpt-4o-mini", "", GatewayConfig("", "sk-openai", "openai"))
    assert "add a Gemini key" in app.live_problem()
    app.save_key("gemini", "AIza-live")
    assert saved["gateway"].key_for("gemini") == "AIza-live" and not app.live_problem()
    assert (app.gateway.chosen, app.gateway.api_key, app.settings.cleanup_model) == ("openai", "sk-openai", "gpt-4o-mini")
    assert app.dictation.cleanup.model == "gpt-4o-mini" and app.speech_in_use() == "parakeet"
    app.save_key("openai", "sk-new")  # the cleanup's own key: used from the next dictation
    assert app.dictation.cleanup.config.api_key == "sk-new" and app.gateway.chosen == "openai"
    app.save_server("http://localhost:8000/v1", "", ("vllm", SPEECH_SERVER))
    assert app.gateway.speech_server() == ("http://localhost:8000/v1", "") and app.gateway.chosen == "openai"
    assert app.settings.speech_model == "parakeet"  # kept for when it's chosen
    app.save_key("gemini", "")  # removed
    assert "add a Gemini key" in app.live_problem() and app.gateway.key_for("openai") == "sk-new"


def test_a_new_install_starts_without_a_speech_model_then_downloads_parakeet(no_parakeet, tray_app, monkeypatch,
                                                                              tmp_path):
    from PySide6.QtTest import QTest

    from sst.engines import parakeet
    app, _, _ = tray_app
    assert app.dictation is None and not app.loading_speech
    assert "Choose a speech model" in app.window.status_message
    with pytest.raises(RuntimeError, match="isn't downloaded"):
        app._fallback_engine()  # a cloud model has nothing to fall back on yet
    folder = tmp_path / "downloaded"

    def fake_download(model, progress, cancelled):
        progress(model.size, model.size)
        folder.mkdir()
        (folder / "encoder.int8.onnx").write_bytes(b"")
        monkeypatch.setattr(parakeet, "MODEL_DIR", folder)  # as if the download had finished
    monkeypatch.setattr(sst_app.downloads, "download", fake_download)
    app.download_speech_model("parakeet")  # Parakeet was the setting all along: it loads once downloaded
    assert _wait_for(lambda: app.speech_in_use() == "parakeet", QTest.qWait)
    assert app.listener.running and "Ready: hold Ctrl+Win" in app.window.status_message


def test_a_new_install_can_start_with_a_cloud_model_only(no_parakeet, tray_app):
    from PySide6.QtTest import QTest

    app, _, _ = tray_app
    app.use_cloud_speech("gemini", "AIza-test", "")
    assert _wait_for(lambda: app.speech_in_use() == "gemini", QTest.qWait)
    assert app.listener.running and app._local is None  # nothing on this computer


def test_a_profile_on_a_provider_no_longer_offered_dictates_with_gemini_or_parakeet(tray_app):
    from PySide6.QtTest import QTest

    app, _, _ = tray_app
    app.use_cloud_speech("groq", "gsk-test", "")  # Groq chosen before it was hidden: no crash, Parakeet types
    assert app.settings.speech_model == "groq" and app.gateway.key_for("groq") == "gsk-test"  # the choice is kept
    QTest.qWait(50)
    assert _wait_for(lambda: app.speech_in_use() == "parakeet" and not app.loading_speech, QTest.qWait)
    app.save_key("gemini", "AIza-test")
    app.use_cloud_speech("openai", "sk-test", "")  # with a Gemini key: Gemini, on its first model
    assert _wait_for(lambda: app.speech_in_use() == "gemini", QTest.qWait)
    assert app.dictation.engine.model == "gemini-3.5-transcribe"


def test_the_tray_app_dictates_through_the_voice_pipeline(tray_app):
    app, _, _ = tray_app
    pipeline = app.dictation.pipeline
    assert pipeline is not None and pipeline.engine is app.dictation.engine
    assert pipeline.stages.formatter is not None and pipeline.stages.guard is not None and pipeline.stages.llm is None
    assert app.recorder.preroll_seconds == 2.0 and app.recorder.warm_seconds == float("inf")  # always-on, 2 s pre-roll
    app.add_words(["PostgreSQL"])
    app.add_sound_alike("post grass", "PostgreSQL")
    assert pipeline.stages.dictionary.correct("the post grass server").text == "the PostgreSQL server"
    assert "PostgreSQL" in app.dictation.engine.words  # the recogniser listens for the term too
    app.apply_settings(dataclasses.replace(app.settings, format_text=False))
    assert app.dictation.pipeline.stages.formatter is None
    app.apply_settings(dataclasses.replace(app.settings, voice_pipeline=False, always_on_mic=False))
    assert app.dictation.pipeline is None and app.recorder.warm_seconds == sst_app.WARM_SECONDS  # the classic way



def test_translate_starts_with_ctrl_c_c_and_follows_the_settings(tray_app):
    app, _, _ = tray_app
    controller = app.translator
    assert controller.listener is not None and controller.listener.hotkey.text == "ctrl+c+c" and controller.double_copy
    assert not app.translate_ready()  # no AI model yet
    app.apply_settings(dataclasses.replace(app.settings, translate_shortcut=""))
    assert controller.listener is None  # off
    app.apply_settings(dataclasses.replace(app.settings, translate_shortcut="ctrl+alt+l"))
    assert controller.listener.hotkey.text == "ctrl+alt+l" and not controller.double_copy


def test_snippets_are_read_from_the_profile_at_each_dictation(tray_app):
    app, _, _ = tray_app
    assert app.snippets() == []
    app.apply_settings(dataclasses.replace(app.settings, snippets=[{"cue": "my email", "text": "xyz@gmail.com"}]))
    assert [s.text for s in app.snippets()] == ["xyz@gmail.com"]
    assert app.dictation.snippets == app.snippets  # both ways of dictating ask it
    assert app.dictation.pipeline is None or app.dictation.pipeline.stages.snippets == app.snippets


def test_voice_commands_go_to_text_transform(tray_app, monkeypatch):
    app, _, _ = tray_app
    assert app.voice_command("Make it concise.") is None  # no AI model: typed as said, nothing lost
    assert app.voice_command("Undo that.") == "undo"  # the pill says there is nothing to undo
    monkeypatch.setattr(app, "transform_ready", lambda: True)
    assert app.voice_command("Make it concise.") == "concise"
    assert app.voice_command("Make it concise and send it to Priya.") is None
    app.apply_settings(dataclasses.replace(app.settings, command_phrases={"concise": "trim it"}))
    assert app.voice_command("Trim it.") == "concise" and app.voice_command("Make it concise.") is None
    app.apply_settings(dataclasses.replace(app.settings, voice_commands=False))
    assert app.voice_command("Trim it.") is None
    assert app.dictation.command == app.voice_command  # both ways of dictating ask it
    assert app.dictation.pipeline is None or app.dictation.pipeline.stages.command == app.voice_command
    ran = []
    monkeypatch.setattr(app.transforms, "run_command", ran.append)
    app.signals.command.emit("bullets")  # from the dictation's thread, carried out on the app's
    assert ran == ["bullets"]


def test_text_transform_starts_with_its_shortcut_and_uses_the_cleanup_model(tray_app):
    app, saved, history = tray_app
    controller = app.transforms
    assert controller.listener is not None and controller.listener.hotkey.text == "double ctrl"  # the default
    assert not app.transform_ready()  # no AI model yet
    app.apply_settings(dataclasses.replace(app.settings, transform_shortcut=""))
    assert controller.listener is None  # off
    app.apply_settings(dataclasses.replace(app.settings, transform_shortcut="ctrl+alt+t"))
    assert controller.listener.hotkey.text == "ctrl+alt+t"
    app._on_result("hello there", "Hello there.", 1.0)
    assert controller.last_typed[0] == "Hello there. "  # as typed: the shortcut takes it when nothing is selected
    app.remember("Short.", "A longer original.")
    assert history[0]["text"] == "Short."


def test_live_translation_needs_a_gemini_key_and_asks_once_before_the_first_start(tray_app, monkeypatch):
    app, _, _ = tray_app
    assert "add a Gemini key" in app.start_live() and not app.live_running()
    app.gateway = GatewayConfig(provider="gemini", api_key="AIza-test")
    asked = []
    monkeypatch.setattr(sst_app.QMessageBox, "question", lambda *args: asked.append(args) or
                        sst_app.QMessageBox.StandardButton.No)
    assert app.start_live() == "Not started." and len(asked) == 1 and not app.settings.live_told
    started = []

    class FakeSession:  # a LiveSession's ways, without devices or Google
        transcript = speaker = None

        def __init__(self):
            self.lanes = {}

        def add(self, lane, capture, engine_factory):
            self.lanes[lane] = capture
            started.append(True)

        def remove(self, lane):
            self.lanes.pop(lane, None)

        def stop(self):
            self.lanes.clear()
            started.append(False)
    monkeypatch.setattr(app, "_live_session", lambda config, on_event: FakeSession())
    monkeypatch.setattr(app, "_live_lane", lambda lane, config: (lane, None))  # no device is built
    monkeypatch.setattr(sst_app.QMessageBox, "question", lambda *args: asked.append(args) or
                        sst_app.QMessageBox.StandardButton.Yes)
    assert app.start_live() == "" and app.live_running() and app.settings.live_told and app.live_action.isChecked()
    from sst.live.contracts import Kind, LiveEvent
    app.live.event.emit(LiveEvent(Kind.NOTE, "Nothing heard from the microphone", lane="mic"))  # the bar's status line
    assert app.live_note() == "Nothing heard from the microphone"  # the Live page shows it too
    app.stop_live()
    assert not app.live_running() and started == [True, False] and not app.live_action.isChecked()
    assert app.live_note() == ""
    assert app.start_live() == "" and len(asked) == 2  # told once: not asked again
    app.stop_live()


def test_live_translation_hears_the_laptop_and_the_microphone_and_saves_in_the_profile(tray_app):
    app, _, _ = tray_app
    app.gateway = GatewayConfig(provider="gemini", api_key="AIza-test")
    from sst.live.contracts import MIC, SYSTEM
    from sst.live.gemini import GeminiLiveTranslate
    from sst.live.wasapi import Capture
    app.apply_settings(dataclasses.replace(app.settings, live_source="both", live_target="en", live_mic_target="ja"))
    config = app.live_config()
    assert config.lanes == (SYSTEM, MIC) and (config.target, config.mic_target) == ("en", "ja")
    session = app._live_session(config, lambda event: None)
    assert session.transcript.folder == app.profile.folder() / "live captions" and session.transcript.config is config
    for lane, what, target in ((SYSTEM, "output device", "en"), (MIC, "microphone", "ja")):
        capture, engine_factory = app._live_lane(lane, config)  # built, not started: no device is opened
        engine = engine_factory(lambda event: None)
        assert isinstance(capture, Capture) and capture.what == what and isinstance(engine, GeminiLiveTranslate)
        assert (engine.key, engine.config.target, engine.lane) == ("AIza-test", target, lane)


def test_the_source_and_screen_sharing_apply_while_it_runs(tray_app, monkeypatch):
    app, _, _ = tray_app
    calls = []
    monkeypatch.setattr(app.live, "set_source", lambda source: calls.append(("source", source)) or "")
    monkeypatch.setattr(app.live, "set_hidden", lambda hidden: calls.append(("hidden", hidden)))
    monkeypatch.setattr(type(app.live), "running", property(lambda self: True))
    assert app.set_live_source("microphone") == "" and app.settings.live_source == "microphone"
    app.set_live_hidden(False)
    assert not app.settings.live_hide_from_share and calls == [("source", "microphone"), ("hidden", False)]


def test_a_running_live_translation_keeps_the_languages_it_started_with(tray_app, monkeypatch):
    app, _, _ = tray_app
    app.apply_settings(dataclasses.replace(app.settings, live_target="en", live_mic_target="ja"))
    assert app.live_languages() == ("en", "ja")
    monkeypatch.setattr(type(app.live), "running", property(lambda self: True))
    app.live.config = app.live_config()  # what it started with
    app.apply_settings(dataclasses.replace(app.settings, live_target="ta", live_mic_target="ko"))
    assert app.live_languages() == ("en", "ja")  # the page says what it translates into: the new ones wait for a start


def test_ctrl_alt_l_starts_and_stops_live_translation_from_any_app(tray_app, monkeypatch):
    app, _, _ = tray_app
    listener = app.live_listener
    assert listener is not None and listener.running and listener.hotkey.label == "Ctrl+Alt+L"
    assert app.live_action.text() == "Live translation\tCtrl+Alt+L"
    toggled = []
    monkeypatch.setattr(app, "toggle_live", lambda: toggled.append(True))
    listener.send("press")
    listener.send("release")
    assert toggled == [True]  # once per press
    app.apply_settings(dataclasses.replace(app.settings, translate_shortcut="ctrl+alt+l"))  # Translate's keys now
    assert app.live_listener is None and not listener.running and app.live_shortcut_clash() == "Translate"
    assert app.live_action.text() == "Live translation"
    app.apply_settings(dataclasses.replace(app.settings, translate_shortcut="ctrl+c+c", live_shortcut="win+alt+l"))
    assert app.live_listener.hotkey.label == "Win+Alt+L" and app.live_action.text() == "Live translation\tWin+Alt+L"


def test_idle_in_the_tray_nothing_polls(tray_app, monkeypatch):
    """Hidden in the tray, Rflow wakes up for a key event, not 200 times a second to look for one (phase 44)."""
    from PySide6.QtCore import QTimer
    from PySide6.QtTest import QTest
    app, _, _ = tray_app
    app.window.hide()
    controllers = (app.transforms, app.translator)
    assert not app.ticker.isActive()  # no recording, the microphone closed: no housekeeping either
    assert not any(t.isActive() for c in controllers for t in c.findChildren(QTimer))
    handled = []
    monkeypatch.setattr(app.dictation, "handle", lambda event, at: handled.append((event, at)))
    app.listener.send("press", 1.0)  # from the hook's thread in the real app: queued to this one
    app.listener.send("release", 2.0)
    QTest.qWait(10)
    assert handled == [("press", 1.0), ("release", 2.0)]
    # A recording, or the microphone kept open, needs the 1-second housekeeping (the time limit, closing or watching
    # the microphone), and only then.
    warm = {"on": True}
    monkeypatch.setattr(type(app.recorder), "warm", property(lambda self: warm["on"]))
    app._wake_ticker()
    assert app.ticker.isActive() and app.ticker.interval() == 1000
    warm["on"] = False
    app._tick()
    assert not app.ticker.isActive()


def test_where_the_bar_was_left_is_used_at_the_next_start(tray_app, monkeypatch):
    app, _, _ = tray_app
    app._live_moved([40, 50, 600, 220])
    assert app.settings.live_bar == [40, 50, 600, 220]
    started = []
    monkeypatch.setattr(app.live, "start", lambda config, geometry=None: started.append(geometry))
    app.gateway = GatewayConfig(provider="gemini", api_key="AIza-test")
    app.settings = dataclasses.replace(app.settings, live_told=True)
    app.start_live()
    assert started == [[40, 50, 600, 220]]


def test_past_sessions_are_listed_newest_first(tray_app):
    from datetime import datetime
    app, _, _ = tray_app
    folder = app.live_folder()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "2026-10-05 14-03-12 live captions.txt").write_text(
        "Rflow live translation\n\n[14:03:12] a\n           b\n\n[14:03:20] c\n\n", encoding="utf-8")
    (folder / "2026-10-04 09-00-00 live captions.txt").write_text("Rflow live translation\n\n[09:00:00] x\n\n",
                                                                  encoding="utf-8")
    (folder / "notes.txt").write_text("not a session", encoding="utf-8")
    assert [(began, lines) for began, lines, _ in app.live_sessions()] == [
        (datetime(2026, 10, 5, 14, 3, 12), 2), (datetime(2026, 10, 4, 9, 0, 0), 1)]


def test_an_update_that_didnt_install_is_said_with_the_next_offer(tray_app, monkeypatch):
    app, _, _ = tray_app
    told = []
    monkeypatch.setattr(app, "_notify", lambda title, message, icon=None: told.append((title, message)))
    app.install_problem = ("Rflow 9.9.9 didn't install: Visual Studio Code was using Rflow's files. Close it, then click "
                           "Update now.")
    app._on_update_found(sst_app.updates.Update("9.9.9", "", "", "", "", 0))
    assert app.window.banner_text.text() == app.install_problem
    assert told == [("Rflow 9.9.9 didn't install",
                     "Visual Studio Code was using Rflow's files. Close it, then click Update now.")]
    monkeypatch.setattr(sst_app.shell, "open_url", lambda url: True)  # a source checkout shows the release page
    app.start_update()  # tried again: said again only if it fails again
    assert app.install_problem == ""


def test_the_live_transcript_is_copied_from_the_bar_and_the_live_page(tray_app, monkeypatch):
    app, _, _ = tray_app
    copied, said = [], []
    monkeypatch.setattr(sst_app.textaccess, "set_clipboard", copied.append)
    monkeypatch.setattr(app.live, "flash", said.append)
    path = app.live_folder() / "2026-10-05 14-03-12 live captions.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Rflow live translation\n\n[14:03:12] 今日は\n           Today\n\n[14:03:20] Hi\n\n", encoding="utf-8")
    monkeypatch.setattr(type(app.live), "transcript_path", property(lambda self: None))
    app._copy_live_from_bar()
    assert said == ["Nothing to copy yet"] and copied == []
    monkeypatch.setattr(type(app.live), "transcript_path", property(lambda self: path))
    app._copy_live_from_bar()
    assert said[-1] == "Copied 2 lines" and copied[-1] == path.read_text(encoding="utf-8").strip() + "\n"
    assert app.copy_live_session(path, translation_only=True) == "" and copied[-1].endswith("[14:03:12] Today\n[14:03:20] Hi\n")

    def busy(text):
        raise OSError("held by another app")
    monkeypatch.setattr(sst_app.textaccess, "set_clipboard", busy)
    app._copy_live_from_bar()
    assert said[-1] == "Another app is using the clipboard: try again"
    assert app.copy_live_session(path.with_name("gone.txt")) == "The transcript couldn't be read"


def test_gemini_speaks_at_once_and_the_laptops_voice_is_downloaded_to_stand_in(tray_app, monkeypatch):
    app, _, _ = tray_app
    from sst.live import voice as voices
    from sst.live.contracts import MIC, SYSTEM, LiveConfig
    from sst.live.speaker import Speaker
    have = set()
    monkeypatch.setattr(sst_app.Voice, "ready", lambda self, root=None: self.key in have)
    speaker = app._live_speaker(LiveConfig(source="microphone", mic_target="ja", speak=True))
    assert speaker.own == {MIC} and speaker.lanes == set()  # Gemini says Japanese; the laptop's voice speaks English
    assert speaker._load() is None and app.live_voice_state()[0] == "missing"  # nothing to stand in yet
    started = []
    monkeypatch.setattr(app, "download_live_voice", lambda: started.append(True))
    app.set_live_speak(True)
    assert app.settings.live_speak and started == [True]
    have.add(voices.DANNY.key)  # an older Rflow's Danny: speaks until Ryan is here, and Ryan is still downloaded
    assert app.live_voice() is voices.DANNY and app.live_voice_state() == ("ready", 0, "")
    app.set_live_speak(True)
    assert started == [True, True]
    have.add(voices.RYAN.key)
    speaker = app._live_speaker(LiveConfig(target="en", speak=True, speak_speed=1.3, duck=0.25))
    assert isinstance(speaker, Speaker) and speaker.lanes == speaker.own == {SYSTEM} and speaker.speed == 1.3
    assert speaker.ducker.depth == 0.25  # like an interpreter: the other apps kept lower while it runs
    assert app.live_voice() is voices.RYAN and app.live_voice_state() == ("ready", 0, "")
    app.set_live_speak(True)
    assert started == [True, True]  # here: nothing more to download


def test_how_low_the_other_apps_stay_while_it_runs_is_saved_and_used_at_once(tray_app, monkeypatch):
    app, _, _ = tray_app
    depths = []
    monkeypatch.setattr(app.live, "set_duck", depths.append)
    assert app.settings.live_duck == 0.1 and app.live_config().duck == 0.1  # the default: 10%
    app.set_live_duck(0.6)
    assert app.settings.live_duck == 0.6 and app.live_config().duck == 0.6 and depths == [0.6]
    assert sst_app.Settings.load(app.profile.settings_file).live_duck == 0.6
    app.set_live_duck(0.0)  # 0%: kept as the user chose it (the Ducker makes it inaudible, never silence)
    assert sst_app.Settings.load(app.profile.settings_file).live_duck == 0.0 and depths[-1] == 0.0


def test_the_cost_saver_is_saved_used_at_once_and_the_bar_and_page_stay_in_step(tray_app, monkeypatch):
    app, _, _ = tray_app
    app.gateway = GatewayConfig(provider="gemini", api_key="AIza-test")
    app.settings = dataclasses.replace(app.settings, live_told=True)
    assert not app.settings.live_save and not app.live_config().save  # off by default: live translation as before
    saving, notices = [], []

    class FakeSession:  # a LiveSession's ways and spend, without devices or Google
        transcript = speaker = None

        def __init__(self, config):
            self.lanes, self.config = {}, config

        def add(self, lane, capture, engine_factory):
            self.lanes[lane] = capture

        def set_saving(self, on):
            saving.append(on)

        def spend(self):
            return SimpleNamespace(cost=0.42, saved=0.26, seconds=660.0, unsent_s=420.0)

        def quiet_for(self, now=None):
            return 0.0

        def stop(self):
            self.lanes.clear()
    monkeypatch.setattr(app, "_live_session", lambda config, on_event: FakeSession(config))
    monkeypatch.setattr(app, "_live_lane", lambda lane, config: (lane, None))
    monkeypatch.setattr(app, "_notify", lambda *args: notices.append(args))
    assert app.start_live() == "" and not app.live.bar.save_button.isChecked() and app.live_spend() is None
    app.live.bar.save_button.click()  # the bar's button: saved to the profile, on every way at once
    assert app.settings.live_save and saving == [True] and app.live.config.save and app.live.bar.save_button.isChecked()
    assert sst_app.Settings.load(app.profile.settings_file).live_save and app.live_config().save
    assert app.live_spend().cost == 0.42 and app.live.bar.title.text().endswith("$0.42 · saved $0.26")
    app.set_live_save(False)  # the Live page's toggle: the bar follows
    assert not app.settings.live_save and saving == [True, False] and not app.live.bar.save_button.isChecked()
    app.set_live_save(True)
    app.live.session.quiet_for = lambda now=None: 900.0  # nobody spoke for 15 min
    app.live.tick()
    assert not app.live_running() and app.live_stopped_why() == "no speech for 15 min"
    assert app.live_last_spend().saved == 0.26 and notices[-1][1] == "Live translation stopped: no speech for 15 min."
    app.start_live()
    assert app.live.config.save and app.live.bar.save_button.isChecked()  # the next start saves too
    app.stop_live()


def test_the_voices_download_shows_its_progress_and_speaks_when_done(tray_app, monkeypatch):
    app, _, _ = tray_app
    notes, speak = [], []
    monkeypatch.setattr(app.live, "set_voice_note", notes.append)
    monkeypatch.setattr(app.live, "set_speak", speak.append)
    app.settings = dataclasses.replace(app.settings, live_speak=True)
    app.voice_downloading = 0
    app._on_voice_progress(42)
    assert app.live_voice_state() == ("downloading", 42, "") and notes[-1] == "Downloading the voice: 42%"
    app._on_voice_done("")
    assert app.voice_downloading is None and notes[-1] == "" and speak == [True]
    app._on_voice_done("the model server answered 503")
    assert app.live_voice_state()[2] == "the model server answered 503"


# ---- never lose work (phase 30)

def test_crashes_leave_a_report_and_the_next_start_finds_it(tmp_path, monkeypatch, caplog):
    import sys
    import threading
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)  # put back after the test
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    enabled = []
    monkeypatch.setattr(sst_app.faulthandler, "enable", enabled.append)  # not the real one: pytest has its own
    assert sst_app.install_crash_handlers(tmp_path) == ""  # a first start: nothing left behind
    assert (tmp_path / "running").exists() and enabled
    sys.excepthook(ValueError, ValueError("a slot failed"), None)
    assert "Unexpected error" in caplog.text and "a slot failed" in caplog.text
    enabled[0].write("Windows fatal exception: access violation\n")  # what faulthandler writes as Rflow dies
    enabled[0].close()
    left = sst_app.install_crash_handlers(tmp_path)  # the next start: "running" is still there
    sst_app._crash_file.close()
    assert left.startswith("crash-") and "access violation" in (tmp_path / left).read_text(encoding="utf-8")
    sst_app.mark_clean_exit(tmp_path)
    assert not (tmp_path / "running").exists()


def test_quitting_or_signing_out_is_not_a_crash(tmp_path, monkeypatch):
    import sys
    import threading
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    files = []
    monkeypatch.setattr(sst_app.faulthandler, "enable", files.append)
    sst_app.install_crash_handlers(tmp_path)
    files[0].write("a first-chance exception a library caught\n")
    files[0].close()
    sst_app.mark_clean_exit(tmp_path)  # Rflow quit normally afterwards
    assert sst_app.install_crash_handlers(tmp_path) == ""
    sst_app._crash_file.close()


def test_a_start_that_fails_says_why_instead_of_vanishing(tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(sst_app.QMessageBox, "critical", lambda *args: shown.append(args[2]))
    (tmp_path / "running").write_text("1", encoding="utf-8")

    def broken():
        raise PermissionError("C:\\Users\\x\\AppData\\Roaming\\sst is not writable")
    assert sst_app.start(broken, tmp_path) is None
    assert shown[0].startswith("Rflow couldn't start: C:\\Users\\x") and str(tmp_path) in shown[0]
    assert not (tmp_path / "running").exists()  # it said why: not a crash


def test_the_crash_left_behind_is_mentioned_once_rflow_is_up(tmp_path):
    told = []

    class App:
        def _notify(self, title, message, icon):
            told.append(message)
    assert isinstance(sst_app.start(App, tmp_path, left="crash-20261006-101500.log"), App)
    assert told == ["Rflow closed unexpectedly last time. A crash report is in its logs folder "
                    "(crash-20261006-101500.log)."]


def test_a_damaged_dictionary_doesnt_stop_rflow_and_the_user_is_told(tray_app, monkeypatch):
    app, _, _ = tray_app
    told = []
    monkeypatch.setattr(app, "_notify", lambda title, message, icon=None: told.append(message))
    path = app.profile.folder() / "dictionary.db"
    app.dictionary.close()
    for side in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
        if side.exists():
            side.unlink()
    path.write_bytes(b"not a database" * 50)
    app._load_profile()
    assert told and told[0].startswith("Your dictionary file was damaged")
    app.dictionary.add_term("Kubernetes")  # and it works


def test_settings_that_cant_be_saved_are_said_once_and_still_used(tray_app, monkeypatch):
    app, _, _ = tray_app
    told = []
    monkeypatch.setattr(app, "_notify", lambda title, message, icon=None: told.append(message))

    def full_disk(self, path=None):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(Settings, "save", full_disk)
    app.apply_settings(dataclasses.replace(app.settings, sounds=False))
    app.apply_settings(dataclasses.replace(app.settings, sounds=True))
    assert len(told) == 1 and told[0].startswith("Rflow couldn't save your settings") and app.settings.sounds


def test_an_update_never_cuts_a_dictation_or_live_translation(tray_app, monkeypatch):
    app, _, _ = tray_app
    installed, quit_ = [], []
    monkeypatch.setattr(sst_app.updates, "install", installed.append)
    monkeypatch.setattr(sst_app, "release_running_mutex", lambda: None)
    monkeypatch.setattr(app, "quit", lambda: quit_.append(True))
    doing = ["this dictation"]
    monkeypatch.setattr(app, "busy", lambda: doing[0])
    app._on_update_ready("Rflow-Setup.exe")
    assert installed == [] and app._update_timer.isActive() and app._update_waiting == "Rflow-Setup.exe"
    app._install_when_idle()  # still dictating
    assert installed == []
    doing[0] = ""
    app._install_when_idle()  # done: now it installs
    assert [str(p) for p in installed] == ["Rflow-Setup.exe"] and quit_ and not app._update_timer.isActive()


def test_busy_names_what_an_update_waits_for(tray_app, monkeypatch):
    app, _, _ = tray_app
    assert app.busy() == ""
    monkeypatch.setattr(type(app.live), "running", property(lambda self: True))
    assert app.busy() == "live translation stops"


def test_a_history_that_cant_be_written_never_fails_a_dictation(tray_app, monkeypatch):
    app, _, _ = tray_app
    told = []
    monkeypatch.setattr(app, "_notify", lambda title, message, icon=None: told.append(message))

    def unwritable(text, heard=None, path=None):
        raise PermissionError(13, "Access is denied")
    monkeypatch.setattr(sst_app, "add_to_history", unwritable)
    app._on_result("hello", "Hello.", 1.2)
    app._on_result("again", "Again.", 1.0)
    assert len(told) == 1 and told[0].startswith("Rflow couldn't save your dictations to the history")


def test_only_a_real_crash_counts_not_an_exception_windows_handles_itself():
    seen_here = "Windows fatal exception: code 0x8001010d\n\nThread 0x00004804 (most recent call first):\n  File ..."
    assert not sst_app.crashed(seen_here)  # the owner's laptop, 2026-10-06, while Rflow ran on normally
    assert sst_app.crashed(seen_here + "\nWindows fatal exception: access violation\n\nCurrent thread ...")
    assert sst_app.crashed("Fatal Python error: Segmentation fault\n")
    assert sst_app.crashed("Windows fatal exception: code 0xc0000409\n") and not sst_app.crashed("")


# ---- Start over (phase 31): every test works in temporary folders, never in Rflow's real ones

def _rflow_data(root):
    """Rflow's two data folders as they'd be after some use: %APPDATA%\\sst and %LOCALAPPDATA%\\sst."""
    roaming, local = root / "Roaming" / "sst", root / "Local" / "sst"
    for path in (roaming / "settings.json", roaming / "gateway.json", roaming / "profiles" / "rahul" / "dictionary.db",
                 roaming / "live captions" / "2026-10-06 10-00-00 live captions.txt", local / "logs" / "sst.log",
                 local / "bench" / "2026-10-06_100000" / "01.wav", local / "recordings" / "a.wav",
                 local / "models" / "parakeet" / "encoder.int8.onnx"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    return roaming, local


def test_erase_data_deletes_rflows_folders_and_can_keep_the_models(tmp_path):
    roaming, local = _rflow_data(tmp_path)
    neighbour = tmp_path / "Roaming" / "Other app"
    neighbour.mkdir()
    (neighbour / "keep.txt").write_text("x", encoding="utf-8")
    assert sst_app.erase_data([roaming, local], keep=("models",)) == []
    assert not roaming.exists() and [p.name for p in local.iterdir()] == ["models"]
    assert (local / "models" / "parakeet" / "encoder.int8.onnx").exists() and (neighbour / "keep.txt").exists()
    assert sst_app.erase_data([roaming, local]) == [] and not local.exists()  # without keeping the models


def test_erase_data_never_touches_a_folder_not_named_sst(tmp_path):
    home = tmp_path / "home"  # e.g. APPDATA unset: never the user's own folder
    (home / "Documents").mkdir(parents=True)
    (home / "notes.txt").write_text("mine", encoding="utf-8")
    assert sst_app.erase_data([home]) == []
    assert (home / "notes.txt").exists() and (home / "Documents").exists()


def test_erase_data_says_what_another_program_holds_open(tmp_path):
    roaming, _ = _rflow_data(tmp_path)
    with open(roaming / "settings.json", encoding="utf-8"):  # open: Windows doesn't let it be deleted
        left = sst_app.erase_data([roaming])
    assert left == [roaming / "settings.json"] and not (roaming / "profiles").exists()


def test_start_over_stops_everything_deletes_the_data_and_starts_rflow_again(tray_app, tmp_path, monkeypatch):
    app, _, _ = tray_app
    roaming, local = _rflow_data(tmp_path / "data")
    monkeypatch.setattr(sst_app, "DATA_DIRS", (roaming, local))
    events, started = [], []
    erase, close = sst_app.erase_data, app.dictionary.close
    monkeypatch.setattr(sst_app, "close_log_files", lambda: events.append("logs"))
    monkeypatch.setattr(sst_app, "erase_data", lambda folders, keep=(): events.append(("erase", keep)) or erase(folders, keep))
    monkeypatch.setattr(app.dictionary, "close", lambda: (events.append("dictionary"), close()))
    monkeypatch.setattr(sst_app.subprocess, "Popen", lambda command, **options: started.append(command) or events.append("start"))
    monkeypatch.setattr(sst_app.QApplication, "exit", lambda code=0: events.append("exit"))
    app.start_over(keep_models=True)
    assert events == ["dictionary", "logs", ("erase", ("models",)), "start", "exit"]  # files let go of before deleting
    assert not app.listener.running and not app.live.running and not app.tray.isVisible()
    assert not roaming.exists() and [p.name for p in local.iterdir()] == ["models"]
    assert started[0][-1] == f"--after={os.getpid()}"  # the new Rflow waits for this one to end


def test_rflow_started_again_waits_for_the_one_that_started_it():
    import subprocess
    import sys
    import time
    assert sst_app.restart_command(1234)[0] == sys.executable and sst_app.restart_command(1234)[-1] == "--after=1234"
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.5)"])
    began = time.monotonic()
    sst_app.wait_for_exit(child.pid, 10)
    assert child.poll() is not None and time.monotonic() - began >= 0.3


# ---- Start with your voice (sst.wakeui): sst.wake with its models faked (conftest), the microphones are fakes, nothing
# is typed for real

class WakeMic:
    """A microphone kept open (always-on), faked: start() opens a take whose pre-roll holds the keyword; stop_later()
    adds a second of speech after it."""
    rate, raw, device, notice, warm = 16_000, False, None, "", True

    def __init__(self, name="Mic A"):
        self.name, self.tail, self.warm_seconds, self.preroll_seconds = name, 0.3, 0.0, 0.4
        self.tap, self.level, self.current_take, self.opened, self.closed = None, 0.0, None, 0, False

    @property
    def busy(self):
        return self.current_take is not None

    def keep_open(self):
        self.opened += 1

    def start(self):
        self.current_take = Take([np.full(16_000, 0.1, dtype=np.float32)], self.rate)

    def stop_later(self):
        take, self.current_take = self.current_take, None
        take.chunks.append(np.full(16_000, 0.1, dtype=np.float32))
        take.done.set()
        return take

    def tick(self, now):
        pass

    def close(self):
        self.closed = True

    def describe(self):
        return {"device": self.name}


@pytest.fixture
def wake_app(tray_app, wake, monkeypatch, tmp_path):
    """The tray app with the wake word on and set up ("okay type"), its models downloaded, listening on a fake
    always-on microphone; dictation types into a list, without sounds, through the classic path. Its listeners are made
    at once, not on a thread."""
    from sst import audio, downloads, wakeui
    app, saved, history = tray_app
    app.wake.background = lambda work, done: done(work(), None)
    monkeypatch.setattr(downloads.Download, "installed", lambda self, root=None: True)
    monkeypatch.setattr(audio, "call_quality", lambda device: False)
    monkeypatch.setattr(audio, "input_device_names", lambda refresh=True: ["Mic A"])
    monkeypatch.setattr(audio, "default_microphone", lambda: "Mic A")
    own = []
    monkeypatch.setattr(wakeui, "Recorder", lambda device: own.append(WakeMic(device)) or own[-1])
    mic = WakeMic()
    app.recorder = app.dictation.recorder = mic
    typed = []
    app.dictation.paste, app.dictation.sounds, app.dictation.save = typed.append, False, False
    app.wake.setup = wake.WakeSetup("okay type", "okay type", wake.VoicePrint([0.1]), 9)
    app.apply_settings(dataclasses.replace(app.settings, wake_enabled=True, voice_pipeline=False))
    yield SimpleNamespace(app=app, wake=wake, mic=mic, typed=typed, own=own, saved=saved, history=history)
    app.wake.stop()


def _heard(app, text):
    app.dictation.engine.transcribe = lambda audio, rate: text


def test_the_wake_word_listens_on_the_always_on_microphone_once_set_up(wake_app):
    app, mic = wake_app.app, wake_app.mic
    listener = wake_app.wake.WakeListener.made[-1]
    assert app.wake.listening and app.wake.shared and mic.tap == app.wake.feeder.put  # every block goes to it
    assert mic.warm_seconds == float("inf") and mic.preroll_seconds >= 2.0  # the words right after the keyword kept
    assert listener.end_phrases == ["stop it"] and listener.voice_check
    state = app.wake_state()
    assert (state.keyword, state.heard, state.models, state.listening, state.note) == ("okay type", 9, "ready", "Mic A", "")
    app.apply_settings(dataclasses.replace(app.settings, wake_end_phrases=["stop it", "end it"], wake_voice_check=False))
    listener = wake_app.wake.WakeListener.made[-1]
    assert listener.end_phrases == ["stop it", "end it"] and not listener.voice_check
    app.apply_settings(dataclasses.replace(app.settings, wake_enabled=False))
    assert not app.wake.listening and mic.tap is None and app.wake_state().listening == ""


def test_a_turn_types_the_words_between_the_keyword_and_the_end_phrase(wake_app):
    app, listener = wake_app.app, wake_app.wake.WakeListener.made[-1]
    _heard(app, "Okay type, send the report tomorrow. Stop it.")
    listener.on_wake()
    assert app.dictation.recording and app.wake.active and app.pill.state == "recording"  # as Ctrl+Win+Space
    assert app.dictation.trim is None  # it went with this recording only
    listener.on_end(True)
    app.dictation.wait()
    assert wake_app.typed == ["send the report tomorrow. "] and not app.wake.active and not app.dictation.recording


def test_a_voice_that_isnt_the_users_is_dropped_without_a_sound(wake_app):
    app, listener = wake_app.app, wake_app.wake.WakeListener.made[-1]
    states = []
    app.dictation.on_state = lambda state, message: states.append(state)
    listener.on_wake()
    listener.on_end(False)
    app.dictation.wait()
    assert wake_app.typed == [] and states == ["recording", "idle"]  # no "Cancelled", nothing typed or kept


def test_ctrl_win_or_esc_still_end_a_turn_and_the_listener_waits_again(wake_app):
    from PySide6.QtTest import QTest
    app, listener = wake_app.app, wake_app.wake.WakeListener.made[-1]
    _heard(app, "Okay type, hello there.")
    listener.on_wake()
    app.listener.events.put(("press", sst_app.time.monotonic()))  # Ctrl+Win
    app._pump()
    app.dictation.wait()
    assert wake_app.typed == ["hello there. "] and not app.wake.active
    assert _wait_for(lambda: listener.ended == 1, QTest.qWait)  # told on its own thread: back to waiting
    listener.on_wake()
    app.listener.events.put(("cancel", sst_app.time.monotonic()))  # Esc
    app._pump()
    app.dictation.wait()
    assert wake_app.typed == ["hello there. "] and not app.wake.active


def test_a_command_said_after_the_keyword_runs_on_the_text_box_without_waiting(wake_app, monkeypatch):
    from PySide6.QtTest import QTest
    app, listener = wake_app.app, wake_app.wake.WakeListener.made[-1]
    ran = []
    monkeypatch.setattr(app, "transform_ready", lambda: True)
    monkeypatch.setattr(app.transforms, "run_command", ran.append)
    _heard(app, "Okay type, hello there")
    listener.on_wake()
    listener.on_pause()  # a few words, then a pause: not a command, dictation goes on
    QTest.qWait(200)
    assert app.dictation.recording and ran == []
    _heard(app, "Okay type, make it concise.")
    listener.on_pause()
    assert _wait_for(lambda: ran == ["concise"], QTest.qWait)
    app.dictation.wait()
    assert not app.dictation.recording and not app.wake.active and wake_app.typed == []  # not typed: a command
    assert _wait_for(lambda: listener.ended == 1, QTest.qWait)


def test_the_listener_is_made_on_a_thread_and_its_status_and_ticks_follow_it(wake_app, monkeypatch):
    # Its spotter loads a model the first time (up to ~2 s): never on the UI thread. Nothing polls for its status.
    from PySide6.QtTest import QTest

    from sst import wakeui
    app, wake = wake_app.app, wake_app.wake
    app.wake.background = lambda work, done: wakeui._in_background(app.wake, work, done)
    told = []
    monkeypatch.setattr(app.window, "isVisible", lambda: True)
    monkeypatch.setattr(app.window, "show_wake", lambda: told.append(app.wake_state().listening))
    app.ticker.stop()
    app.apply_settings(dataclasses.replace(app.settings, wake_end_phrases=["stop it", "end it"]))
    assert not app.wake.listening and told == [""]  # being made
    assert _wait_for(lambda: app.wake.listening, QTest.qWait)
    assert wake.WakeListener.made[-1].thread != "MainThread" and told == ["", "Mic A"] and app.ticker.isActive()
    app.apply_settings(dataclasses.replace(app.settings, wake_end_phrases=["end it"]))
    app.apply_settings(dataclasses.replace(app.settings, wake_enabled=False))  # off before the new one was made
    QTest.qWait(300)
    assert not app.wake.listening and wake.WakeListener.made[-1].end_phrases == ["end it"]  # made, then dropped


def test_listening_pauses_for_live_translation_dictation_with_the_keys_and_text_transform(wake_app):
    from sst.live.contracts import LiveConfig
    app = wake_app.app
    app.wake.tick(sst_app.time.monotonic())
    assert not app.wake.feeder.paused and app.wake_state().note == ""
    app.dictation.handle("press", sst_app.time.monotonic())  # Ctrl+Win: a dictation of its own
    app.wake.tick(sst_app.time.monotonic())
    assert app.wake.feeder.paused and "dictate" in app.wake_state().note
    app.dictation.end(keep=False, quiet=True)
    app.transforms.busy = True
    app.wake.tick(sst_app.time.monotonic())
    assert app.wake.feeder.paused and "Text Transform" in app.wake_state().note
    app.transforms.busy = False
    app.live.session, app.live.config = SimpleNamespace(stop=lambda: None), LiveConfig(source="computer")
    app.wake.tick(sst_app.time.monotonic())
    assert not app.wake.feeder.paused  # live translation of the laptop's sound leaves the microphone alone
    app.live.config = LiveConfig(source="both")
    app.wake.tick(sst_app.time.monotonic())
    assert app.wake.feeder.paused and "live translation" in app.wake_state().note
    app.live.session = None
    app.wake.tick(sst_app.time.monotonic())
    assert not app.wake.feeder.paused


def test_with_a_bluetooth_headset_chosen_the_built_in_microphone_listens_and_records(wake_app, monkeypatch):
    from sst import audio
    app = wake_app.app
    monkeypatch.setattr(audio, "call_quality", lambda device: device in ("Headset (Buds)", None))
    monkeypatch.setattr(audio, "input_device_names", lambda refresh=True: ["Headset (Buds)", "Mic A"])
    app.apply_settings(dataclasses.replace(app.settings, microphone="Headset (Buds)"))
    own = wake_app.own[-1]
    assert app.wake.listening and not app.wake.shared and own.name == "Mic A" and app.wake_state().listening == "Mic A"
    assert own.tap == app.wake.feeder.put and own.warm_seconds == float("inf") and wake_app.mic.tap is None
    listener = wake_app.wake.WakeListener.made[-1]
    _heard(app, "Okay type, hello. Stop it.")
    listener.on_wake()
    assert app.dictation.recorder is own  # recorded where the keyword was heard: never the headset's microphone
    listener.on_end(True)
    app.dictation.wait()
    assert wake_app.typed == ["hello. "] and app.dictation.recorder is wake_app.mic
    monkeypatch.setattr(audio, "input_device_names", lambda refresh=True: ["Headset (Buds)"])
    app.apply_settings(dataclasses.replace(app.settings, raw_audio=True))  # applied again: no wired microphone now
    assert not app.wake.listening and own.closed and app.wake_state().note == BLUETOOTH


def test_it_listens_on_the_bluetooth_headset_when_the_user_lets_it(wake_app, monkeypatch):
    from sst import audio
    app, mic = wake_app.app, wake_app.mic
    monkeypatch.setattr(audio, "call_quality", lambda device: device in ("Headset (Buds)", None))
    monkeypatch.setattr(audio, "input_device_names", lambda refresh=True: ["Headset (Buds)", "Mic A"])
    app.apply_settings(dataclasses.replace(app.settings, microphone="Headset (Buds)", wake_bluetooth=True))
    assert app.wake.listening and app.wake.shared and mic.tap == app.wake.feeder.put
    assert mic.keep_call_quality and mic.warm_seconds == float("inf")  # kept open on the headset, as the user chose
    state = app.wake_state()
    assert state.listening == "Headset (Buds)" and state.instead == ""
    app.apply_settings(dataclasses.replace(app.settings, wake_bluetooth=False))
    assert app.wake.listening and not app.wake.shared and wake_app.own[-1].name == "Mic A"
    assert not mic.keep_call_quality and mic.tap is None  # dictation's headset opens only to dictate again
    state = app.wake_state()
    assert state.listening == "Mic A" and state.instead == "Headset (Buds)"


def test_the_setup_records_where_the_wake_word_listens_and_learns_from_the_tries(wake_app, monkeypatch):
    from sst import audio
    from sst.live import voice
    app, wake = wake_app.app, wake_app.wake
    monkeypatch.setattr(audio, "call_quality", lambda device: device in ("Headset (Buds)", None))
    monkeypatch.setattr(audio, "input_device_names", lambda refresh=True: ["Headset (Buds)", "Mic A"])
    monkeypatch.setattr(voice.Voice, "ready", lambda self, root=None: False)  # no synthetic voice: no check
    recorders, shown = [], []
    monkeypatch.setattr(sst_app, "Recorder", lambda device: recorders.append(SimpleNamespace(device=device)) or recorders[-1])

    class Dialog:
        def __init__(self, parent, **options):
            self.options = options
            shown.append(self)

        def exec(self):
            self.setting_up = app.wake.setting_up
            return 0
    monkeypatch.setattr(sst_app, "WakeSetupDialog", Dialog)
    app.apply_settings(dataclasses.replace(app.settings, microphone="Headset (Buds)", wake_bluetooth=True))
    app.open_wake_setup()
    options = shown[0].options
    assert options["microphone"] == "Headset (Buds)" and options["problem"] == ""  # where it listens: said, and used
    recorder = options["new_recorder"]()
    assert recorder.device == "Headset (Buds)" and recorder.keep_call_quality
    assert (options["tries"], options["passes"], options["misses"], options["quiet_db"]) == (
        wake.TRIES, wake.PASS, wake.MISSES, wake.QUIET_DB)
    assert options["hearing"]("karthi") is None  # skipped without the voice
    assert isinstance(options["enroll"]("karthi"), wake.Enrollment)
    assert shown[0].setting_up and not app.wake.setting_up  # the tries never start a dictation
    app.apply_settings(dataclasses.replace(app.settings, wake_bluetooth=False))
    app.open_wake_setup()
    assert shown[1].options["microphone"] == "Mic A" and shown[1].options["new_recorder"]().device == "Mic A"


def test_a_saved_setup_belongs_to_its_profile_and_goes_with_it(wake_app):
    app, wake = wake_app.app, wake_app.wake
    setup = wake.WakeSetup("computer", "computer", wake.VoicePrint([0.3, 0.4]), 10)
    app.save_wake_setup(setup)
    assert settings_module.read_wake(app.profile.wake_file) == setup.to_json() and app.wake.listening
    assert wake.WakeListener.made[-1].setup.keyword == "computer"
    app.create_profile("Rahul")  # its own: none yet, and off
    assert app.wake.setup is None and not app.wake.listening
    rahul = app.profile
    app.save_wake_setup(wake.WakeSetup("okay type", "okay type", None, 8))
    assert rahul.wake_file.exists()
    app.switch_profile("default")
    assert app.wake.setup.keyword == "computer" and app.wake.listening
    app.delete_profile("rahul")
    assert not rahul.wake_file.exists()


def test_switching_on_downloads_the_models_once_with_their_progress(tray_app, wake, monkeypatch):
    from PySide6.QtTest import QTest

    from sst import downloads
    app, _, _ = tray_app
    fetched, shown = [], []

    def download(model, progress, cancelled):
        progress(model.size // 2, model.size)
        progress(model.size, model.size)
        fetched.append(model.folder)
    monkeypatch.setattr(downloads, "download", download)
    monkeypatch.setattr(downloads.Download, "installed", lambda self, root=None: self.folder in fetched)
    monkeypatch.setattr(app.window, "refresh", lambda: shown.append(app.wake_state().percent))
    monkeypatch.setattr(app.window, "isVisible", lambda: True)
    state = app.wake_state()
    assert state.models == "missing" and state.size == wake.KWS.size + wake.VOICE.size and state.keyword == ""
    app.set_wake_enabled(True)
    assert app.settings.wake_enabled and app.wake_state().models == "downloading"
    assert _wait_for(lambda: app.wake_downloading is None, QTest.qWait)
    assert fetched == [wake.KWS.folder, wake.VOICE.folder] and 100 in shown and app.wake_problem == ""
    assert app.wake.problem == "Set up your keyword to start."  # downloaded: the setup is next



# ---- Meeting mode (phase 49): every part faked, nothing recorded, looked at or sent

class FakeMeetingPart:
    """A recorder or a window's frames: started and stopped, nothing opened."""

    def __init__(self, folder, log):
        self.folder, self.log, self.seconds, self.frames = folder, log, 0.0, []

    def start(self):
        self.log.append(("start", type(self).__name__))

    def stop(self):
        self.log.append(("stop", type(self).__name__))
        self.seconds = 125.0


@pytest.fixture
def meeting_app(tray_app, monkeypatch):
    """The tray app with a meeting in Teams to be found, its parts faked and Gemini's minutes written at once."""
    from sst.meeting import apps
    from sst.meeting.minutes import MinutesResult
    app, _, _ = tray_app
    app.gateway = GatewayConfig(provider="gemini", api_key="AIza-test")
    teams = apps.MeetingWindow("Microsoft Teams", "Weekly sync | Microsoft Teams", 101, 7)
    monkeypatch.setattr(apps, "find", lambda: [teams])
    monkeypatch.setattr(apps, "alive", lambda window: True)  # the meeting window stays open
    log, shown, opened = [], [], []
    monkeypatch.setattr(app, "_meeting_recorder", lambda folder: FakeMeetingPart(folder, log))
    monkeypatch.setattr(app, "_meeting_frames", lambda window, folder: FakeMeetingPart(folder, log))

    def summarise(folder):
        log.append(("summarise", folder))
        (folder / "minutes.md").write_text("# Weekly sync\n\n- [ ] Send the slides (Priya)\n", encoding="utf-8")
        return MinutesResult("Weekly sync: the release moves to Friday", "We met.")
    monkeypatch.setattr(app, "_summarise_meeting", summarise)
    monkeypatch.setattr(app.tray, "showMessage", lambda title, message, *_: shown.append(message))
    monkeypatch.setattr(sst_app.shell, "open_url", lambda url: opened.append(sst_app.Path(url.toLocalFile())))
    monkeypatch.setattr(app, "start_live", lambda: log.append(("start", "live")) or "")
    monkeypatch.setattr(app, "stop_live", lambda: log.append(("stop", "live")))
    return SimpleNamespace(app=app, teams=teams, log=log, shown=shown, opened=opened)


def test_a_meeting_asks_once_records_translates_then_writes_its_minutes(meeting_app, monkeypatch):
    from PySide6.QtTest import QTest
    m, asked = meeting_app, []
    app = m.app
    assert app.meeting_windows() == [m.teams] and app.meeting is None and app.meeting_cost() is None
    monkeypatch.setattr(sst_app.QMessageBox, "question", lambda *args: asked.append(args) or
                        sst_app.QMessageBox.StandardButton.No)
    assert app.start_meeting(m.teams) == "Not started." and not app.settings.meeting_told and not m.log
    assert "Tell the people in the meeting that it is recorded" in asked[0][2]
    monkeypatch.setattr(sst_app.QMessageBox, "question", lambda *args: asked.append(args) or
                        sst_app.QMessageBox.StandardButton.Yes)
    assert app.start_meeting(m.teams) == "" and app.settings.meeting_told and app.settings.live_told
    session = app.meeting
    assert session.folder.parent == app.profile.folder() / "meetings" and session.folder.name.endswith(" Microsoft Teams")
    assert m.log == [("start", "FakeMeetingPart"), ("start", "FakeMeetingPart"), ("start", "live")]
    assert app.meeting_action.isChecked() and app.tray.toolTip() == "Rflow: recording the meeting"
    assert m.shown[-1] == "Recording the meeting in Microsoft Teams."
    assert app.meeting_cost().seconds >= 0 and app.past_meetings() == []  # the one recording isn't a past one yet
    app.stop_meeting()
    session.wait(5)
    assert _wait_for(lambda: app.meeting_status()[0] == "done", QTest.qWait)
    QTest.qWait(20)  # the queued change reaches the tray
    assert ("stop", "live") in m.log and ("summarise", session.folder) in m.log and app.meeting is None
    assert not app.meeting_action.isChecked() and app.tray.toolTip() != "Rflow: recording the meeting"
    assert m.shown[-1] == "Minutes ready: Weekly sync: the release moves to Friday"
    app._message_clicked()  # the notification clicked: the minutes open
    assert m.opened == [session.folder / "minutes.md"]
    assert [f for f, *_ in app.past_meetings()] == [session.folder]
    assert app.start_meeting(m.teams) == "" and len(asked) == 2  # told once: not asked again
    assert app.meeting.folder != session.folder  # a second one in the same minute: a folder of its own
    app.stop_meeting()
    app.last_meeting.wait(5)


def test_past_meetings_are_listed_newest_first_and_their_minutes_open_and_copy(meeting_app, monkeypatch):
    app = meeting_app.app
    root = app.meetings_folder()
    for name, meta, minutes in (("2026-10-24 1530 Zoom", {"app": "Zoom", "seconds": 2710.0}, True),
                                ("2026-10-27 0900 Google Meet", {"app": "Google Meet", "seconds": 60}, False),
                                ("not a meeting", {}, False)):
        (root / name).mkdir(parents=True)
        (root / name / "meeting.json").write_text(sst_app.json.dumps(meta), encoding="utf-8")
        if minutes:
            (root / name / "minutes.md").write_text("# Planning\n", encoding="utf-8")
    found = app.past_meetings()
    assert [(f.name, started.day, name, seconds, has) for f, started, name, seconds, has in found] == [
        ("2026-10-27 0900 Google Meet", 27, "Google Meet", 60.0, False), ("2026-10-24 1530 Zoom", 24, "Zoom", 2710.0, True)]
    copied = []
    monkeypatch.setattr(sst_app.textaccess, "set_clipboard", lambda text: copied.append(text))
    assert app.copy_minutes(found[1][0]) == "" and copied == ["# Planning\n"]
    assert app.copy_minutes(found[0][0]) == "The minutes couldn't be read"
    app.open_meeting(found[1][0])
    app.open_meeting(found[0][0])  # no minutes yet: its folder
    app.open_meeting_folder()
    assert meeting_app.opened == [root / "2026-10-24 1530 Zoom" / "minutes.md", root / "2026-10-27 0900 Google Meet", root]


def test_a_past_meetings_minutes_are_written_again_from_its_recording(meeting_app, monkeypatch):
    from PySide6.QtTest import QTest

    from sst.meeting.minutes import MinutesError
    app, m = meeting_app.app, meeting_app
    folder = app.meetings_folder() / "2026-10-27 0900 Google Meet"
    folder.mkdir(parents=True)
    app.write_minutes(folder)
    app.write_minutes(folder)  # a second click while it's written: nothing more
    assert _wait_for(lambda: not app.writing_minutes, QTest.qWait)
    assert m.log.count(("summarise", folder)) == 1 and m.shown[-1] == "Minutes ready: Weekly sync: the release moves to Friday"
    monkeypatch.setattr(app, "_summarise_meeting", lambda f: (_ for _ in ()).throw(MinutesError("Gemini didn't answer in time.")))
    app.write_minutes(folder)
    assert _wait_for(lambda: not app.writing_minutes, QTest.qWait)
    assert m.shown[-1] == "The minutes couldn't be written: Gemini didn't answer in time. The recording is kept."


def test_a_meeting_needs_a_part_and_a_gemini_key_for_translation_and_minutes(meeting_app):
    app = meeting_app.app
    app.settings = dataclasses.replace(app.settings, meeting_told=True)
    app.gateway = GatewayConfig()
    assert "add a Gemini key" in app.start_meeting(meeting_app.teams)
    for part in ("translate", "minutes"):
        app.set_meeting_option(part, False)
    assert app.meeting_problem() == "" and not app.settings.meeting_minutes  # recording alone needs no key
    app.set_meeting_option("record", False)
    assert "Switch on live translation or Record" in app.start_meeting(meeting_app.teams)
    assert app.start_meeting(None).startswith("Start a meeting first")
    with pytest.raises(ValueError):
        app.set_meeting_option("everything", True)


def test_ctrl_alt_m_starts_and_stops_the_meeting_found_from_any_app(meeting_app, monkeypatch):
    from sst.meeting import apps
    app = meeting_app.app
    listener = app.meeting_listener
    assert listener is not None and listener.running and listener.hotkey.label == "Ctrl+Alt+M"
    assert app.meeting_action.text() == "Meeting mode\tCtrl+Alt+M"
    started = []
    monkeypatch.setattr(app, "start_meeting", lambda window: started.append(window) or "")
    listener.send("press")
    listener.send("release")
    assert started == [meeting_app.teams]  # once per press, in the one meeting found
    opened = []
    monkeypatch.setattr(app.window, "open", lambda page=None: opened.append(page))
    monkeypatch.setattr(apps, "find", lambda: [])
    app.toggle_meeting()
    assert opened == ["meeting"] and meeting_app.shown[-1].startswith("Start a meeting first")
    app.apply_settings(dataclasses.replace(app.settings, live_shortcut="ctrl+alt+m"))  # live translation's keys now
    assert app.meeting_listener is None and not listener.running and app.meeting_shortcut_clash() == "Live translation"
    assert app.meeting_action.text() == "Meeting mode"
