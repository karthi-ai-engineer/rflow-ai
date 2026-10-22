"""Opening Rflow while it runs: the second copy asks the first to show its window before importing the app (sst.instance).
Each check runs a separate Python, since the mutex and the answer are the process's own; test names, never Rflow's."""
import ctypes
import os
import subprocess
import sys
import time

import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtNetwork import QLocalServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEAVY = ("numpy", "sounddevice", "PySide6.QtWidgets", "sst.app", "sst.audio", "sst.window", "sst.engines")


@pytest.fixture(scope="module", autouse=True)
def qt():
    return QCoreApplication.instance() or QCoreApplication([])


def _second_copy(name: str, argv: list[str]) -> subprocess.Popen:
    code = ("import sys; from sst import instance; "
            f"instance.MUTEX = instance.SERVER_NAME = {name!r}; instance._say = print; "
            f"print('code', instance.launch({argv!r})); "
            f"print('heavy', sorted(m for m in {HEAVY!r} if m in sys.modules))")
    env = {**os.environ, "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    return subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env=env, creationflags=subprocess.CREATE_NO_WINDOW)


def _running(name: str) -> tuple[int, QLocalServer]:
    """A stand-in for the running Rflow: it holds the mutex and listens on the pipe."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
    mutex = kernel32.CreateMutexW(None, False, name)
    server = QLocalServer()
    assert server.listen(name)
    return mutex, server


def _close(mutex: int, server: QLocalServer) -> None:
    server.close()
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle(mutex)


def test_a_second_rflow_hands_over_before_importing_the_app():
    name = f"Rflow-test-instance-{os.getpid()}"
    mutex, server = _running(name)
    try:
        t0 = time.perf_counter()
        child = _second_copy(name, [])
        assert server.waitForNewConnection(20000)[0]  # (connected, timed out)
        connection = server.nextPendingConnection()
        for _ in range(100):
            if connection.bytesAvailable() or not connection.waitForReadyRead(100):
                break
        said = bytes(connection.readAll()).decode()
        out, _ = child.communicate(timeout=20)
        took = time.perf_counter() - t0
        print(f"handed over in {took * 1000:.0f} ms")
    finally:
        _close(mutex, server)
    assert said == "show\n"
    assert "code 0" in out and "heavy []" in out, out  # nothing of the app, numpy, audio or Qt's widgets was imported
    assert took < 10  # well under 0.5 s on the laptop: Python and Qt's network module only


def test_started_at_sign_in_while_rflow_runs_it_only_ends():
    name = f"Rflow-test-instance-startup-{os.getpid()}"
    mutex, server = _running(name)
    try:
        out, _ = _second_copy(name, ["--startup"]).communicate(timeout=20)
        assert not server.waitForNewConnection(200)[0]  # the running one isn't disturbed
    finally:
        _close(mutex, server)
    assert "code 0" in out and "heavy []" in out, out


def test_the_first_rflow_asks_once_and_keeps_the_mutex():
    name = f"Rflow-test-instance-first-{os.getpid()}"
    code = ("from sst import instance; "
            f"instance.MUTEX = {name!r}; "
            "print(instance.hand_over([]), instance.hand_over([]), instance.already_running())")
    env = {**os.environ, "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=20).stdout
    # Not running: start (and app.main asking again gets the same answer); the mutex is this process's from then on.
    assert out.split() == ["False", "False", "True"]
