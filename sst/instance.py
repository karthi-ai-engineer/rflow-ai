"""Is Rflow running already? Asked before the app's heavy imports (Qt's widgets, numpy, the audio and speech modules:
1.5-10 s), so that opening Rflow while it runs shows the running copy's window at once. Only the standard library here,
and Qt's network module when there is a running copy to call."""
import ctypes
import os
import sys

APP_NAME = "Rflow"
# A named mutex shared by every copy (tray app, console, source checkout; sst.dictate.already_running asks the same);
# Windows frees it when the process ends.
MUTEX = "SST-Dictation-dictate"
# Opening Rflow while it runs asks the running copy, through this local pipe, to show its window.
SERVER_NAME = f"Rflow-window-{os.environ.get('USERNAME', 'user')}"
ASFW_ANY = -1  # AllowSetForegroundWindow: any process
AFTER = "--after="  # Rflow started again by Start over waits for the one that started it (its process id) to end

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.CreateMutexW.restype = ctypes.c_void_p
_kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
_kernel32.OpenProcess.restype = ctypes.c_void_p
_kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
_kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
_kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
_handed_over: bool | None = None  # hand_over's answer: asked once per process (launch, then sst.app.main)


def already_running(name: str | None = None) -> bool:
    """True if another process holds the mutex. This process holds it from now on (the handle is kept until it ends)."""
    _kernel32.CreateMutexW(None, False, name or MUTEX)
    return ctypes.get_last_error() == 183  # ERROR_ALREADY_EXISTS


def wait_for_exit(pid: int, seconds: float = 30) -> None:
    """Wait for the Rflow that started this one to end: until then it holds the mutex that says Rflow runs, and this
    one would only ask it to show its window."""
    SYNCHRONIZE = 0x00100000
    handle = _kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if handle:  # none: it has ended already
        _kernel32.WaitForSingleObject(handle, int(seconds * 1000))
        _kernel32.CloseHandle(handle)


def show_running_window(name: str | None = None) -> bool:
    """Ask a running Rflow to show its window; False if none answers (e.g. the console `sst dictate` or 1.1 and older)."""
    from PySide6.QtNetwork import QLocalSocket

    socket = QLocalSocket()
    socket.connectToServer(name or SERVER_NAME)
    if not socket.waitForConnected(1500):
        return False
    # Windows only lets the process the user just started take the foreground. Pass that on to the running copy,
    # or its window would only flash in the taskbar.
    ctypes.windll.user32.AllowSetForegroundWindow(ASFW_ANY)
    socket.write(b"show\n")
    socket.waitForBytesWritten(1500)
    socket.disconnectFromServer()
    return True


def _say(message: str) -> None:
    MB_ICONINFORMATION = 0x40
    ctypes.windll.user32.MessageBoxW(None, message, APP_NAME, MB_ICONINFORMATION)


def hand_over(argv: list[str]) -> bool:
    """True if Rflow runs already: it was asked to show its window (not when this one started at sign-in, --startup),
    and this one should end. Started again by Start over (--after=<pid>), first wait for the old one to end."""
    global _handed_over
    if _handed_over is None:
        after = next((a.removeprefix(AFTER) for a in argv if a.startswith(AFTER)), "")
        if after.isdigit():
            wait_for_exit(int(after))
        _handed_over = already_running()
        if _handed_over and "--startup" not in argv and not show_running_window():
            _say(f"{APP_NAME} is already running. Look for its icon in the tray.")
    return _handed_over


def launch(argv: list[str] | None = None) -> int:
    """Rflow.exe's entry (and `sst app`): hand over to a running Rflow, else import the app and start it."""
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" not in argv and hand_over(argv):
        return 0
    from sst.app import main

    return main(argv)
