"""Opening the user's own programs (a transcript in their editor, a folder, a web page) without lending them Rflow's DLLs.

The installed Rflow is a PyInstaller program: its bootloader points Windows' DLL search at Rflow's own folder
(SetDllDirectory), and the programs Rflow starts inherit it. An editor opened from Rflow then loads Rflow's
VCRUNTIME140.dll and keeps it open for as long as it runs, so the next update can't replace it and the installer rolls
back (2026-10-26: VS Code, opened by the Live page's Open, held 2.6.1 back three times). So the search is Windows' own
while Rflow starts something, and Rflow's folder again right after."""
import contextlib
import ctypes
import sys

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices


@contextlib.contextmanager
def plain_dll_search():
    """Windows' own DLL search order meanwhile (what a program started now inherits), then Rflow's again."""
    if sys.platform != "win32":
        yield
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetDllDirectoryW.argtypes = [ctypes.c_ulong, ctypes.c_wchar_p]
    kernel32.SetDllDirectoryW.argtypes = [ctypes.c_wchar_p]
    folder = ctypes.create_unicode_buffer(32768)
    had = kernel32.GetDllDirectoryW(len(folder), folder)  # 0: none set (a source checkout)
    kernel32.SetDllDirectoryW(None)
    try:
        yield
    finally:
        if had:
            kernel32.SetDllDirectoryW(folder.value)


def open_url(url: QUrl | str) -> bool:
    """A file, folder or web page in the program Windows opens it with (QDesktopServices.openUrl, which starts it now)."""
    with plain_dll_search():
        return QDesktopServices.openUrl(url if isinstance(url, QUrl) else QUrl(url))
