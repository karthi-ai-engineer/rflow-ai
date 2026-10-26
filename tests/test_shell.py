"""Programs Rflow opens get Windows' own DLL search, not Rflow's folder (sst.shell)."""
import ctypes
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from sst import shell  # noqa: E402

kernel32 = ctypes.WinDLL("kernel32")
kernel32.GetDllDirectoryW.argtypes = [ctypes.c_ulong, ctypes.c_wchar_p]
kernel32.SetDllDirectoryW.argtypes = [ctypes.c_wchar_p]


def _dll_directory() -> str:
    folder = ctypes.create_unicode_buffer(32768)
    kernel32.GetDllDirectoryW(len(folder), folder)
    return folder.value


def test_while_rflow_starts_a_program_its_folder_is_out_of_the_dll_search(tmp_path, monkeypatch):
    kernel32.SetDllDirectoryW(str(tmp_path))  # what PyInstaller's bootloader does for the installed Rflow
    try:
        seen = []
        monkeypatch.setattr(shell.QDesktopServices, "openUrl", lambda url: seen.append((url.toString(), _dll_directory())))
        shell.open_url("https://example.com")
        assert seen == [("https://example.com", "")]  # what the program started then inherits: nothing of Rflow's
        assert _dll_directory() == str(tmp_path)  # Rflow's own search again right after
    finally:
        kernel32.SetDllDirectoryW(None)


def test_without_a_folder_set_none_is_set_after(monkeypatch):
    kernel32.SetDllDirectoryW(None)  # a source checkout
    monkeypatch.setattr(shell.QDesktopServices, "openUrl", lambda url: True)
    assert shell.open_url("https://example.com")
    assert _dll_directory() == ""
