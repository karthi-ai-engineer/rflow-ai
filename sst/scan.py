"""Scan my computer: what this computer has, and how each speech model would run on it.

People who choose "On this computer" don't know which model their computer can run. The scan reads the processor,
memory, free disk space and graphics cards (through Windows itself, no extra tools), runs a one-second processor
benchmark, and times the models that are already on the computer on a short recording. Models not downloaded yet are
estimated from the benchmark, scaled from the reference laptop where every number was measured. Each model then gets a
verdict in plain words.
"""
import ctypes
import json
import logging
import os
import shutil
import struct
import sysconfig
import threading
import time
import winreg
from ctypes import wintypes
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from sst import DOWNLOADS_DIR
from sst.settings import CONFIG_DIR

SCAN_FILE = CONFIG_DIR / "scan.json"  # the last scan: the computer's, so not part of a profile

# The reference laptop (i5-1334U, 10 cores, 32 GB, Intel Iris Xe, no NVIDIA card), measured on 2026-10-01: the
# benchmark's score, and each model's seconds for the 7-second sample sentence and memory once loaded.
REFERENCE_SCORE = 228.0  # GFLOPS
SECONDS = {"parakeet": 1.0, "whisper-turbo": 10.0}  # Whisper: 9-17 s while other programs used the processor
SECONDS_ON_NVIDIA = {"whisper-turbo": 1.0}  # a typical NVIDIA card with CUDA, float16
MEMORY_GB = {"parakeet": 1.0, "whisper-turbo": 3.3}
SYSTEM_GB = 2.5  # left for Windows and the other programs
FAST, USABLE = 2.0, 5.0  # seconds per sentence: fast up to 2 s, a short wait up to 5 s, slow beyond

log = logging.getLogger(__name__)


@dataclass
class Computer:
    machine: str = ""  # "x64", or "x64 on ARM64 (emulated)" on an ARM laptop (machine())
    processor: str = ""
    cores: int = 0
    threads: int = 0
    memory_gb: float = 0.0
    free_memory_gb: float = 0.0
    free_disk_gb: float = 0.0  # where speech models are downloaded
    graphics: list[str] = field(default_factory=list)
    nvidia: str = ""  # an NVIDIA card's name, if there is one
    cuda: bool = False  # Whisper can use the NVIDIA card (its CUDA libraries load)
    score: float = 0.0  # the benchmark, GFLOPS

    def summary(self) -> str:
        graphics = self.nvidia or (", ".join(self.graphics) if self.graphics else "no graphics card found")
        nvidia = " (Whisper can use it)" if self.cuda else " (no NVIDIA card)" if not self.nvidia else \
            " (NVIDIA's CUDA libraries are missing: Whisper runs on the processor)"
        machine = f" ({self.machine})" if self.machine else ""
        return (f"{self.processor or 'Unknown processor'}{machine} · {self.cores} cores · {self.memory_gb:.0f} GB memory "
                f"({self.free_memory_gb:.0f} GB free) · {self.free_disk_gb:.0f} GB free disk · {graphics}{nvidia}")


@dataclass
class Verdict:
    key: str
    level: str  # "recommended", "fast", "usable", "slow" or "no"
    seconds: float  # for a sentence of about 7 seconds
    measured: bool
    reason: str


def computer(benchmark: bool = True) -> Computer:
    total, free = _memory()
    graphics = _graphics()
    nvidia = next((g for g in graphics if "nvidia" in g.lower()), "")
    return Computer(machine=machine(), processor=_processor(), cores=_cores(), threads=os.cpu_count() or 0, memory_gb=total,
                    free_memory_gb=free, free_disk_gb=_free_disk(DOWNLOADS_DIR), graphics=graphics, nvidia=nvidia,
                    cuda=bool(nvidia) and _cuda(), score=score() if benchmark else 0.0)


def score(seconds: float = 1.0) -> float:
    """GFLOPS of float32 matrix multiplication on all cores: the kind of work both speech engines do. OpenBLAS keeps to
    one thread in Rflow (sst/__init__.py), so a thread per physical core multiplies at once (numpy lets go of the GIL
    meanwhile): as fast as OpenBLAS's own threads, and each calling thread's buffer (~30 MB, kept) is only taken here."""
    a = np.random.default_rng(0).standard_normal((768, 768), dtype=np.float32)
    workers = max(1, _cores()) if os.environ.get("OPENBLAS_NUM_THREADS") == "1" else 1
    a @ a  # warm up
    counts, t0 = [0] * workers, time.perf_counter()
    end = t0 + seconds

    def multiply(i: int) -> None:
        while True:  # at least one each, so a busy PC scores slow rather than nothing
            a @ a
            counts[i] += 1
            if time.perf_counter() >= end:
                break

    threads = [threading.Thread(target=multiply, args=(i,), daemon=True) for i in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sum(counts) * 2 * 768**3 / (time.perf_counter() - t0) / 1e9


def judge(pc: Computer, models: list, measured: dict[str, float]) -> list[Verdict]:
    """A verdict for each local model (sst.engines.SpeechModel), from what was measured or else estimated."""
    verdicts = []
    for model in models:
        key = model.key
        need = MEMORY_GB.get(key, 1.0)
        if key in measured:
            seconds, how = measured[key], True
        elif pc.cuda and key in SECONDS_ON_NVIDIA:
            seconds, how = SECONDS_ON_NVIDIA[key], False
        else:
            seconds, how = SECONDS.get(key, 5.0) * REFERENCE_SCORE / max(pc.score, 1.0), False
        size_gb = model.download.size / 1e9 if model.download else 0.0
        if pc.memory_gb and pc.memory_gb < need + SYSTEM_GB:
            verdicts.append(Verdict(key, "no", seconds, how, f"needs about {need:.1f} GB of memory; this computer has "
                                                             f"{pc.memory_gb:.0f} GB"))
            continue
        if size_gb and not model.installed() and pc.free_disk_gb < size_gb + 0.5:
            verdicts.append(Verdict(key, "no", seconds, how, f"needs {size_gb:.1f} GB of free disk space; "
                                                             f"{pc.free_disk_gb:.1f} GB are free"))
            continue
        # The reason gives the speed; the level's own words (on the page) say what it means.
        reason = f"about {seconds:.1f} s per sentence ({'measured' if how else 'estimated'})"
        level = "fast" if seconds <= FAST else "usable" if seconds <= USABLE else "slow"
        if pc.free_memory_gb and pc.free_memory_gb < need:
            reason += f"; close some programs first (it needs {need:.1f} GB of free memory)"
        verdicts.append(Verdict(key, level, seconds, how, reason))
    fast = [v for v in verdicts if v.level == "fast"]
    if fast:  # the quickest fast model; with English-only Parakeet first among equals, it's the default
        best = min(fast, key=lambda v: (v.seconds > FAST / 2, v.key != "parakeet", v.seconds))
        best.level = "recommended"
    return verdicts


def save(pc: Computer, verdicts: list[Verdict], path: Path | None = None) -> dict:
    data = {"time": time.strftime("%Y-%m-%d %H:%M"), "computer": asdict(pc), "verdicts": [asdict(v) for v in verdicts]}
    path = path or SCAN_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as e:
        log.warning("Could not save the scan: %s", e)
    return data


def load(path: Path | None = None) -> dict | None:
    try:
        data = json.loads((path or SCAN_FILE).read_text(encoding="utf-8"))
        Computer(**data["computer"])
        [Verdict(**v) for v in data["verdicts"]]
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return None


# ---- reading the computer, through Windows itself

# IMAGE_FILE_MACHINE_*: what IsWow64Process2 says the computer itself is
_MACHINES = {0x8664: "x64", 0xAA64: "ARM64", 0x014C: "x86", 0x01C4: "ARM"}
_PROGRAMS = {"win-amd64": "x64", "win-arm64": "ARM64", "win32": "x86"}  # sysconfig.get_platform()


def machine() -> str:
    """Which kind of Windows computer this is, and how Rflow runs on it: "x64" on Intel and AMD, "x64 on ARM64
    (emulated)" on an ARM laptop (Snapdragon), where Windows 11 runs the x64 program through its emulation. Rflow is
    built for x64 only: Whisper's runtime (CTranslate2) has no ARM64 build."""
    # What this program (Python, or the frozen Rflow.exe) was built for. Not platform.machine(): since Python 3.12 it
    # asks Windows for the processor, so an emulated x64 program on a Snapdragon gets "ARM64" too.
    program = _PROGRAMS.get(sysconfig.get_platform(), sysconfig.get_platform())
    native = _native_machine() or program
    return program if native == program else f"{program} on {native} (emulated)"


def _native_machine() -> str:
    """The computer's own processor kind ("ARM64" on a Snapdragon, whatever kind of program asks)."""
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.IsWow64Process2.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_ushort), ctypes.POINTER(ctypes.c_ushort)]
        process, native = ctypes.c_ushort(), ctypes.c_ushort()
        if kernel32.IsWow64Process2(kernel32.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(native)):
            return _MACHINES.get(native.value, "")
    except (OSError, AttributeError):  # IsWow64Process2 is Windows 10 1709 and later
        pass
    return ""

class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def _memory() -> tuple[float, float]:
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(_MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return 0.0, 0.0
    return status.ullTotalPhys / 2**30, status.ullAvailPhys / 2**30


def _free_disk(folder: Path) -> float:
    while not folder.exists() and folder.parent != folder:
        folder = folder.parent  # the models folder may not exist yet: its drive counts
    try:
        return shutil.disk_usage(folder).free / 1e9
    except OSError:
        return 0.0


def _processor() -> str:
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return " ".join(str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).split())
    except OSError:
        return ""


def _cores() -> int:
    """Physical cores (GetLogicalProcessorInformationEx, one record per core)."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    size = wintypes.DWORD(0)
    kernel32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))  # RelationProcessorCore = 0
    buffer = ctypes.create_string_buffer(size.value)
    if not size.value or not kernel32.GetLogicalProcessorInformationEx(0, buffer, ctypes.byref(size)):
        return os.cpu_count() or 0
    count, offset = 0, 0
    while offset < size.value:
        relationship, record = struct.unpack_from("<II", buffer.raw, offset)
        count += relationship == 0
        offset += record or size.value
    return count


_DISPLAY_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"


def _graphics() -> list[str]:
    """Display adapters, from the registry (the same list Device Manager shows), without software-only ones."""
    names = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS) as root:
            for i in range(winreg.QueryInfoKey(root)[0]):
                try:
                    with winreg.OpenKey(root, winreg.EnumKey(root, i)) as adapter:
                        name = str(winreg.QueryValueEx(adapter, "DriverDesc")[0])
                except OSError:
                    continue
                if name and "basic" not in name.lower() and "remote" not in name.lower() and name not in names:
                    names.append(name)
    except OSError:
        pass
    return names


def _cuda() -> bool:
    """Whisper can use the NVIDIA card: the driver reports a CUDA device and NVIDIA's cuBLAS library loads."""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() < 1:
            return False
    except Exception:
        return False
    for name in ("cublas64_12.dll", "cublas64_11.dll"):
        try:
            ctypes.WinDLL(name)
            return True
        except OSError:
            continue
    return False
