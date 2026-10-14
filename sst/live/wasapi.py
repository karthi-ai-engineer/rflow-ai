"""Sound in and out for live translation, through Core Audio (WASAPI, ctypes).

In: what the laptop plays, or what its microphone hears, as the live models want it: 16 kHz mono 16-bit frames of
100 ms. What the laptop plays is captured as 32-bit float and brought up to a level the model hears (sst.live.level)
before the 16-bit frames are made: turned far down, it would otherwise be lost in them.

    Capture.speakers()    everything the laptop plays except Rflow's own sound: Windows' process loopback, leaving out
                          this process, so the spoken translation isn't heard and translated again. Windows without it
                          (before Windows 11 / build 20348) get the default output's whole mix, and `hears_self` says
                          Rflow's voice is in it. Process loopback follows the default output; a call often plays on
                          another device, Windows' default output for calls (a Bluetooth headset's "Hands-Free"
                          endpoint): while something plays there, that device is captured too (endpoint loopback) and
                          mixed in. Rflow's voice plays on the default output, so it isn't in that one
    Capture.microphone()  the microphone chosen in Rflow (Settings.microphone, by name), else Windows' default one,
                          else its default for calls; opened raw where the device allows it (no echo cancellation or
                          noise suppression made for one close talker: a room's voices come through), else in its
                          default mode, never in communications mode
    -> IAudioClient -> its format (usually 48 kHz float; process loopback is asked for 16 kHz mono and converts) ->
    mono (the channels' average, or the loudest channel where they cancel out) -> 16 kHz (a box filter, then
    interpolation; state kept across packets) -> its level (quiet sound raised; the microphone's noise floor kept down)
    -> 100 ms frames

About once a second the output devices' peak meters are read (IAudioMeterInformation, no audio): when nothing has
been heard on the devices listened to for a while and another device is playing, `on_note` says where the sound
may be (and "" once it's heard again).

Out: Player, the spoken translation through Windows' default output (16-bit mono at the voice's rate: Windows converts
it). It also says whether that output is private (headphones, a headset), which a microphone can't hear.

Windows sends no loopback packets at all while nothing plays, so silence is filled in to keep the stream's clock going
(the model hears a pause, not a jump); and when nothing at all has come for a while and Windows' output is muted, the
capture says so (on PCs where Windows applies its volume before the loopback, muted means nothing reaches Rflow). When
the default device changes (headphones plugged in) or goes away, capture and player open the new one. Its own small
COM helpers, on purpose: live translation shares no code with dictation's microphone (sst.audio, sst.devices); shared
mode lets both have it open.
"""
import ctypes
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable
from ctypes import POINTER, Structure, byref, c_int, c_long, c_ulong, c_ushort, c_void_p, wintypes

import numpy as np

from sst.live.contracts import FRAME_MS, RATE
from sst.live.level import Leveler

FRAME = RATE * FRAME_MS // 1000  # 1,600 samples
SILENCE_AFTER = 0.15  # seconds without a packet: nothing is playing, so silence is added
CHECK_DEVICE_EVERY = 2.0  # seconds between looks at which device is the default
POLL_EVERY = 1.0  # seconds between looks at the output devices' meters
MUTED_AFTER = 3.0  # seconds without any sound from the computer before Windows' mute is looked at
SOUND = 1e-7  # a sample above this (-140 dBFS) is sound, however quiet: float keeps it
MUTED = "Windows' sound is muted, so live translation hears nothing. Unmute it: a low volume is fine."
_BUFFER = 2_000_000  # 200 ms, in 100 ns units: WASAPI's buffer, read every 10 ms

log = logging.getLogger(__name__)


# ---- turning the device's audio into 16 kHz mono 16-bit (pure numpy: tested without Windows)

class Resampler:
    """A stream from `rate` to 16 kHz, packet by packet, with no seams: a box filter against aliasing (its window ~ the
    rate ratio), then linear interpolation. Plenty for speech recognition."""

    def __init__(self, rate: int, target: int = RATE):
        self.ratio = rate / target
        self.width = max(1, round(self.ratio))
        self._tail = np.zeros(self.width - 1, dtype=np.float32)  # the input the filter still needs
        self._buf = np.zeros(0, dtype=np.float32)  # filtered input not yet used
        self._pos = 0.0  # where the next output sample falls in _buf

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.ratio == 1:
            return x
        if self.width > 1:
            raw = np.concatenate([self._tail, x])
            filtered = np.convolve(raw, np.full(self.width, 1 / self.width, dtype=np.float32), mode="valid")
            self._tail = raw[len(raw) - (self.width - 1):]
        else:
            filtered = x
        buf = np.concatenate([self._buf, filtered])
        last = len(buf) - 1
        n = int((last - self._pos) // self.ratio) + 1 if last >= self._pos else 0
        positions = self._pos + self.ratio * np.arange(n)
        out = np.interp(positions, np.arange(len(buf)), buf).astype(np.float32) if n else np.zeros(0, np.float32)
        following = self._pos + self.ratio * n
        drop = min(int(following), len(buf))
        self._buf, self._pos = buf[drop:], following - drop
        return out


class HighPass:
    """What's below ~`cutoff` Hz taken out: a raw microphone's rumble (a fan, the desk, mains hum), which its own
    processing used to filter; measured on the owner's laptop, 3/4 of its raw noise. Speech is above it. A linear-phase
    FIR (a windowed sinc: TAPS long, 32 ms late at 16 kHz), state kept across packets."""
    TAPS = 1023

    def __init__(self, cutoff: float = 90.0, rate: int = RATE):
        n = np.arange(self.TAPS) - (self.TAPS - 1) / 2
        low = np.sinc(2 * cutoff / rate * n) * np.blackman(self.TAPS)
        self.kernel = (-low / low.sum()).astype(np.float32)
        self.kernel[(self.TAPS - 1) // 2] += 1.0
        self._tail = np.zeros(self.TAPS - 1, dtype=np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raw = np.concatenate([self._tail, np.asarray(x, dtype=np.float32)])
        self._tail = raw[len(raw) - (self.TAPS - 1):]
        return np.convolve(raw, self.kernel, mode="valid").astype(np.float32)


class Downmix:
    """Several channels as one, without losing the sound: their average, unless the channels cancel out in it (a
    microphone array wired out of phase, or a stereo effect): then the loudest channel. Judged on the last half second
    or so, with some hysteresis, so it doesn't flip back and forth."""
    SMOOTH = 0.95  # per packet (~10 ms): about half a second
    CANCELS, MIXES = 0.25, 0.4  # the average's power against the channels': below, it cancels (correlation < -0.5)

    def __init__(self):
        self._channels: np.ndarray | None = None  # each channel's power, smoothed
        self._mixed = 0.0  # the average's power, smoothed
        self.loudest = -1  # the channel used alone; -1: the average

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """`x`: samples by channels."""
        if x.shape[1] == 1:
            return x[:, 0]
        mean = x.mean(axis=1)
        if len(x):
            power = (x.astype(np.float64) ** 2).mean(axis=0)
            mixed = float((mean.astype(np.float64) ** 2).mean())
            if self._channels is None or len(self._channels) != len(power):
                self._channels, self._mixed = power, mixed
            else:
                self._channels = self.SMOOTH * self._channels + (1 - self.SMOOTH) * power
                self._mixed = self.SMOOTH * self._mixed + (1 - self.SMOOTH) * mixed
            each = float(self._channels.mean())
            if each > 1e-12:  # digital silence says nothing either way
                ratio = self._mixed / each
                if ratio < self.CANCELS:
                    self.loudest = int(self._channels.argmax())
                elif ratio > self.MIXES:
                    self.loudest = -1
        return mean if self.loudest < 0 else np.ascontiguousarray(x[:, self.loudest])


class Mixer:
    """Two streams at one rate (the default output's and the call device's), added sample by sample as their packets
    come. Windows sends nothing for a device while nothing plays on it: a stream that sent nothing for IDLE reads
    counts as silence, so the other isn't held back. One that runs ahead by more than LAG (the devices' clocks drift)
    is let through, the other padded with silence."""
    IDLE = 15  # reads (~10 ms apart): 150 ms
    LAG = 0.2  # seconds

    def __init__(self, rate: int):
        self.lag = int(self.LAG * rate)
        self._bufs = [np.zeros(0, np.float32), np.zeros(0, np.float32)]
        self._quiet = [0, 0]

    def __call__(self, a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray:
        for i, x in enumerate((a, b)):
            if x is not None and len(x):
                self._bufs[i] = np.concatenate([self._bufs[i], np.asarray(x, np.float32)])
                self._quiet[i] = 0
            else:
                self._quiet[i] += 1
        lengths = [len(buf) for buf in self._bufs]
        waiting = [n for n, quiet in zip(lengths, self._quiet, strict=True) if quiet < self.IDLE]
        n = min(waiting) if waiting else max(lengths)
        n = max(n, max(lengths) - self.lag)
        return self._take(n)

    def flush(self) -> np.ndarray:
        """All that's left, mixed."""
        return self._take(max(len(buf) for buf in self._bufs))

    def _take(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros(0, np.float32)
        out = np.zeros(n, np.float32)
        for i, buf in enumerate(self._bufs):
            used = min(n, len(buf))
            out[:used] += buf[:used]
            self._bufs[i] = buf[used:]
        return out


def decode(data: bytes, channels: int, bits: int, is_float: bool, downmix: Downmix | None = None) -> np.ndarray:
    """A packet in the device's format as mono float in [-1, 1] (the channels' average, or as `downmix` makes it)."""
    if is_float and bits == 32:
        x = np.frombuffer(data, dtype="<f4")
    elif bits == 16:
        x = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768
    elif bits == 32:
        x = np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648
    elif bits == 24:
        b = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        x = ((b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) << 8 >> 8).astype(np.float32) / 8388608
    else:
        raise ValueError(f"unsupported audio format: {bits}-bit{' float' if is_float else ''}")
    x = x[: len(x) - len(x) % channels].reshape(-1, channels)
    if downmix is not None:
        return downmix(x)
    return x.mean(axis=1) if channels > 1 else x[:, 0]


class Framer:
    """16 kHz samples in, 100 ms PCM16 frames out (bytes)."""

    def __init__(self):
        self._pending = np.zeros(0, dtype=np.int16)

    def __call__(self, samples: np.ndarray) -> list[bytes]:
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
        self._pending = np.concatenate([self._pending, pcm])
        frames, used = [], 0
        while len(self._pending) - used >= FRAME:
            frames.append(self._pending[used:used + FRAME].tobytes())
            used += FRAME
        self._pending = self._pending[used:]
        return frames


def pcm16(samples: np.ndarray) -> bytes:
    """Float samples in [-1, 1] as 16-bit little-endian PCM."""
    return (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2").tobytes()


# ---- Core Audio through ctypes

class _GUID(Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]

    @classmethod
    def of(cls, text: str) -> "_GUID":
        return cls.from_buffer_copy(uuid.UUID(text).bytes_le)


class _WAVEFORMATEX(Structure):
    _pack_ = 1
    _fields_ = [("wFormatTag", wintypes.WORD), ("nChannels", wintypes.WORD), ("nSamplesPerSec", wintypes.DWORD),
                ("nAvgBytesPerSec", wintypes.DWORD), ("nBlockAlign", wintypes.WORD), ("wBitsPerSample", wintypes.WORD),
                ("cbSize", wintypes.WORD)]

    @classmethod
    def pcm16(cls, rate: int) -> "_WAVEFORMATEX":
        """16-bit mono at `rate`."""
        return cls(1, 1, rate, rate * 2, 2, 16, 0)

    @classmethod
    def float32(cls, rate: int) -> "_WAVEFORMATEX":
        """32-bit float mono at `rate` (WAVE_FORMAT_IEEE_FLOAT)."""
        return cls(3, 1, rate, rate * 4, 4, 32, 0)


class _WAVEFORMATEXTENSIBLE(Structure):
    _pack_ = 1
    _fields_ = [("Format", _WAVEFORMATEX), ("wValidBitsPerSample", wintypes.WORD), ("dwChannelMask", wintypes.DWORD),
                ("SubFormat", _GUID)]


class _PropVariant(Structure):
    """A PROPVARIANT (24 bytes): here a VT_BLOB (cbSize in `value`, pBlobData in `pointer`) or a VT_UI4 (`value`)."""
    _fields_ = [("vt", c_ushort), ("r1", c_ushort), ("r2", c_ushort), ("r3", c_ushort), ("value", c_ulong),
                ("pointer", c_void_p)]


class _PropertyKey(Structure):
    _fields_ = [("fmtid", _GUID), ("pid", wintypes.DWORD)]


class _LoopbackParams(Structure):
    """AUDIOCLIENT_ACTIVATION_PARAMS with its AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS."""
    _fields_ = [("ActivationType", c_int), ("TargetProcessId", wintypes.DWORD), ("ProcessLoopbackMode", c_int)]


class _ClientProperties(Structure):
    """AudioClientProperties (Windows 8.1+), for IAudioClient2::SetClientProperties."""
    _fields_ = [("cbSize", ctypes.c_uint32), ("bIsOffload", wintypes.BOOL), ("eCategory", c_int), ("Options", c_int)]


_CLSID_ENUMERATOR = _GUID.of("BCDE0395-E52F-467C-8E3D-C4579291692E")
_IID_ENUMERATOR = _GUID.of("A95664D2-9614-4F35-A746-DE8DB63617E6")
_IID_AUDIO_CLIENT = _GUID.of("1CB9AD4C-DBFA-4C32-B178-C2F568A703B2")
_IID_AUDIO_CLIENT2 = _GUID.of("726778CD-F60A-4EDA-82DE-E47610CD78AA")
_IID_METER = _GUID.of("C02216F6-8C67-4B5B-9D00-D008E73E0064")  # IAudioMeterInformation
_IID_CAPTURE_CLIENT = _GUID.of("C8ADBD64-E71E-48A0-A4DE-185C395CD317")
_IID_RENDER_CLIENT = _GUID.of("F294ACFC-3146-4483-A7BF-ADDCA7C260E2")
_IID_ENDPOINT_VOLUME = _GUID.of("5CDF2C82-841E-4546-9722-0CF74078229A")
_IID_UNKNOWN = bytes(_GUID.of("00000000-0000-0000-C000-000000000046"))
_IID_COMPLETION = bytes(_GUID.of("41D949AB-9862-444A-80F6-C261334DA5EB"))  # IActivateAudioInterfaceCompletionHandler
_IID_AGILE = bytes(_GUID.of("94EA2B94-E9CC-49E0-C0FF-EE64CA8F5B90"))  # IAgileObject: Windows calls it from any thread
_FLOAT = uuid.UUID("00000003-0000-0010-8000-00AA00389B71").bytes_le
_FORM_FACTOR = (_GUID.of("1DA5D803-D492-4EDD-8C23-E0C0FFEE7F0E"), 0)  # PKEY_AudioEndpoint_FormFactor
_FRIENDLY_NAME = (_GUID.of("A45C254E-DF1C-4EFD-8020-67D146A850E0"), 14)  # PKEY_Device_FriendlyName
_RAW_SUPPORTED = (_GUID.of("8943B373-388C-4395-B557-BC6DBAFFAFDB"), 2)  # PKEY_Devices_AudioDevice_RawProcessingSupported
_VT_BOOL, _VT_UI4, _VT_LPWSTR = 11, 19, 31
_ACTIVE = 0x1  # DEVICE_STATE_ACTIVE
_RAW = 0x1  # AUDCLNT_STREAMOPTIONS_RAW: none of the device's effects (AudioCategory_Other: no ducking of other apps)
_PRIVATE = {3, 5, 6}  # Headphones, Headset, Handset: what they play, no microphone hears
E_RENDER, E_CAPTURE = 0, 1  # EDataFlow: what's played, what's heard
E_CONSOLE, E_COMMUNICATIONS = 0, 2  # ERole: the default device, the default for calls
_CLSCTX_ALL, _SHARED, _LOOPBACK, _SILENT = 0x17, 0, 0x00020000, 0x2
_EVENT_CALLBACK = 0x00040000
_AUTOCONVERT = 0x80000000 | 0x08000000  # AUTOCONVERTPCM | SRC_DEFAULT_QUALITY: Windows takes the voice's own format
_PROCESS_LOOPBACK = "VAD\\Process_Loopback"


def _method(obj, index: int, *argtypes):
    vtable = ctypes.cast(obj, POINTER(POINTER(c_void_p)))[0]
    return lambda *args: ctypes.WINFUNCTYPE(ctypes.HRESULT, c_void_p, *argtypes)(vtable[index])(obj, *args)


def _release(obj) -> None:
    if obj:
        vtable = ctypes.cast(obj, POINTER(POINTER(c_void_p)))[0]
        ctypes.WINFUNCTYPE(wintypes.ULONG, c_void_p)(vtable[2])(obj)


def _com() -> None:
    """COM on this thread (multithreaded; a second call only counts up)."""
    ctypes.WinDLL("ole32").CoInitializeEx(None, 0x0)


def _enumerator() -> c_void_p:
    enumerator = c_void_p()
    ole32 = ctypes.WinDLL("ole32")
    ole32.CoCreateInstance.restype = ctypes.HRESULT
    ole32.CoCreateInstance(byref(_CLSID_ENUMERATOR), None, _CLSCTX_ALL, byref(_IID_ENUMERATOR), byref(enumerator))
    return enumerator


def _default_device(enumerator, flow: int, role: int) -> c_void_p:
    device = c_void_p()
    _method(enumerator, 4, c_int, c_int, POINTER(c_void_p))(flow, role, byref(device))  # GetDefaultAudioEndpoint
    return device


def _device_id(device) -> str:
    raw = ctypes.c_wchar_p()
    _method(device, 5, POINTER(ctypes.c_wchar_p))(byref(raw))  # GetId
    value = raw.value or ""
    ctypes.WinDLL("ole32").CoTaskMemFree(raw)
    return value


def _default_id(enumerator, flow: int, role: int) -> str:
    device = _default_device(enumerator, flow, role)
    try:
        return _device_id(device)
    finally:
        _release(device)


def _device_by_id(enumerator, device_id: str) -> c_void_p:
    device = c_void_p()
    _method(enumerator, 5, ctypes.c_wchar_p, POINTER(c_void_p))(device_id, byref(device))  # GetDevice
    return device


def _endpoints(enumerator, flow: int) -> list[str]:
    """The ids of the active devices of `flow`."""
    collection, count, out = c_void_p(), wintypes.UINT(), []
    _method(enumerator, 3, c_int, wintypes.DWORD, POINTER(c_void_p))(flow, _ACTIVE, byref(collection))  # EnumAudioEndpoints
    try:
        _method(collection, 3, POINTER(wintypes.UINT))(byref(count))  # GetCount
        for i in range(count.value):
            device = c_void_p()
            _method(collection, 4, wintypes.UINT, POINTER(c_void_p))(i, byref(device))  # Item
            try:
                out.append(_device_id(device))
            finally:
                _release(device)
    finally:
        _release(collection)
    return out


def _activate(device, iid: _GUID = _IID_AUDIO_CLIENT) -> c_void_p:
    client = c_void_p()
    _method(device, 3, POINTER(_GUID), wintypes.DWORD, c_void_p, POINTER(c_void_p))(
        byref(iid), _CLSCTX_ALL, None, byref(client))  # IMMDevice::Activate
    return client


def _property(device, key: tuple):
    """A device property: a number, a bool, a string, or None (not set, or another kind)."""
    store, value = c_void_p(), _PropVariant()
    try:
        _method(device, 4, wintypes.DWORD, POINTER(c_void_p))(0, byref(store))  # OpenPropertyStore(STGM_READ)
        _method(store, 5, POINTER(_PropertyKey), POINTER(_PropVariant))(byref(_PropertyKey(*key)), byref(value))  # GetValue
        if value.vt == _VT_UI4:
            return value.value
        if value.vt == _VT_BOOL:
            return bool(value.value & 0xFFFF)
        if value.vt == _VT_LPWSTR:  # its pointer is the union's first member, at offset 8
            text = c_void_p.from_buffer(value, 8).value
            return ctypes.wstring_at(text) if text else ""
        return None
    except OSError:
        return None
    finally:
        ctypes.WinDLL("ole32").PropVariantClear(byref(value))
        _release(store)


def _name(device) -> str:
    return _property(device, _FRIENDLY_NAME) or ""


def _output_muted(enumerator) -> bool:
    """Windows' default output is muted or at zero (only read, never changed)."""
    device, volume = c_void_p(), c_void_p()
    try:
        device = _default_device(enumerator, E_RENDER, E_CONSOLE)
        _method(device, 3, POINTER(_GUID), wintypes.DWORD, c_void_p, POINTER(c_void_p))(
            byref(_IID_ENDPOINT_VOLUME), _CLSCTX_ALL, None, byref(volume))  # IMMDevice::Activate
        mute, level = wintypes.BOOL(), ctypes.c_float()
        _method(volume, 15, POINTER(wintypes.BOOL))(byref(mute))  # GetMute
        _method(volume, 9, POINTER(ctypes.c_float))(byref(level))  # GetMasterVolumeLevelScalar
        return bool(mute.value) or level.value <= 0.0
    except OSError:
        return False
    finally:
        _release(volume)
        _release(device)


def _initialize(client, flags: int, fmt) -> None:
    _method(client, 3, c_int, wintypes.DWORD, ctypes.c_longlong, ctypes.c_longlong, POINTER(_WAVEFORMATEX), c_void_p)(
        _SHARED, flags, _BUFFER, 0, fmt, None)


def _private(device) -> bool:
    """The device is headphones or a headset (its form factor): a microphone can't hear what it plays."""
    return _property(device, _FORM_FACTOR) in _PRIVATE


_QI = ctypes.WINFUNCTYPE(c_long, c_void_p, POINTER(_GUID), POINTER(c_void_p))
_REF = ctypes.WINFUNCTYPE(c_ulong, c_void_p)
_DONE = ctypes.WINFUNCTYPE(c_long, c_void_p, c_void_p)


class _CompletionTable(Structure):
    _fields_ = [("QueryInterface", _QI), ("AddRef", _REF), ("Release", _REF), ("ActivateCompleted", _DONE)]


class _CompletionObject(Structure):
    _fields_ = [("table", POINTER(_CompletionTable))]


class _Completion:
    """A COM object made in Python: the IActivateAudioInterfaceCompletionHandler that Windows calls when the process
    loopback is open. Agile, so Windows may call it from its own thread; kept alive by the stream that asked."""

    def __init__(self):
        self.done, self._refs = threading.Event(), 1
        self._table = _CompletionTable(_QI(self._query), _REF(self._add), _REF(self._drop), _DONE(self._completed))
        self.object = _CompletionObject(ctypes.pointer(self._table))

    def _query(self, this, iid, out):
        if bytes(iid.contents) in (_IID_UNKNOWN, _IID_COMPLETION, _IID_AGILE):
            out[0] = this
            self._refs += 1
            return 0
        out[0] = None
        return -2147467262  # E_NOINTERFACE

    def _add(self, this):
        self._refs += 1
        return self._refs

    def _drop(self, this):
        self._refs -= 1
        return self._refs

    def _completed(self, this, operation):
        self.done.set()
        return 0


def _process_loopback(pid: int) -> tuple[c_void_p, _Completion]:
    """An IAudioClient for everything the laptop plays except process `pid` and its children."""
    params = _LoopbackParams(1, pid, 1)  # AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK, EXCLUDE_TARGET_PROCESS_TREE
    blob = _PropVariant(vt=65, value=ctypes.sizeof(params), pointer=ctypes.addressof(params))  # VT_BLOB
    handler, operation = _Completion(), c_void_p()
    activate = ctypes.WinDLL("Mmdevapi").ActivateAudioInterfaceAsync  # Windows 8+; process loopback: Windows 11
    activate.restype = ctypes.HRESULT
    activate.argtypes = [ctypes.c_wchar_p, POINTER(_GUID), c_void_p, c_void_p, POINTER(c_void_p)]
    activate(_PROCESS_LOOPBACK, byref(_IID_AUDIO_CLIENT), byref(blob), ctypes.addressof(handler.object),
             byref(operation))
    try:
        if not handler.done.wait(5):
            raise OSError("Windows didn't open the process loopback")
        result, client = c_long(), c_void_p()
        _method(operation, 3, POINTER(c_long), POINTER(c_void_p))(byref(result), byref(client))  # GetActivateResult
        if result.value:
            _release(client)
            raise OSError(f"process loopback refused ({result.value & 0xFFFFFFFF:#x})")
        return client, handler
    finally:
        _release(operation)


class _Capturing:
    """What both kinds of capture stream share: reading packets, following the default device, closing (call from one
    thread, with COM set up there)."""
    hears_self = False  # Rflow's own voice can be in what it captures
    flow, role = E_RENDER, E_CONSOLE  # the default device it follows
    rate = channels = bits = block = 0
    is_float = False

    def _prepare(self) -> None:
        self.enumerator, self.device, self.client, self.capture = (c_void_p() for _ in range(4))
        self.event, self._handler = None, None
        self.downmix = Downmix()
        self.note = ""  # for the log: which device, in which mode

    def _start(self) -> None:
        _method(self.client, 14, POINTER(_GUID), POINTER(c_void_p))(
            byref(_IID_CAPTURE_CLIENT), byref(self.capture))  # GetService
        _method(self.client, 10)()  # Start

    def default_changed(self) -> bool:
        try:
            return _default_id(self.enumerator, self.flow, self.role) != self.device_id
        except OSError:  # no such device at all for a moment
            return True

    def muted(self) -> bool:
        """Windows' default output is muted (or at zero): only what's played can be silenced by it."""
        return self.flow == E_RENDER and _output_muted(self.enumerator)

    def read(self) -> list[np.ndarray]:
        """The packets waiting now, as mono float at the device's rate (silent packets as zeros)."""
        out, size = [], wintypes.UINT()
        next_size = _method(self.capture, 5, POINTER(wintypes.UINT))
        get = _method(self.capture, 3, POINTER(c_void_p), POINTER(wintypes.UINT), POINTER(wintypes.DWORD), c_void_p,
                      c_void_p)
        release = _method(self.capture, 4, wintypes.UINT)
        next_size(byref(size))
        while size.value:
            data, frames, flags = c_void_p(), wintypes.UINT(), wintypes.DWORD()
            get(byref(data), byref(frames), byref(flags), None, None)
            try:
                if flags.value & _SILENT or not data.value:
                    out.append(np.zeros(frames.value, dtype=np.float32))
                else:
                    raw = ctypes.string_at(data.value, frames.value * self.block)
                    out.append(decode(raw, self.channels, self.bits, self.is_float, self.downmix))
            finally:
                release(frames.value)
            next_size(byref(size))
        return out

    def close(self) -> None:
        if self.client:
            try:
                _method(self.client, 11)()  # Stop
            except OSError:
                pass
        for obj in (self.capture, self.client, self.device, self.enumerator):
            _release(obj)
        self.capture = self.client = self.device = self.enumerator = c_void_p()
        if self.event:
            ctypes.windll.kernel32.CloseHandle(c_void_p(self.event))
            self.event = None


class _Stream(_Capturing):
    """One opened stream on a default device: the output device in loopback (its whole mix, Rflow's voice too), or a
    microphone."""

    def __init__(self, flow: int = E_RENDER, role: int = E_CONSOLE):
        self.flow, self.role, self.hears_self = flow, role, flow == E_RENDER
        self._prepare()
        try:
            self.enumerator = _enumerator()
            self.device = _default_device(self.enumerator, flow, role)
            self.device_id = _device_id(self.device)
            self._open()
            self._start()
        except Exception:
            self.close()
            raise

    def _open(self, properties: _ClientProperties | None = None) -> None:
        """An audio client on self.device in its mix format; `properties` (raw) are set first, through IAudioClient2."""
        self.client = _activate(self.device, _IID_AUDIO_CLIENT2 if properties else _IID_AUDIO_CLIENT)
        if properties is not None:
            _method(self.client, 16, POINTER(_ClientProperties))(byref(properties))  # SetClientProperties
        mix = POINTER(_WAVEFORMATEX)()
        _method(self.client, 8, POINTER(POINTER(_WAVEFORMATEX)))(byref(mix))  # GetMixFormat
        try:
            f = mix.contents
            self.rate, self.channels, self.bits, self.block = f.nSamplesPerSec, f.nChannels, f.wBitsPerSample, f.nBlockAlign
            if f.wFormatTag == 0xFFFE:  # WAVEFORMATEXTENSIBLE: the real format is its SubFormat
                self.is_float = bytes(ctypes.cast(mix, POINTER(_WAVEFORMATEXTENSIBLE)).contents.SubFormat) == _FLOAT
            else:
                self.is_float = f.wFormatTag == 3
            _initialize(self.client, _LOOPBACK if self.flow == E_RENDER else 0, mix)
        finally:
            ctypes.WinDLL("ole32").CoTaskMemFree(mix)


def choose_microphone(available: list[tuple[str, str]], wanted: str, console: str, communications: str) -> tuple[str, str]:
    """The microphone to listen to, as (its id, why): `wanted` (a name chosen in Rflow; a name cut short by an older
    Rflow matches the start of the full one), else Windows' default, else its default for calls. `available`: the
    active microphones as (id, name)."""
    if wanted:
        found = next((i for i, name in available if name == wanted), None) or next(
            (i for i, name in available if name.startswith(wanted)), None)
        if found:
            return found, "chosen in Rflow"
    missing = f" ({wanted} isn't connected)" if wanted else ""
    if console:
        return console, "Windows' default" + missing
    if communications:
        return communications, "Windows' default for calls" + missing
    raise OSError("no microphone is connected")


class _Microphone(_Stream):
    """The microphone to listen to (choose_microphone; `wanted` is read at each look, so a new choice is followed),
    opened raw where the device allows it: without the echo cancellation, noise suppression and gain made for one
    person close to it, which hear a room's voices weakly. Elsewhere in its default mode (never communications)."""

    def __init__(self, wanted: str | Callable[[], str] = ""):
        self.flow, self.role, self.hears_self = E_CAPTURE, E_CONSOLE, False
        self._wanted = wanted if callable(wanted) else (lambda: wanted)
        self._prepare()
        try:
            self.enumerator = _enumerator()
            self.device_id, why = self._choose()
            self.device = _device_by_id(self.enumerator, self.device_id)
            mode = "default mode"
            if _property(self.device, _RAW_SUPPORTED) is True:
                try:
                    self._open(_ClientProperties(ctypes.sizeof(_ClientProperties), 0, 0, _RAW))  # AudioCategory_Other
                    mode = "raw"
                except OSError as e:
                    log.info("Live translation: the microphone refused raw sound (%s): its default mode instead", e)
                    _release(self.client)
                    self.client = c_void_p()
            else:
                mode += ", raw not supported"
            if not self.client:
                self._open()
            self.note = f"{_name(self.device) or 'unnamed'}, {why}, {mode}"
            self._start()
        except Exception:
            self.close()
            raise

    def _choose(self) -> tuple[str, str]:
        available = []
        for device_id in _endpoints(self.enumerator, E_CAPTURE):
            device = _device_by_id(self.enumerator, device_id)
            try:
                available.append((device_id, _name(device)))
            finally:
                _release(device)
        defaults = []
        for role in (E_CONSOLE, E_COMMUNICATIONS):
            try:
                defaults.append(_default_id(self.enumerator, E_CAPTURE, role))
            except OSError:  # E_NOTFOUND: none
                defaults.append("")
        return choose_microphone(available, self._wanted() or "", *defaults)

    def default_changed(self) -> bool:
        """Another microphone would be chosen now: the chosen one plugged in or out, or Windows' default changed."""
        try:
            return self._choose()[0] != self.device_id
        except OSError:
            return True


class _ProcessLoopback(_Capturing):
    """Everything the laptop plays except this process's own sound (the spoken translation), already as 16 kHz mono
    32-bit float (16-bit where Windows won't): Windows converts. Measured on the owner's laptop: another program's tone
    came through, Rflow's didn't; float was accepted."""

    def __init__(self, rate: int = RATE):
        self.rate, self.channels = rate, 1
        self._prepare()
        try:
            self.enumerator = _enumerator()
            self.device_id = _default_id(self.enumerator, E_RENDER, E_CONSOLE)  # followed like the other streams
            kernel32 = ctypes.windll.kernel32
            kernel32.CreateEventW.restype = c_void_p
            self.event = kernel32.CreateEventW(None, False, False, None)
            for fmt in (_WAVEFORMATEX.float32(rate), _WAVEFORMATEX.pcm16(rate)):
                self.client, self._handler = _process_loopback(os.getpid())
                try:
                    _initialize(self.client, _LOOPBACK | _EVENT_CALLBACK, byref(fmt))
                    break
                except OSError:
                    if fmt.wFormatTag != 3:
                        raise
                    _release(self.client)  # float refused: a new client for 16-bit (a client initializes once)
                    self.client = c_void_p()
            self.bits, self.block, self.is_float = fmt.wBitsPerSample, fmt.nBlockAlign, fmt.wFormatTag == 3
            _method(self.client, 13, c_void_p)(c_void_p(self.event))  # SetEventHandle: it runs event-driven; we poll
            self._start()
        except Exception:
            self.close()
            raise


def _muted(stream) -> bool:
    muted = getattr(stream, "muted", None)  # a stream that can't tell (a fake) isn't muted
    return bool(muted is not None and muted())


def _speakers():
    """What the laptop plays without Rflow's own sound where Windows can leave it out (checked against the whole mix,
    _Checked), else the whole mix."""
    try:
        main = _ProcessLoopback()
    except Exception as e:  # an older Windows: no process loopback
        log.info("Live translation: process loopback isn't available (%s); capturing the whole output", e)
        return _Stream(E_RENDER, E_CONSOLE)
    try:
        return _Checked(main, _Stream(E_RENDER, E_CONSOLE))
    except Exception as e:  # the check is a help, not a need
        log.info("Live translation: the whole output couldn't be opened to check process loopback (%s)", e)
        return main


WHOLE_AFTER = 1.5  # seconds of sound in the whole mix, with none at all through process loopback, before using the mix
WHOLE_SOUND = 1e-4  # a peak above this (-80 dBFS) in the whole mix is something playing
VOICE_MARGIN = 0.5  # seconds after Rflow's own voice was last queued during which the whole mix may still carry it
_voice_until = 0.0  # time.monotonic() until which Rflow's own voice may be playing (Player.play)


def _loud(packets: list[np.ndarray], floor: float) -> bool:
    return any(len(p) and float(np.max(np.abs(p))) > floor for p in packets)


class _Checked:
    """Process loopback, checked against the default output's whole mix until it has heard something. On the owner's
    laptop (2026-10-13/14, #176: a Teams call on Bluetooth earbuds, Teams playing to "Headphones", the laptop's mic)
    process loopback, even of Teams' process alone, got only silence while the whole mix carried the call, and the
    output meters read 0 as well. So both are read: once process loopback hears a sound the mix is closed; if instead the
    mix has had WHOLE_AFTER seconds of sound (not Rflow's own voice) while process loopback had none, the mix is used for
    the rest of this stream. Rflow's voice is in it then (`hears_self`), so the session silences it meanwhile."""

    def __init__(self, main, whole, clock: Callable[[], float] = time.monotonic):
        self.main, self.whole, self._clock = main, whole, clock
        self.rate, self.channels, self.bits, self.is_float = main.rate, main.channels, main.bits, main.is_float
        self.device_id, self.note = getattr(main, "device_id", ""), getattr(main, "note", "")
        self._resample = Resampler(whole.rate, main.rate)
        self._whole_sound = 0.0
        self.switched = False

    @property
    def hears_self(self) -> bool:
        return self.switched or getattr(self.main, "hears_self", False)

    def read(self) -> list[np.ndarray]:
        if self.switched:
            packets = self.whole.read()
            return [self._resample(np.concatenate(packets))] if packets else []
        packets = self.main.read()
        if self.whole is None:
            return packets
        extra = self.whole.read()  # read even when not needed: its buffer mustn't overflow
        if _loud(packets, SOUND):
            log.info("Live translation: process loopback hears what plays; the whole mix isn't needed")
            self._close(self.whole)
            self.whole = None
            return packets
        if extra and _loud(extra, WHOLE_SOUND) and self._clock() >= _voice_until:
            self._whole_sound += sum(len(p) for p in extra) / self.whole.rate
            if self._whole_sound >= WHOLE_AFTER:
                log.info("Live translation: process loopback hears nothing while the output plays (a call on Bluetooth "
                         "earbuds?): it hears the output's whole mix instead, Rflow's own voice too")
                self.switched, self.note = True, "the whole mix: process loopback heard nothing"
                self._close(self.main)
                return [self._resample(np.concatenate(extra))]
        return packets

    def _active(self):
        return self.whole if self.switched else self.main

    def default_changed(self) -> bool:
        return self._active().default_changed()

    def muted(self) -> bool:
        return _muted(self._active())

    @staticmethod
    def _close(stream) -> None:
        try:
            stream.close()
        except Exception as e:
            log.debug("Live translation: closing a capture failed: %s", e)

    def close(self) -> None:
        if not self.switched:
            self._close(self.main)
        if self.whole is not None:
            self._close(self.whole)
            self.whole = None


class _Meters:
    """The output devices' peak meters (IAudioMeterInformation: no audio, only how loud each plays now), and which
    devices are Windows' default output and its default for calls. Each meter opened once, kept while its device is."""

    def __init__(self):
        self.enumerator = _enumerator()
        self._meters: dict[str, c_void_p] = {}

    def read(self) -> tuple[str, str, dict[str, float]]:
        """(the default output's id, the default for calls' id, {active output's id: its peak now, 0..1})."""
        defaults = []
        for role in (E_CONSOLE, E_COMMUNICATIONS):
            try:
                defaults.append(_default_id(self.enumerator, E_RENDER, role))
            except OSError:
                defaults.append("")
        ids, peaks = _endpoints(self.enumerator, E_RENDER), {}
        for gone in set(self._meters) - set(ids):
            _release(self._meters.pop(gone))
        for device_id in ids:
            try:
                meter = self._meters.get(device_id)
                if meter is None:
                    device, meter = _device_by_id(self.enumerator, device_id), c_void_p()
                    try:
                        _method(device, 3, POINTER(_GUID), wintypes.DWORD, c_void_p, POINTER(c_void_p))(
                            byref(_IID_METER), _CLSCTX_ALL, None, byref(meter))  # IMMDevice::Activate
                    finally:
                        _release(device)
                    self._meters[device_id] = meter
                peak = ctypes.c_float()
                _method(meter, 3, POINTER(ctypes.c_float))(byref(peak))  # GetPeakValue
                peaks[device_id] = peak.value
            except OSError:  # going away just now
                _release(self._meters.pop(device_id, c_void_p()))
        return defaults[0], defaults[1], peaks

    def name(self, device_id: str) -> str:
        try:
            device = _device_by_id(self.enumerator, device_id)
        except OSError:
            return "another device"
        try:
            return _name(device) or "another device"
        finally:
            _release(device)

    def close(self) -> None:
        for meter in self._meters.values():
            _release(meter)
        self._meters.clear()
        _release(self.enumerator)
        self.enumerator = c_void_p()


PLAYING = 1e-3  # a peak above this (-60 dBFS) is a device playing something
CALL_LINGER = 30.0  # seconds the call device stays captured after it last played (a pause in the call)
ELSEWHERE_AFTER = 5.0  # seconds of nothing heard, while another device plays, before saying where the sound may be
ELSEWHERE = "Nothing plays on {heard}: the sound may be on {other}. Make it Windows' default output to translate it."


class _Speakers:
    """What the laptop plays (`main`: process loopback, or the default output's whole mix), and also, while something
    plays there, Windows' default output for calls when that's another device: a call on a Bluetooth headset's
    "Hands-Free" endpoint, which process loopback (following the default output) doesn't hear. That one is opened only
    while it plays and closed after CALL_LINGER of silence, so a headset isn't held in call mode by Rflow. Rflow's own
    voice plays on the default output (Player), never there: `hears_self` is main's. poll(now), about once a second,
    looks at the meters and says (a message, "" when it's over, None for no change) when the sound may be elsewhere."""

    def __init__(self, main, meters: Callable | None = None, open_call: Callable | None = None):
        self.main = main
        self._open_call = open_call or (lambda: _Stream(E_RENDER, E_COMMUNICATIONS))
        self.rate, self.channels, self.bits, self.is_float = main.rate, main.channels, main.bits, main.is_float
        self.device_id = getattr(main, "device_id", "")
        self.note = getattr(main, "note", "")
        try:
            self.meters = (meters or _Meters)()
        except Exception as e:  # the meters are a help, not a need
            log.info("Live translation: no output meters (%s)", e)
            self.meters = None
        self.call = None  # the call device's loopback, while open
        self._mix: Mixer | None = None
        self._resample: Resampler | None = None
        self._call_heard = -1e9  # when the call device last played
        self._heard: float | None = None  # when the devices listened to last played
        self._other_since: dict[str, float] = {}  # other devices playing, since when
        self._told = ""

    @property
    def hears_self(self) -> bool:  # main's, which can change (_Checked turning to the whole mix)
        return getattr(self.main, "hears_self", False)

    def read(self) -> list[np.ndarray]:
        packets = self.main.read()  # its failure (unplugged) reopens everything
        if self.call is None:
            if self._mix is None:
                return packets
            rest, self._mix = self._mix.flush(), None  # the call device just closed: what's left of the mix first
            return [rest, *packets]
        try:
            extra = self.call.read()
        except OSError as e:
            log.info("Live translation: the call device went away (%s)", e)
            self._close_call()
            extra = []
        a = np.concatenate(packets) if packets else None
        b = self._resample(np.concatenate(extra)) if extra else None
        mixed = self._mix(a, b) if self._mix is not None else (a if a is not None else np.zeros(0, np.float32))
        return [mixed] if len(mixed) else []

    def poll(self, now: float) -> str | None:
        if self.meters is None:
            return None
        try:
            console, calls, peaks = self.meters.read()
        except OSError as e:
            log.debug("Live translation: the output meters couldn't be read: %s", e)
            return None
        if self._heard is None:
            self._heard = now
        separate = bool(calls) and calls != console
        if separate and peaks.get(calls, 0.0) > PLAYING:
            self._call_heard = now
        want = separate and now - self._call_heard < CALL_LINGER
        if self.call is not None and (not want or self.call.device_id != calls):
            log.info("Live translation: no longer captures the device for calls")
            self._close_call()
        if want and self.call is None:
            self._open_call_device()
        listened = {console} | ({calls} if self.call is not None else set())
        if any(peaks.get(d, 0.0) > PLAYING for d in listened):
            self._heard = now
        self._other_since = {d: self._other_since.get(d, now) for d, peak in peaks.items()
                             if d not in listened and peak > PLAYING}
        if now - self._heard < ELSEWHERE_AFTER:
            message = ""
        else:
            others = [d for d, since in self._other_since.items() if now - since >= ELSEWHERE_AFTER - 1.0]
            message = ELSEWHERE.format(heard=self.meters.name(console) if console else "the default output",
                                       other=self.meters.name(others[0])) if others else ""
        if message == self._told:
            return None
        self._told = message
        if message:
            log.info("Live translation: %s", message)
        return message

    def _open_call_device(self) -> None:
        try:
            self.call = self._open_call()
        except Exception as e:
            log.info("Live translation: couldn't capture the device for calls: %s", e)
            return
        self._mix, self._resample = Mixer(self.rate), Resampler(self.call.rate, self.rate)
        log.info("Live translation also hears the device for calls (%d Hz, %d channels)", self.call.rate,
                 self.call.channels)

    def _close_call(self) -> None:
        if self.call is not None:
            self.call.close()
            self.call = None

    def default_changed(self) -> bool:
        return self.main.default_changed()

    def muted(self) -> bool:
        return _muted(self.main)

    def close(self) -> None:
        self._close_call()
        self.main.close()
        if self.meters is not None:
            self.meters.close()
            self.meters = None


class Capture:
    """start(on_frame) calls on_frame(bytes) with each 100 ms frame, from its own thread: of what the laptop plays
    (speakers()) or of what its microphone hears (microphone()). `hears_self`: Rflow's own voice can be in it. `level`:
    quiet sound is raised (True: as the computer's; or a function making a Leveler). `on_problem(message)`, if set,
    hears why nothing can be captured (muted). `on_note(message)`, if set, hears a hint about where the sound may be
    instead (another output device playing while nothing is heard), and "" once it no longer applies; called from the
    capture's thread."""

    def __init__(self, opener: Callable = _Stream, clock: Callable[[], float] = time.monotonic,
                 what: str = "output device", level: bool | Callable[[], Leveler] = False, highpass: bool = False):
        self._opener, self._clock, self.what, self.level, self.highpass = opener, clock, what, level, highpass
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.device_rate = 0  # the device's own rate, once opened (for the log)
        self.hears_self = False
        self.on_problem: Callable[[str], None] | None = None
        self.on_note: Callable[[str], None] | None = None
        self._notice = ""

    @classmethod
    def speakers(cls) -> "Capture":
        return cls(lambda: _Speakers(_speakers()), what="output device", level=True)

    @classmethod
    def microphone(cls, wanted: str | Callable[[], str] = "") -> "Capture":
        """The microphone chosen in Rflow (`wanted`: its name, or a function giving it now), else Windows' default."""
        return cls(lambda: _Microphone(wanted), what="microphone", level=Leveler.for_microphone, highpass=True)

    def start(self, on_frame: Callable[[bytes], None]) -> None:
        self._stop.clear()
        opened = threading.Event()
        problem: list[Exception] = []
        self._thread = threading.Thread(target=self._run, args=(on_frame, opened, problem),
                                        name=f"live-{self.what.split()[0]}", daemon=True)
        self._thread.start()
        opened.wait(5)
        if problem:  # the first open failed: say so to the caller rather than capturing nothing quietly
            self.stop()
            raise problem[0]

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(3)

    def _run(self, on_frame, opened: threading.Event, problem: list) -> None:
        _com()
        first = True
        try:
            while not self._stop.is_set():
                try:
                    stream = self._opener()
                except Exception as e:
                    if first:
                        problem.append(e)
                        opened.set()
                        return
                    log.warning("Live translation: couldn't open the %s: %s", self.what, e)
                    self._stop.wait(1.0)
                    continue
                self.device_rate, self.hears_self = stream.rate, getattr(stream, "hears_self", False)
                if first:
                    first = False
                    opened.set()
                note = getattr(stream, "note", "")
                log.info("Live translation hears the %s (%d Hz, %d channels, %d-bit%s%s%s)", self.what, stream.rate,
                         stream.channels, stream.bits, " float" if stream.is_float else "",
                         "" if self.hears_self or self.what != "output device" else ", without Rflow's own sound",
                         f"; {note}" if note else "")
                try:
                    self._pump(stream, on_frame)
                except OSError as e:  # unplugged, or the format changed: open whatever is the default now
                    log.info("Live translation: the %s changed (%s)", self.what, e)
                finally:
                    stream.close()
                    self._tell("")  # a hint about the old device no longer applies
        finally:
            ctypes.WinDLL("ole32").CoUninitialize()

    def _tell(self, notice: str) -> None:
        if notice != self._notice:
            self._notice = notice
            if self.on_note is not None:
                self.on_note(notice)

    def _pump(self, stream, on_frame) -> None:
        resample, frame = Resampler(stream.rate), Framer()
        level = None if not self.level else Leveler() if self.level is True else self.level()
        rumble = HighPass() if self.highpass else None
        poll = getattr(stream, "poll", None)
        last_packet = last_fill = last_check = last_poll = sound_at = self._clock()
        told = False  # that Windows' output is muted, until sound comes again
        while not self._stop.is_set():
            now = self._clock()
            packets = stream.read()
            self.hears_self = getattr(stream, "hears_self", False)  # it may turn to the whole mix
            if packets:
                last_packet = last_fill = now
                samples = resample(np.concatenate(packets))
                if rumble is not None:
                    samples = rumble(samples)
                if level is not None:
                    if len(samples) and float(np.max(np.abs(samples))) > SOUND:
                        sound_at, told = now, False
                    samples = level(samples)
                for pcm in frame(samples):
                    on_frame(pcm)
            elif now - last_packet >= SILENCE_AFTER and now - last_fill >= FRAME_MS / 1000:
                last_fill = now  # nothing plays: Windows sends nothing, so silence keeps the clock going
                on_frame(bytes(FRAME * 2))
            if now - last_check >= CHECK_DEVICE_EVERY:
                last_check = now
                if stream.default_changed():
                    log.info("Live translation: a new default %s", self.what)
                    return
                if level is not None and not told and now - sound_at >= MUTED_AFTER and _muted(stream):
                    told = True
                    log.info("Live translation: nothing heard, and Windows' output is muted")
                    if self.on_problem is not None:
                        self.on_problem(MUTED)
            if poll is not None and now - last_poll >= POLL_EVERY:
                last_poll = now
                notice = poll(now)
                if notice is not None:
                    self._tell(notice)
            self._stop.wait(0.01)


# ---- the spoken translation out

class _Render:
    """One opened output stream on the default output device: 16-bit mono at `rate` (Windows converts it). Its own
    thread's COM; `private`: the device is headphones or a headset."""

    def __init__(self, rate: int):
        _com()
        self.enumerator, self.device, self.client, self.render = (c_void_p() for _ in range(4))
        try:
            self.enumerator = _enumerator()
            self.device = _default_device(self.enumerator, E_RENDER, E_CONSOLE)
            self.device_id = _device_id(self.device)
            self.private = _private(self.device)
            self.client = _activate(self.device)
            _initialize(self.client, _AUTOCONVERT, byref(_WAVEFORMATEX.pcm16(rate)))
            size = wintypes.UINT()
            _method(self.client, 4, POINTER(wintypes.UINT))(byref(size))  # GetBufferSize
            self.size = size.value
            _method(self.client, 14, POINTER(_GUID), POINTER(c_void_p))(byref(_IID_RENDER_CLIENT), byref(self.render))
            _method(self.client, 10)()  # Start: silence until samples come
        except Exception:
            self.close()
            raise

    def queued(self) -> int:
        """Frames written and not yet played."""
        padding = wintypes.UINT()
        _method(self.client, 6, POINTER(wintypes.UINT))(byref(padding))  # GetCurrentPadding
        return padding.value

    def write(self, pcm: bytes) -> int:
        """As many of the 16-bit samples as fit in the buffer now; how many went in."""
        n = min(self.size - self.queued(), len(pcm) // 2)
        if n <= 0:
            return 0
        data = c_void_p()
        _method(self.render, 3, wintypes.UINT, POINTER(c_void_p))(n, byref(data))  # GetBuffer
        ctypes.memmove(data.value, pcm, n * 2)
        _method(self.render, 4, wintypes.UINT, wintypes.DWORD)(n, 0)  # ReleaseBuffer
        return n

    def flush(self) -> None:
        """Silence at once: what was queued isn't played."""
        for index in (11, 12, 10):  # Stop, Reset, Start
            _method(self.client, index)()

    def default_changed(self) -> bool:
        try:
            return _default_id(self.enumerator, E_RENDER, E_CONSOLE) != self.device_id
        except OSError:
            return True

    def close(self) -> None:
        if self.client:
            try:
                _method(self.client, 11)()  # Stop
            except OSError:
                pass
        for obj in (self.render, self.client, self.device, self.enumerator):
            _release(obj)
        self.render = self.client = self.device = self.enumerator = c_void_p()


class Player:
    """The spoken translation through Windows' default output, following it when it changes. play() is called from one
    thread (the voice's: COM lives there); interrupt() from any. `private`: the output is headphones or a headset."""

    def __init__(self, opener: Callable = _Render, clock: Callable[[], float] = time.monotonic):
        self._opener, self._clock = opener, clock
        self._stream = None
        self._rate, self._checked = 0, 0.0
        self._interrupt = threading.Event()
        self.private = False

    def play(self, samples: np.ndarray, rate: int) -> float:
        """Queue `samples` (float, -1..1) for playing. Returns once they're all in Windows' buffer, with the seconds
        still to be heard then (up to its 0.2 s), or 0 when interrupted (the rest isn't played)."""
        global _voice_until
        self._interrupt.clear()
        _voice_until = max(_voice_until, time.monotonic() + len(samples) / rate + VOICE_MARGIN)
        pcm, failures = pcm16(samples), 0
        while pcm and not self._interrupt.is_set():
            try:
                stream = self._open(rate)
                n = stream.write(pcm)
                failures = 0
            except Exception as e:  # unplugged, or the device changed under it: the default again
                failures += 1
                log.info("The voice's output went away (%s): opening the default again", e)
                self._close()
                if failures >= 3:
                    raise
                time.sleep(0.1)
                continue
            pcm = pcm[n * 2:]
            if not n:
                time.sleep(0.01)
        if self._stream is None:
            return 0.0
        if self._interrupt.is_set():
            self._stream.flush()
            return 0.0
        return self._stream.queued() / rate

    def interrupt(self) -> None:
        self._interrupt.set()

    def close(self) -> None:
        self._close()

    def _open(self, rate: int):
        now = self._clock()
        if self._stream is not None and (rate != self._rate or (
                now - self._checked >= CHECK_DEVICE_EVERY and self._stream.default_changed())):
            log.info("The voice follows the new default output")
            self._close()
        if now - self._checked >= CHECK_DEVICE_EVERY:
            self._checked = now
        if self._stream is None:
            self._stream, self._rate = self._opener(rate), rate
            self.private = self._stream.private
            log.info("The voice plays through the default output (%s)", "headphones" if self.private else "speakers")
        return self._stream

    def _close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
