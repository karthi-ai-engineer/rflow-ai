"""Windows' own OCR (Windows.Media.Ocr), called through ctypes: on this laptop, free, and no package to ship.

Windows OCR is on-device, 20-150 ms a frame on a laptop CPU, and gives each word's box; the owner's screen translator
measured it against PP-OCRv6/v5 on a Japanese UI test image: 3.8% character errors against 6.9%, at 76-132 ms against
~3 s. Japanese needs Windows' Japanese OCR component (Settings > Time & language > Language > Japanese, or
`Add-WindowsCapability -Online -Name Language.OCR~~~ja-JP~0.0.1.0`); without it `WinOcr` raises OcrMissing.

The WinRT calls, by vtable slot (IUnknown 0-2, IInspectable 3-5, then each interface's own methods in order):
  Windows.Globalization.Language       ILanguageFactory.CreateLanguage(tag)                      6
  Windows.Media.Ocr.OcrEngine          IOcrEngineStatics.MaxImageDimension 6, TryCreateFromLanguage 9
  Windows.Security.Cryptography.CryptographicBuffer  ICryptographicBufferStatics.CreateFromByteArray  9
  Windows.Graphics.Imaging.SoftwareBitmap  ISoftwareBitmapStatics.CreateCopyFromBuffer(buffer, Gray8, w, h)  9
  IOcrEngine.RecognizeAsync 6 -> IAsyncOperation<OcrResult> (IAsyncInfo.Status 7, polled; GetResults 8)
  IOcrResult.Lines 6, IOcrLine.Words 6, IOcrWord.BoundingRect 6 / Text 7, IVectorView.GetAt 6 / Size 7
One engine per thread that made it (WinRT objects here live in that thread's apartment): ScreenLens keeps it on its
OCR thread.
"""
import ctypes
import time
from ctypes import POINTER, byref, c_float, c_int, c_uint32, c_void_p, wintypes
from dataclasses import dataclass

GRAY8 = 62  # BitmapPixelFormat.Gray8
RO_INIT_MULTITHREADED = 1
S_FALSE, RPC_E_CHANGED_MODE = 1, -2147417850  # RoInitialize's "already" answers: the thread is usable as it is
STARTED, COMPLETED = 0, 1  # AsyncStatus
TIMEOUT_S = 10.0


class OcrMissing(RuntimeError):
    """Windows OCR, or its language, isn't on this computer."""


@dataclass
class Word:
    text: str
    x: float
    y: float
    width: float
    height: float


class _GUID(ctypes.Structure):
    _fields_ = [("a", ctypes.c_ulong), ("b", ctypes.c_ushort), ("c", ctypes.c_ushort), ("d", ctypes.c_ubyte * 8)]

    @classmethod
    def of(cls, text: str) -> "_GUID":
        h = text.strip("{}").replace("-", "")
        return cls(int(h[:8], 16), int(h[8:12], 16), int(h[12:16], 16), (ctypes.c_ubyte * 8)(*bytes.fromhex(h[16:])))


class _Rect(ctypes.Structure):
    _fields_ = [("x", c_float), ("y", c_float), ("width", c_float), ("height", c_float)]


IID_LANGUAGE_FACTORY = "{9B0252AC-0C27-44F8-B792-9793FB66C63E}"
IID_OCR_STATICS = "{5BFFA85A-3384-3540-9940-699120D428A8}"
IID_CRYPTO_BUFFER_STATICS = "{320B7E22-3CB0-4CDF-8663-1D28910065EB}"
IID_BITMAP_STATICS = "{DF0385DB-672F-4A9D-806E-C2442F343E86}"
IID_ASYNC_INFO = "{00000036-0000-0000-C000-000000000046}"

_combase = None


def _api():
    global _combase
    if _combase is None:
        lib = ctypes.WinDLL("combase")
        lib.RoInitialize.argtypes = [c_int]
        lib.RoInitialize.restype = ctypes.c_long  # read, not raised: S_FALSE and RPC_E_CHANGED_MODE are fine
        lib.WindowsCreateString.argtypes = [wintypes.LPCWSTR, c_uint32, POINTER(c_void_p)]
        lib.WindowsCreateString.restype = ctypes.c_long
        lib.WindowsDeleteString.argtypes = [c_void_p]
        lib.WindowsGetStringRawBuffer.argtypes = [c_void_p, POINTER(c_uint32)]
        lib.WindowsGetStringRawBuffer.restype = ctypes.c_void_p
        lib.RoGetActivationFactory.argtypes = [c_void_p, POINTER(_GUID), POINTER(c_void_p)]
        lib.RoGetActivationFactory.restype = ctypes.c_long
        _combase = lib
    return _combase


def _check(hr: int, what: str) -> None:
    if hr < 0:
        raise OSError(f"{what} failed (HRESULT 0x{hr & 0xFFFFFFFF:08X})")


def _call(obj: int, slot: int, *args, argtypes=()):
    """Method `slot` of a COM/WinRT object's vtable, called with `args` (their ctypes types in argtypes)."""
    vtable = ctypes.cast(ctypes.cast(obj, POINTER(c_void_p))[0], POINTER(c_void_p))
    method = ctypes.WINFUNCTYPE(ctypes.c_long, c_void_p, *argtypes)(vtable[slot])
    return method(obj, *args)


def _release(obj) -> None:
    if obj:
        _call(obj, 2)


def _out(obj: int, slot: int, what: str, *args, argtypes=()) -> int:
    """A method whose last argument is an interface pointer it returns."""
    result = c_void_p()
    _check(_call(obj, slot, *args, byref(result), argtypes=(*argtypes, POINTER(c_void_p))), what)
    return result.value or 0


class _HString:
    def __init__(self, text: str):
        self.handle = c_void_p()
        _check(_api().WindowsCreateString(text, len(text), byref(self.handle)), "WindowsCreateString")

    def __enter__(self):
        return self.handle

    def __exit__(self, *_):
        _api().WindowsDeleteString(self.handle)


def _text(handle: c_void_p) -> str:
    if not handle.value:
        return ""
    size = c_uint32()
    raw = _api().WindowsGetStringRawBuffer(handle, byref(size))
    text = ctypes.wstring_at(raw, size.value) if raw else ""
    _api().WindowsDeleteString(handle)
    return text


def _factory(class_name: str, iid: str) -> int:
    result = c_void_p()
    with _HString(class_name) as name:
        _check(_api().RoGetActivationFactory(name, byref(_GUID.of(iid)), byref(result)), f"{class_name}'s factory")
    return result.value or 0


def _items(vector: int) -> list[int]:
    """An IVectorView's items (each to be released by the caller), and the view released."""
    size = c_uint32()
    _check(_call(vector, 7, byref(size), argtypes=(POINTER(c_uint32),)), "IVectorView.Size")
    items = [_out(vector, 6, "IVectorView.GetAt", c_uint32(i), argtypes=(c_uint32,)) for i in range(size.value)]
    _release(vector)
    return items


class WinOcr:
    """recognize(gray bytes, w, h) -> lines of Words, in the image's pixels. Use from one thread only."""

    def __init__(self, language: str = "ja"):
        hr = _api().RoInitialize(RO_INIT_MULTITHREADED)
        if hr < 0 and hr != RPC_E_CHANGED_MODE:
            _check(hr, "RoInitialize")
        try:
            statics = _factory("Windows.Media.Ocr.OcrEngine", IID_OCR_STATICS)
        except OSError as e:
            raise OcrMissing(f"Windows OCR isn't available on this computer ({e}).") from None
        factory = _factory("Windows.Globalization.Language", IID_LANGUAGE_FACTORY)
        try:
            with _HString(language) as tag:
                lang = _out(factory, 6, "Language", tag, argtypes=(c_void_p,))
            dim = c_uint32()
            _check(_call(statics, 6, byref(dim), argtypes=(POINTER(c_uint32),)), "OcrEngine.MaxImageDimension")
            self.max_dim = int(dim.value) or 10000
            self._engine = _out(statics, 9, "OcrEngine.TryCreateFromLanguage", lang, argtypes=(c_void_p,))
            _release(lang)
        finally:
            _release(factory)
            _release(statics)
        if not self._engine:
            raise OcrMissing(f"Windows' OCR for this language ({language}) isn't installed: Settings > Time & language > "
                             "Language & region > add Japanese (its optical character recognition).")
        self._crypto = _factory("Windows.Security.Cryptography.CryptographicBuffer", IID_CRYPTO_BUFFER_STATICS)
        self._bitmaps = _factory("Windows.Graphics.Imaging.SoftwareBitmap", IID_BITMAP_STATICS)

    def recognize(self, gray: bytes, width: int, height: int) -> list[list[Word]]:
        """The lines Windows reads in an 8-bit grey image (row after row, `width` bytes each)."""
        data = (ctypes.c_ubyte * len(gray)).from_buffer_copy(gray)
        buffer = _out(self._crypto, 9, "CryptographicBuffer.CreateFromByteArray", c_uint32(len(gray)), data,
                      argtypes=(c_uint32, POINTER(ctypes.c_ubyte)))
        bitmap = operation = info = result = 0
        try:
            bitmap = _out(self._bitmaps, 9, "SoftwareBitmap.CreateCopyFromBuffer", buffer, GRAY8, width, height,
                          argtypes=(c_void_p, c_int, c_int, c_int))
            operation = _out(self._engine, 6, "OcrEngine.RecognizeAsync", bitmap, argtypes=(c_void_p,))
            info = _query(operation, IID_ASYNC_INFO)
            deadline, status = time.monotonic() + TIMEOUT_S, c_int(STARTED)
            while True:
                _check(_call(info, 7, byref(status), argtypes=(POINTER(c_int),)), "IAsyncInfo.Status")
                if status.value != STARTED or time.monotonic() > deadline:
                    break
                time.sleep(0.003)
            if status.value != COMPLETED:
                raise OSError("Windows OCR didn't finish" if status.value == STARTED else "Windows OCR failed")
            result = _out(operation, 8, "IAsyncOperation.GetResults")
            return [_words(line) for line in _items(_out(result, 6, "OcrResult.Lines"))]
        finally:
            for obj in (result, info, operation, bitmap, buffer):
                _release(obj)

    def close(self) -> None:
        for name in ("_engine", "_crypto", "_bitmaps"):
            _release(getattr(self, name, 0))
            setattr(self, name, 0)


def _query(obj: int, iid: str) -> int:
    result = c_void_p()
    _check(_call(obj, 0, byref(_GUID.of(iid)), byref(result), argtypes=(POINTER(_GUID), POINTER(c_void_p))),
           "QueryInterface")
    return result.value or 0


def _words(line: int) -> list[Word]:
    words = []
    for word in _items(_out(line, 6, "OcrLine.Words")):
        rect, text = _Rect(), c_void_p()
        _check(_call(word, 6, byref(rect), argtypes=(POINTER(_Rect),)), "OcrWord.BoundingRect")
        _check(_call(word, 7, byref(text), argtypes=(POINTER(c_void_p),)), "OcrWord.Text")
        words.append(Word(_text(text), rect.x, rect.y, rect.width, rect.height))
        _release(word)
    _release(line)
    return words
