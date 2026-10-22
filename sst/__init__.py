"""sst: record from the microphone and transcribe it locally."""
import os
import sys
from pathlib import Path

# numpy's OpenBLAS starts a thread and reserves its buffers for every core when it loads (+350 MB private memory and
# 11 threads on a 12-thread laptop). Rflow's numpy work is small (levels, filters, soft shadows); the speech engines
# (onnxruntime, CTranslate2) bring their own thread pools. Set before anything imports numpy: this module runs first.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

__version__ = "2.6.0"  # the one place to change it; the installer and release tags follow it

if getattr(sys, "frozen", False):  # the installed app (Rflow.exe / rflow-cli.exe, built by build_installer.cmd)
    MODELS_DIR = Path(sys.executable).parent / "models"
    RECORDINGS_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "sst" / "recordings"
else:
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    MODELS_DIR = PROJECT_DIR / "models"
    RECORDINGS_DIR = PROJECT_DIR / "recordings"
# Speech models downloaded when the user chooses them (sst.downloads): one folder for the installed app and the source
# checkout, so a model is downloaded once.
DOWNLOADS_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "sst" / "models"
