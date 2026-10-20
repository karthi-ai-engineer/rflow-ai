"""sst: record from the microphone and transcribe it locally."""
import os
import sys
from pathlib import Path

__version__ = "2.5.3"  # the one place to change it; the installer and release tags follow it

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
