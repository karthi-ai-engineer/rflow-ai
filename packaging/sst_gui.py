"""Entry point of the installed tray app, Rflow.exe (no console window). See sst/app.py. A running Rflow is asked to show
its window before the app's heavy imports (sst/instance.py)."""
import sys

from sst.instance import launch

sys.exit(launch())
