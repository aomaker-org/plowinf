# Path: plowinf/src/local_code_plower/__init__.py
"""Compatibility layer routing local_code_plower imports to plowinf."""

import sys
import plowinf
from plowinf import *  # noqa: F401, F403
from plowinf import __all__, __version__

# Register alias in sys.modules
sys.modules["local_code_plower"] = plowinf
sys.modules["local_code_plower.backends"] = sys.modules.get("plowinf.backends")
sys.modules["local_code_plower.forensics"] = sys.modules.get("plowinf.forensics")
sys.modules["local_code_plower.patcher"] = sys.modules.get("plowinf.patcher")
sys.modules["local_code_plower.queue"] = sys.modules.get("plowinf.queue")
sys.modules["local_code_plower.safety"] = sys.modules.get("plowinf.safety")
sys.modules["local_code_plower.worker"] = sys.modules.get("plowinf.worker")
sys.modules["local_code_plower.cli"] = sys.modules.get("plowinf.cli")

# end of file: plowinf/src/local_code_plower/__init__.py
