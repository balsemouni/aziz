"""T5: verify CAG imports."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from cag_config import CAGConfig, get_config_preset
from cag_system import CAGSystemFreshSession
print("CAG: all top-level imports OK — T5 PASS")
