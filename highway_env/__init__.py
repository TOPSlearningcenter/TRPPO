import os
import sys

__version__ = "1.9.1"

try:
    from farama_notifications import notifications

    if "highway_env" in notifications and __version__ in notifications["gymnasium"]:
        print(notifications["highway_env"][__version__], file=sys.stderr)

except Exception:  # nosec
    pass

# Hide pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from gymnasium.envs.registration import register
from highway_env.envs.common.abstract import MultiAgentWrapper


def register_highway_envs():
    """Import the envs module so that envs register themselves."""
    # highway_env.py
    register(
        id="highway-v0",
        entry_point="highway_env.envs:HighwayEnv_0",
    )

    register(
        id="highway-fast-v0",
        entry_point="highway_env.envs:HighwayEnvFast_0",
    )

register_highway_envs()
