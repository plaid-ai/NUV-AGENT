"""Release-time build identity.

The release workflow replaces COMPONENT_SHA before creating the source archive.
Environment overrides remain available for non-release developer builds.
"""

AGENT_VERSION = "0.1.121"
COMPONENT_SHA = "unknown"
