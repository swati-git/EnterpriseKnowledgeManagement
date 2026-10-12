import os
import sys
from pathlib import Path

# Offline tests: model and search clients are constructed at import time,
# so give them placeholder keys. No network calls are made; every chain is stubbed.
os.environ.setdefault("COHERE_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
