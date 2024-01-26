from pathlib import Path
from tempfile import gettempdir


def get_tests_dir() -> Path:
  result = Path(gettempdir()) / "en-tts.tests"
  result.mkdir(parents=True, exist_ok=True)
  return result
