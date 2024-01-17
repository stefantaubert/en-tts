
from pathlib import Path


def get_tests_dir() -> Path:
  result = Path("/tmp/en-tts.tests")
  result.mkdir(parents=True, exist_ok=True)
  return result

