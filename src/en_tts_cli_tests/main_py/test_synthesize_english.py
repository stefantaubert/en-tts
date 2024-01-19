from en_tts.helper import get_default_device
from en_tts_cli.cli import reset_work_dir
from en_tts_cli.globals import get_work_dir
from en_tts_cli.main import synthesize_english


def test_component():
  text = "This is a test abbb? And I'm there 31."
  work_dir = get_work_dir()

  reset_work_dir()
  synthesize_english(text, 5000, 1.0, 0.0005, 0, get_default_device(),
                     0.2, 1.0, 2, False, False, work_dir / "output.wav")
