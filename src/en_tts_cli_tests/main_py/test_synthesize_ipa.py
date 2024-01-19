from en_tts.helper import get_default_device
from en_tts_cli.cli import reset_work_dir
from en_tts_cli.globals import get_work_dir
from en_tts_cli.main import synthesize_ipa


def test_component():
  text = 'ð|ˈɪ|s|SIL0|ˈɪ|z|SIL0|ə|SIL0|tː|ˈɛ|s|t|SIL0|ˈæ|b|?|SIL2\nə|n˘|d|SIL0|ˈaɪ˘|m|SIL0|ð|ˈɛr˘|SIL0|θ|ˈʌr|d˘|ˌi|-|wː|ˈʌː|nː|.|SIL2'
  work_dir = get_work_dir()

  reset_work_dir()
  synthesize_ipa(text, 5000, 1.0, 0.0005, 0, get_default_device(),
                 0.2, 1.0, 2, work_dir / "output.wav")
