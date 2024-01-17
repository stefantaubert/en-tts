from logging import getLogger

from en_tts.helper import normalize_audio
from en_tts.io import save_audio
from en_tts.synthesizer import Synthesizer
from en_tts_tests.helper import get_tests_dir


def test_component():
  conf_dir = get_tests_dir()

  s = Synthesizer(conf_dir)

  text = 'ð|ˈɪ|s|SIL0|ˈɪ|z|SIL0|ə|SIL0|tː|ˈɛ|s|t|SIL0|ˈæ|b|?|SIL2\nə|n˘|d|SIL0|ˈaɪ˘|m|SIL0|ð|ˈɛr˘|SIL0|θ|ˈʌr|d˘|ˌi|-|wː|ˈʌː|nː|.|SIL2'

  audio = s.synthesize(text)
  save_audio(audio, conf_dir / "output.wav")
  normalize_audio(conf_dir / "output.wav", conf_dir / "output_norm.wav")
  logger = getLogger(__name__)
  logger.info(conf_dir / "output_norm.wav")
  assert len(audio) > 0
