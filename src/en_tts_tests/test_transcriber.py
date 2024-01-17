from en_tts.transcriber import Transcriber
from en_tts_tests.helper import get_tests_dir


def test_component():
  conf_dir = get_tests_dir()

  t = Transcriber(conf_dir)

  text = "This is a test abbb? And I'm there 31."
  text_ipa = t.transcribe_to_ipa(text, False, False)

  assert text_ipa == 'ð|ˈɪ|s|SIL0|ˈɪ|z|SIL0|ə|SIL0|tː|ˈɛ|s|t|SIL0|ˈæ|b|?|SIL2\nə|n˘|d|SIL0|ˈaɪ˘|m|SIL0|ð|ˈɛr˘|SIL0|θ|ˈʌr|d˘|ˌi|-|wː|ˈʌː|nː|.|SIL2'
