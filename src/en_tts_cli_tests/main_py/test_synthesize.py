from logging import getLogger

from en_tts_cli.main import synthesize_english


def test_component_english():
  synthesize_english("This is a test!", "English", getLogger(), getLogger("flogger"))


def test_component_ipa():
  synthesize_english('ð|ˈɪ|s|SIL0|ˈɪ|z|SIL0|ə|SIL0|tː|ˈɛ|s|t|!|SIL2\n\nð|ˈɪ|s|SIL0|ˈɪ|z|SIL0|ə|SIL0|tː|ˈɪ|s|t|!|SIL2',
                     "IPA", getLogger(), getLogger("flogger"))
