import shutil
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from logging import Logger, getLogger
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Callable, List
from typing import OrderedDict as ODType
from typing import cast

from en_tts_cli.logging_configuration import configure_root_logger
from en_tts_cli.main import normalize_eng_text, synthesize


def test_component_english():
  synthesize("This is a test!", "English", getLogger(), getLogger("flogger"))


def test_component_ipa():
  synthesize('ð|ˈɪ|s|SIL0|ˈɪ|z|SIL0|ə|SIL0|tː|ˈɛ|s|t|!|SIL2\n\nð|ˈɪ|s|SIL0|ˈɪ|z|SIL0|ə|SIL0|tː|ˈɪ|s|t|!|SIL2',
             "IPA", getLogger(), getLogger("flogger"))
