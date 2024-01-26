from logging import getLogger

import numpy as np

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

  np.testing.assert_array_almost_equal(
    audio[:10],
    np.array(
      [
        4.06468927e-04, -1.38543465e-03, -2.93170044e-04, -3.17878075e-05,
        -1.35282415e-03, -1.24711674e-04, -6.18043647e-04, -2.78891705e-04,
        9.47587105e-05, -4.59646282e-04
      ],
      dtype=np.float64,
    )
  )
  assert audio.dtype == np.float64
  assert audio.shape == (58682,)


def test_empty():
  conf_dir = get_tests_dir()

  s = Synthesizer(conf_dir)

  text = ''

  audio = s.synthesize(text)
  assert audio.dtype == np.float64
  assert audio.shape == (0,)
