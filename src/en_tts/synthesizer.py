import logging
from logging import getLogger
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
from tacotron import Synthesizer as TacotronSynthesizer
from tacotron import get_speaker_mapping
from tqdm import tqdm
from waveglow import Synthesizer as WaveglowSynthesizer
from waveglow import try_copy_to

from en_tts.globals import DEFAULT_CONF_DIR
from en_tts.helper import get_default_device, get_sample_count
from en_tts.resources import get_taco_model, get_wg_model


class Synthesizer():
  def __init__(
      self,
      conf_dir: Path = DEFAULT_CONF_DIR,
      device: torch.device = get_default_device()
  ) -> None:
    logger = getLogger(__name__)

    tacotron_logger = getLogger("tacotron")
    tacotron_logger.parent = logger
    tacotron_logger.setLevel(logging.WARNING)

    waveglow_logger = getLogger("waveglow")
    waveglow_logger.parent = logger
    waveglow_logger.setLevel(logging.WARNING)

    self._device = device
    self._conf_dir = conf_dir
    tacotron_ckp = get_taco_model(conf_dir, device)
    self._tacotron_ckp = tacotron_ckp
    self._tacotron = TacotronSynthesizer(
      checkpoint=tacotron_ckp,
      custom_hparams=None,
      device=device,
    )
    waveglow_ckp = get_wg_model(conf_dir, device)
    self._waveglow_ckp = waveglow_ckp
    self._waveglow = WaveglowSynthesizer(
      waveglow_ckp,
      device=device,
    )
    self._speaker = list(get_speaker_mapping(tacotron_ckp).keys())[0]
    self._paragraph_sep = "\n\n"
    self._sentence_sep = "\n"
    self._symbol_seperator = "|"

  def synthesize(self, text_ipa: str, max_decoder_steps: int = 5000, seed: int = 0, sigma: float = 1.0, denoiser_strength: float = 0.0005, silence_sentences: float = 0.2, silence_paragraphs: float = 1.0, silent: bool = False) -> npt.NDArray[np.float64]:
    resulting_wavs = []
    paragraph_sentences = [
      [
        sentence
        for sentence in paragraph.split(self._sentence_sep)
        if sentence != ""
      ]
      for paragraph in text_ipa.split(self._paragraph_sep)
    ]
    sentence_count = sum(1 for p in paragraph_sentences for s in p)

    with tqdm(desc="Synthesizing", total=sentence_count, unit=" sent", disable=silent) as pbar:
      for paragraph_nr, paragraph in enumerate(paragraph_sentences):
        for sentence_nr, sentence in enumerate(paragraph):
          # sentence_id = f"{paragraph_nr+1}-{sentence_nr+1}"
          symbols = sentence.split(self._symbol_seperator)
          # logger.debug(f"Synthesizing {sentence_id} step 1/2...")
          inf_sent_output = self._tacotron.infer(
            symbols=symbols,
            speaker=self._speaker,
            include_stats=False,
            max_decoder_steps=max_decoder_steps,
            seed=seed,
          )

          # if loglevel >= 2:
          #   logfile = work_dir / f"{sentence_id}.npy"
          #   np.save(logfile, inf_sent_output.mel_outputs_postnet)
          #   logger.debug(f"Tacotron output: {logfile.absolute()}")

          mel_var_f = torch.FloatTensor(inf_sent_output.mel_outputs_postnet)
          del inf_sent_output
          mel_var = try_copy_to(mel_var_f, self._device)
          mel_var = mel_var.unsqueeze(0)
          # logger.debug(f"Synthesizing {sentence_id} step 2/2...")
          inference_result = self._waveglow.infer(
            mel_var, sigma=sigma, denoiser_strength=denoiser_strength, seed=seed)
          # wav_inferred_denoised_normalized = normalize_wav(inference_result.wav_denoised)
          del mel_var

          # if loglevel >= 2:
          #   logfile = work_dir / f"{sentence_id}.wav"
          #   float_to_wav(wav_inferred_denoised_normalized, logfile)
          #   flogger.info(f"WaveGlow output: {logfile.absolute()}")

          resulting_wavs.append(inference_result.wav_denoised)
          is_last_sentence_in_paragraph = sentence_nr == len(paragraph) - 1
          if silence_sentences > 0 and not is_last_sentence_in_paragraph:
            pause_samples = np.zeros(
              (get_sample_count(self._waveglow.hparams.sampling_rate, silence_sentences),))
            resulting_wavs.append(pause_samples)
          pbar.update(1)

        is_last_paragraph = paragraph_nr == len(paragraph_sentences) - 1
        if silence_paragraphs > 0 and not is_last_paragraph:
          pause_samples = np.zeros(
            (get_sample_count(self._waveglow.hparams.sampling_rate, silence_paragraphs),))
          resulting_wavs.append(pause_samples)

    if len(resulting_wavs) > 0:
      resulting_wav = cast(npt.NDArray[np.float64], np.concatenate(tuple(resulting_wavs), axis=-1))
      return resulting_wav
    empty_result = cast(npt.NDArray[np.float64], np.zeros((0,), np.float64))
    return empty_result
