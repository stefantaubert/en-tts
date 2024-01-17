
from pathlib import Path

import numpy as np
import torch
from tacotron import Synthesizer as TacotronSynthesizer
from tacotron import get_speaker_mapping
from tqdm import tqdm
from waveglow import Synthesizer as WaveglowSynthesizer
from waveglow import normalize_wav, try_copy_to

from en_tts.globals import DEFAULT_CONF_DIR
from en_tts.helper import get_default_device, get_sample_count
from en_tts.logging import get_logger
from en_tts.resources import get_taco_model, get_wg_model

TACO_CKP = "https://zenodo.org/records/10107104/files/101000.pt"


class Synthesizer():
  def __init__(
      self,
      conf_dir: Path = DEFAULT_CONF_DIR,
      device: torch.device = get_default_device()
  ) -> None:
    logger = get_logger()
    self._device = device
    self._conf_dir = conf_dir
    tacotron_ckp = get_taco_model(conf_dir, device)
    self._tacotron_ckp = tacotron_ckp
    self._tacotron = TacotronSynthesizer(
      checkpoint=tacotron_ckp,
      custom_hparams=None,
      device=device,
      logger=logger,
    )
    waveglow_ckp = get_wg_model(conf_dir, device)
    self._waveglow_ckp = waveglow_ckp
    self._waveglow = WaveglowSynthesizer(
      checkpoint=waveglow_ckp,
      custom_hparams=None,
      device=device,
      logger=logger,
    )
    self._speaker = list(get_speaker_mapping(tacotron_ckp).keys())[0]
    self._paragraph_sep = "\n\n"
    self._sentence_sep = "\n"
    self._symbol_seperator = "|"

  def synthesize(self, text_ipa: str, max_decoder_steps: int = 5000, seed: int = 0, sigma: float = 1.0, denoiser_strength: float = 0.0005, silence_sentences: float = 0.2, silence_paragraphs: float = 1.0, silent: bool = False) -> np.ndarray:
    logger = get_logger()
    resulting_wavs = []
    paragraphs = text_ipa.split(self._paragraph_sep)
    for paragraph_nr, paragraph in enumerate(tqdm(paragraphs, position=0, desc="Paragraph", disable=silent)):
      sentences = paragraph.split(self._sentence_sep)
      sentences = [x for x in sentences if x != ""]
      for sentence_nr, sentence in enumerate(tqdm(sentences, position=1, desc="Sentence", disable=silent)):
        sentence_id = f"{paragraph_nr+1}-{sentence_nr+1}"

        symbols = sentence.split(self._symbol_seperator)
        logger.debug(f"Synthesizing {sentence_id} step 1/2...")
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

        mel_var = torch.FloatTensor(inf_sent_output.mel_outputs_postnet)
        del inf_sent_output
        mel_var = try_copy_to(mel_var, self._device)
        mel_var = mel_var.unsqueeze(0)
        logger.info(f"Synthesizing {sentence_id} step 2/2...")
        inference_result = self._waveglow.infer(mel_var, sigma, denoiser_strength, seed)
        wav_inferred_denoised_normalized = normalize_wav(inference_result.wav_denoised)
        del mel_var

        # if loglevel >= 2:
        #   logfile = work_dir / f"{sentence_id}.wav"
        #   float_to_wav(wav_inferred_denoised_normalized, logfile)
        #   flogger.info(f"WaveGlow output: {logfile.absolute()}")

        resulting_wavs.append(wav_inferred_denoised_normalized)
        is_last_sentence_in_paragraph = sentence_nr == len(sentences) - 1
        if silence_sentences > 0 and not is_last_sentence_in_paragraph:
          pause_samples = np.zeros(
            (get_sample_count(self._waveglow.hparams.sampling_rate, silence_sentences),))
          resulting_wavs.append(pause_samples)

      is_last_paragraph = paragraph_nr == len(paragraphs) - 1
      if silence_paragraphs > 0 and not is_last_paragraph:
        pause_samples = np.zeros(
          (get_sample_count(self._waveglow.hparams.sampling_rate, silence_paragraphs),))
        resulting_wavs.append(pause_samples)

    if len(resulting_wavs) > 0:
      resulting_wav = np.concatenate(tuple(resulting_wavs), axis=-1)
      return resulting_wav
    return np.zeros((0,))

