from argparse import ArgumentParser, Namespace
from typing import Callable, Optional

import torch
from english_text_normalization import *
from ordered_set import OrderedSet
from pronunciation_dictionary import PronunciationDict, SerializationOptions, save_dict
from tacotron_cli import *

from en_tts import *
from en_tts_cli.argparse_helper import (parse_device, parse_float_between_zero_and_one,
                                        parse_non_empty_or_whitespace, parse_non_negative_float,
                                        parse_non_negative_integer, parse_positive_integer)
from en_tts_cli.globals import get_conf_dir, get_work_dir
from en_tts_cli.logging_configuration import get_file_logger, get_logger


def init_synthesize_eng_parser(parser: ArgumentParser) -> Callable[[str, str], None]:
  parser.description = "Synthesize English texts into speech."
  parser.add_argument("input", type=parse_non_empty_or_whitespace, metavar="INPUT",
                      help="text input")
  parser.add_argument("--skip-normalization", action="store_true", help="skip normalization step")
  parser.add_argument("--skip-sentence-separation", action="store_true",
                      help="skip sentence separation step")
  add_common_arguments(parser)

  def parse_ns(ns: Namespace):
    synthesize_english(ns.input, ns.max_decoder_steps, ns.sigma, ns.denoiser_strength, ns.seed, ns.device,
                       ns.silence_sentences, ns.silence_paragraphs, ns.loglevel, ns.skip_normalization, ns.skip_sentence_separation)
  return parse_ns


def init_synthesize_ipa_parser(parser: ArgumentParser) -> Callable[[str, str], None]:
  parser.description = "Synthesize English IPA-transcribed texts into speech."
  parser.add_argument("input", type=parse_non_empty_or_whitespace, metavar="INPUT",
                      help="text input")
  add_common_arguments(parser)

  def parse_ns(ns: Namespace):
    synthesize_ipa(ns.input, ns.max_decoder_steps, ns.sigma, ns.denoiser_strength, ns.seed,
                   ns.device, ns.silence_sentences, ns.silence_paragraphs, ns.loglevel)

  return parse_ns


def add_common_arguments(parser: ArgumentParser) -> None:
  parser.add_argument("--loglevel", metavar="LEVEL", type=int,
                      choices=[0, 1, 2], help="log-level", default=0)
  parser.add_argument("--silence-sentences", metavar="SECONDS", type=parse_non_negative_float,
                      help="add silence between sentences (in seconds)", default=0.2)
  parser.add_argument("--silence-paragraphs", metavar="SECONDS", type=parse_non_negative_float,
                      help="add silence between paragraphs (in seconds)", default=1.0)
  parser.add_argument("--seed", type=parse_non_negative_integer, metavar="SEED",
                      help="seed for generating speech", default=0)
  add_device_argument(parser)
  add_max_decoder_steps_argument(parser)
  add_denoiser_and_sigma_arguments(parser)


def add_denoiser_and_sigma_arguments(parser: ArgumentParser) -> None:
  parser.add_argument("--sigma", metavar="SIGMA", type=parse_float_between_zero_and_one,
                      default=1.0, help="sigma used for WaveGlow synthesis")
  parser.add_argument("--denoiser-strength", metavar="STRENGTH", default=0.0005,
                      type=parse_float_between_zero_and_one, help='strength of denoising to remove model bias after WaveGlow synthesis')


def add_max_decoder_steps_argument(parser: ArgumentParser) -> None:
  parser.add_argument('--max-decoder-steps', type=parse_positive_integer, metavar="STEPS",
                      default=5000, help="maximum step count before synthesis is stopped")


def add_device_argument(parser: ArgumentParser) -> None:
  parser.add_argument("--device", metavar="DEVICE", type=parse_device,
                      default=get_default_device(), help="use this device, e.g., \"cpu\" or \"cuda:0\"")


def synthesize_english(text: str, max_decoder_steps: int, sigma: float, denoiser_strength: float, seed: int, device: torch.device, silence_sentences: float, silence_paragraphs: float, loglevel: int, skip_normalization: bool, skip_sentence_separation: bool):
  text_ipa = convert_eng_to_ipa(text, loglevel, skip_normalization, skip_sentence_separation)
  synthesize_ipa_core(text_ipa, max_decoder_steps, sigma, denoiser_strength,
                      seed, device, silence_sentences, silence_paragraphs, loglevel)


def synthesize_ipa(text_ipa: str, max_decoder_steps: int, sigma: float, denoiser_strength: float, seed: int, device: torch.device, silence_sentences: float, silence_paragraphs: float, loglevel: int):
  flogger = get_file_logger()
  work_dir = get_work_dir()

  if loglevel >= 1:
    try_log_text(text_ipa, "text")

  synthesize_ipa_core(text_ipa, max_decoder_steps, sigma, denoiser_strength,
                      seed, device, silence_sentences, silence_paragraphs, loglevel)


def convert_eng_to_ipa(text: str, loglevel: int, skip_normalization: bool, skip_sentence_separation: bool) -> str:
  logger = get_logger()
  flogger = get_file_logger()
  conf_dir = get_conf_dir()
  work_dir = get_work_dir()

  t = Transcriber(conf_dir)

  text_ipa = t.transcribe_to_ipa(text, skip_normalization, skip_sentence_separation)

  if loglevel >= 1:
    for txt, name in (
      (text, "text"),
      (t.text_normed, "text.normed"),
      (t.text_sentenced, "text.sentenced"),
      (t.text_ipa, "text.ipa"),
      (t.text_ipa_readable, "text.ipa.readable"),
      (text_ipa, "text.ipa.silenced"),
    ):
      try_log_text(txt, name)

    for v, name in (
      (t.vocabulary, "vocabulary"),
      (t.oov1, "oov1"),
      (t.oov2, "oov2"),
      (t.oov3, "oov3"),
    ):
      try_log_voc(v, name)

    for d, name in (
      (t.dict1, "dict1"),
      (t.dict1_single, "dict1.single"),
      (t.dict2, "dict2"),
      (t.dict2_single, "dict2.single"),
      (t.dict1_2, "dict1+2"),
      (t.dict3, "dict3"),
      (t.dict3_single, "dict3.single"),
      (t.dict1_2_3, "dict1+2+3"),
      (t.dict4_arpa, "dict4.arpa"),
      (t.dict4, "dict4"),
      (t.dict4_single, "dict4.single"),
      (t.dict1_2_3_4, "dict1+2+3+4"),
    ):
      try_log_dict(d, name)
  return text_ipa


def synthesize_ipa_core(text_ipa: str, max_decoder_steps: int, sigma: float, denoiser_strength: float, seed: int, device: torch.device, silence_sentences: float, silence_paragraphs: float, loglevel: int):
  logger = get_logger()
  flogger = get_file_logger()
  conf_dir = get_conf_dir()
  work_dir = get_work_dir()

  synthesizer = Synthesizer(conf_dir, device)
  audio = synthesizer.synthesize(text_ipa, max_decoder_steps, seed, sigma,
                                 denoiser_strength, silence_sentences, silence_paragraphs, silent=loglevel == 0)
  save_audio(audio, work_dir / "result.unnormed.wav")

  try:
    normalize_audio(work_dir / "result.unnormed.wav", work_dir / "result.wav")
    logger.info(f'Saved to: {work_dir / "result.wav"}')
  except Exception as error:
    logger.warning("Normalization was not possible", exc_info=error)
    logger.info(f'Saved to: {work_dir / "result.unnormed.wav"}')


def try_log_dict(dictionary: Optional[PronunciationDict], name: str) -> None:
  if dictionary:
    work_dir = get_work_dir()
    logfile = work_dir / f"{name}.dict"
    save_dict(dictionary, logfile, "utf-8", SerializationOptions("TAB", False, True))
    flogger = get_file_logger()
    flogger.info(f"{name}: {logfile.absolute()}")


def try_log_voc(vocabulary: Optional[OrderedSet[str]], name: str) -> None:
  if vocabulary:
    work_dir = get_work_dir()
    logfile = work_dir / f"{name}.txt"
    logfile.write_text("\n".join(vocabulary), "utf-8")
    flogger = get_file_logger()
    flogger.info(f"{name}: {logfile.absolute()}")


def try_log_text(text: Optional[str], name: str) -> None:
  if text:
    work_dir = get_work_dir()
    logfile = work_dir / f"{name}.txt"
    logfile.write_text(text, "utf-8")
    flogger = get_file_logger()
    flogger.info(f"{name}: {logfile.absolute()}")
