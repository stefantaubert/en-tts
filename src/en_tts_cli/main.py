
import pickle
import shutil
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from logging import Logger
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Callable, List
from typing import OrderedDict as ODType
from typing import cast

import numpy as np
import pandas as pd
import torch
import wget
from dict_from_dict import create_dict_from_dict
from dict_from_g2pE import transcribe_with_g2pE
from english_text_normalization import *
from english_text_normalization.normalization_pipeline import (
  execute_pipeline, general_pipeline, remove_whitespace_before_sentence_punctuation)
from ordered_set import OrderedSet
from pandas import DataFrame
from pronunciation_dictionary import (DeserializationOptions, MultiprocessingOptions,
                                      PronunciationDict, SerializationOptions,
                                      get_weighted_pronunciation, load_dict, save_dict)
from pronunciation_dictionary_utils import merge_dictionaries, select_single_pronunciation
from pronunciation_dictionary_utils_cli.pronunciations_map_symbols_json import \
  identify_and_apply_mappings
from tacotron import Synthesizer as TacotronSynthesizer
from tacotron import get_speaker_mapping
from tacotron_cli import *
from tqdm import tqdm
from txt_utils_cli import extract_vocabulary_from_text
from txt_utils_cli.replacement import replace_text
from txt_utils_cli.transcription import transcribe_text_using_dict
from unidecode import unidecode_expect_ascii
from waveglow import CheckpointWaveglow
from waveglow import Synthesizer as WaveglowSynthesizer
from waveglow import TacotronSTFT, float_to_wav, normalize_wav, try_copy_to

from en_tts_cli.argparse_helper import (parse_existing_directory, parse_existing_file, parse_path,
                                        parse_positive_integer)
from en_tts_cli.arpa_ipa_mapping import ARPA_IPA_MAPPING
from en_tts_cli.types import ExecutionResult


def get_device():
  if torch.cuda.is_available():
    return torch.device("cuda:0")
  return torch.device("cpu")


def init_from_mel_batch_parser(parser: ArgumentParser) -> Callable[[str, str], None]:
  parser.description = "Command-line interface for synthesizing English texts into speech."
  parser.add_argument("input", type=str, metavar="INPUT",
                      help="text input")
  return synthesize_ns


def synthesize_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  text = cast(str, ns.input)
  synthesize(text, "English", logger, flogger)


LJS_DUR_DICT = "https://zenodo.org/record/7499098/files/pronunciations.dict"
CMU_IPA_DICT = "https://zenodo.org/record/7500805/files/pronunciations.dict"
TACO_CKP = "https://zenodo.org/records/10107104/files/101000.pt"
WG_CKP = "https://tuc.cloud/index.php/s/yBRaWz5oHrFwigf/download/LJS-v3-580000.pt"


def synthesize(text: str, input_format: str, logger: Logger, flogger: Logger):
  n_jobs = 1
  maxtasksperchild = None

  conf_dir = Path(gettempdir()) / "en-tts"
  conf_dir.mkdir(parents=True, exist_ok=True)
  work_dir = conf_dir / "tmp"
  if work_dir.exists():
    shutil.rmtree(work_dir)
  work_dir.mkdir(parents=True, exist_ok=True)

  ljs_dict_path = conf_dir / "ljs.dict"
  ljs_dict_pkl_path = conf_dir / "ljs.dict.pkl"
  cmu_dict_path = conf_dir / "cmu.dict"
  cmu_dict_pkl_path = conf_dir / "cmu.dict.pkl"
  taco_path = conf_dir / "tacotron.pt"
  wg_path = conf_dir / "waveglow.pt"
  max_decoder_steps = 3000
  sigma = 1.0
  denoiser_strength = 0.0005
  seed = 1
  device = get_device()
  # input_format = "English"
  # input_format = "IPA"
  silence_sentences = 0.2
  silence_paragraphs = 1.0
  serialize_log_opts = SerializationOptions("TAB", False, True)
  loglevel = 0

  punctuation = {".", "!", "?", "\"", ":", "'", ";", ",", "[", "]", "(", ")"}

  if input_format == "English":
    if loglevel >= 1:
      logfile = work_dir / "text.txt"
      logfile.write_text(text, "utf-8")
      flogger.info(f"Text: {logfile.absolute()}")

    text_normed = normalize_eng_text(text)
    if text_normed == text:
      flogger.info("No normalization applied.")
    else:
      flogger.info("Normalization was applied.")
      if loglevel >= 1:
        logfile = work_dir / "text.normed.txt"
        logfile.write_text(text_normed, "utf-8")
        flogger.info(f"Normed text: {logfile.absolute()}")

    vocabulary = extract_vocabulary_from_text(
      text_normed, "\n", " ", False, n_jobs, maxtasksperchild, 2_000_000)

    if loglevel >= 1:
      logfile = work_dir / "vocabulary.txt"
      logfile.write_text("\n".join(vocabulary), "utf-8")
      flogger.info(f"Vocabulary: {logfile.absolute()}")

    if not ljs_dict_path.is_file():
      logger.info("Downloading LJS dictionary ...")
      wget.download(LJS_DUR_DICT, str(ljs_dict_path.absolute()))
      logger.info("Loading LJS dictionary...")
      # 78k lines
      ljs_dict = load_dict(ljs_dict_path, "UTF-8", DeserializationOptions(
        False, False, False, True), MultiprocessingOptions(1, maxtasksperchild, 100_000))
      save_obj(ljs_dict, ljs_dict_pkl_path)
    else:
      logger.info("Loading LJS dictionary...")
      ljs_dict = load_obj(ljs_dict_pkl_path)

    dict1, oov1 = create_dict_from_dict(vocabulary, ljs_dict, trim={
    }, split_on_hyphen=False, ignore_case=False, n_jobs=1, maxtasksperchild=maxtasksperchild, chunksize=10_000)

    if loglevel >= 1:
      logfile = work_dir / "dict1.dict"
      save_dict(dict1, logfile, "utf-8", serialize_log_opts)
      flogger.info(f"Dict1: {logfile.absolute()}")
      if len(oov1) > 0:
        logfile = work_dir / "oov1.txt"
        logfile.write_text("\n".join(oov1), "utf-8")
        flogger.info(f"OOV1: {logfile.absolute()}")

    changed_word_count = select_single_pronunciation(dict1, mode="highest-weight", seed=None,
                                                     mp_options=MultiprocessingOptions(1, maxtasksperchild, 1_000))

    if loglevel >= 1 and changed_word_count > 0:
      logfile = work_dir / "dict1.single.dict"
      save_dict(dict1, logfile, "utf-8", serialize_log_opts)
      flogger.info(f"Dict1 (single pronunciation): {logfile.absolute()}")

    oov2 = OrderedSet()
    if len(oov1) > 0:
      dict2, oov2 = create_dict_from_dict(oov1, ljs_dict, trim=punctuation, split_on_hyphen=True,
                                          ignore_case=True, n_jobs=1, maxtasksperchild=maxtasksperchild, chunksize=10_000)

      if loglevel >= 1:
        logfile = work_dir / "dict2.dict"
        save_dict(dict2, logfile, "utf-8", serialize_log_opts)
        flogger.info(f"Dict1: {logfile.absolute()}")
        if len(oov2) > 0:
          logfile = work_dir / "oov2.txt"
          logfile.write_text("\n".join(oov2), "utf-8")
          flogger.info(f"OOV2: {logfile.absolute()}")

      changed_word_count = select_single_pronunciation(dict2, mode="highest-weight", seed=None,
                                                       mp_options=MultiprocessingOptions(1, maxtasksperchild, 1_000))

      if loglevel >= 1 and changed_word_count > 0:
        logfile = work_dir / "dict2.single.dict"
        save_dict(dict2, logfile, "utf-8", serialize_log_opts)
        flogger.info(f"Dict2 (single pronunciation): {logfile.absolute()}")

      merge_dictionaries(dict1, dict2, mode="add")

      if loglevel >= 1:
        logfile = work_dir / "dict1+dict2.dict"
        save_dict(dict1, logfile, "utf-8", serialize_log_opts)
        flogger.info(f"Dict1+Dict2: {logfile.absolute()}")

    oov3 = OrderedSet()
    if len(oov2) > 0:

      if not cmu_dict_path.is_file():
        logger.info("Downloading CMU dictionary...")
        wget.download(CMU_IPA_DICT, str(cmu_dict_path.absolute()))

        logger.info("Loading CMU dictionary...")
        cmu_dict = load_dict(cmu_dict_path, "UTF-8", DeserializationOptions(
          False, False, False, False), MultiprocessingOptions(n_jobs, maxtasksperchild, 100_000))
        save_obj(cmu_dict, cmu_dict_pkl_path)
      else:
        cmu_dict = load_obj(cmu_dict_pkl_path)

      dict3, oov3 = create_dict_from_dict(oov2, cmu_dict, trim=punctuation, split_on_hyphen=True,
                                          ignore_case=True, n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=10_000)

      if loglevel >= 1:
        logfile = work_dir / "dict3.dict"
        save_dict(dict3, logfile, "utf-8", serialize_log_opts)
        flogger.info(f"Dict3: {logfile.absolute()}")
        if len(oov3) > 0:
          logfile = work_dir / "oov3.txt"
          logfile.write_text("\n".join(oov3), "utf-8")
          flogger.info(f"OOV3: {logfile.absolute()}")

      merge_dictionaries(dict1, dict3, mode="add")

      if loglevel >= 1:
        logfile = work_dir / "dict1+dict2+dict3.dict"
        save_dict(dict1, logfile, "utf-8", serialize_log_opts)
        flogger.info(f"Dict1+Dict2+Dict3: {logfile.absolute()}")

    if len(oov3) > 0:
      dict4 = transcribe_with_g2pE(oov3, weight=1, trim=punctuation,
                                   split_on_hyphen=True, n_jobs=1, maxtasksperchild=None, chunksize=100_000)

      if loglevel >= 1:
        logfile = work_dir / "dict4.arpa.dict"
        save_dict(dict4, logfile, "utf-8", serialize_log_opts)
        flogger.info(f"Dict4 (ARPA): {logfile.absolute()}")
      # mappings = {}
      # TODO hier weitermachen
      identify_and_apply_mappings(logger, flogger, dict4, ARPA_IPA_MAPPING, partial_mapping=False,
                                  mp_options=MultiprocessingOptions(1, maxtasksperchild, 100_000))
      
      
    text_ipa = transcribe_text_using_dict(dict1, text_normed, "\n", "|", " ", seed=None, ignore_missing=False,
                                          n_jobs=n_jobs, maxtasksperchild=maxtasksperchild, chunksize=2_000_000)

    if loglevel >= 1:
      logfile = work_dir / "ipa.txt"
      logfile.write_text(text_ipa, "utf-8")
      flogger.info(f"IPA: {logfile.absolute()}")

      logfile = work_dir / "ipa.readable.txt"
      logfile.write_text(text_ipa.replace("|", ""), "utf-8")
      flogger.info(f"IPA (readable): {logfile.absolute()}")

    text_ipa = replace_text(text_ipa, " ", "SIL0", disable_regex=True)
    text_ipa = replace_text(text_ipa, ",|SIL0", ",|SIL1", disable_regex=True)
    text_ipa = replace_text(text_ipa, r"(\.|\!|\?)", r"\1|SIL2", disable_regex=False)
    text_ipa = replace_text(text_ipa, ";|SIL0", ";|SIL2", disable_regex=True)

    if loglevel >= 1:
      logfile = work_dir / "ipa.silence.txt"
      logfile.write_text(text_ipa, "utf-8")
      flogger.info(f"IPA: {logfile.absolute()}")

  elif input_format == "IPA":
    text_ipa = text
    if loglevel >= 1:
      logfile = work_dir / "ipa.txt"
      logfile.write_text(text_ipa, "utf-8")
      flogger.info(f"IPA: {logfile.absolute()}")

  else:
    raise NotImplementedError()

  if not taco_path.is_file():
    logger.info("Downloading Tacotron checkpoint...")
    wget.download(TACO_CKP, str(taco_path.absolute()))

  logger.info(f"Loading Tacotron checkpoint from: {taco_path.absolute()} ...")
  checkpoint = load_checkpoint(taco_path, device)

  synth = TacotronSynthesizer(
    checkpoint=checkpoint,
    custom_hparams=None,
    device=device,
    logger=logger,
  )

  if not wg_path.is_file():
    logger.info("Downloading Waveglow checkpoint...")
    wget.download(WG_CKP, str(wg_path.absolute()))

  logger.info("Loading Waveglow checkpoint...")
  wg_checkpoint = CheckpointWaveglow.load(wg_path, device, logger)
  wg_synth = WaveglowSynthesizer(
    checkpoint=wg_checkpoint,
    custom_hparams=None,
    device=device,
    logger=logger,
  )

  # taco_stft = TacotronSTFT(wg_synth.hparams, device, logger)

  first_speaker = list(get_speaker_mapping(checkpoint).keys())[0]
  resulting_wavs = []
  paragraphs = text_ipa.split("\n\n")
  for paragraph_nr, paragraph in enumerate(paragraphs):
    sentences = paragraph.split("\n")
    for sentence_nr, sentence in enumerate(sentences):
      if sentence == "":
        continue

      sentence_id = f"{paragraph_nr+1}-{sentence_nr+1}"

      symbols = sentence.split("|")
      logger.info(f"Synthesizing {sentence_id} step 1/2...")
      inf_sent_output = synth.infer(
        symbols=symbols,
        speaker=first_speaker,
        include_stats=False,
        max_decoder_steps=max_decoder_steps,
        seed=seed,
      )

      if loglevel >= 2:
        logfile = work_dir / f"{sentence_id}.npy"
        np.save(logfile, inf_sent_output.mel_outputs_postnet)
        flogger.info(f"Tacotron output: {logfile.absolute()}")

      mel_var = torch.FloatTensor(inf_sent_output.mel_outputs_postnet)
      mel_var = try_copy_to(mel_var, device)
      # del mel
      # mel_var = torch.autograd.Variable(mel_torch)
      mel_var = mel_var.unsqueeze(0)
      logger.info(f"Synthesizing {sentence_id} step 2/2...")
      inference_result = wg_synth.infer(mel_var, sigma, denoiser_strength, seed)
      # del mel_var
      wav_inferred_denoised_normalized = normalize_wav(inference_result.wav_denoised)

      if loglevel >= 2:
        logfile = work_dir / f"{sentence_id}.wav"
        float_to_wav(wav_inferred_denoised_normalized, logfile)
        flogger.info(f"Waveglow output: {logfile.absolute()}")

      resulting_wavs.append(wav_inferred_denoised_normalized)
      is_last_sentence_in_paragraph = sentence_nr == len(sentences) - 1
      if silence_sentences > 0 and not is_last_sentence_in_paragraph:
        pause_samples = np.zeros(
          (get_sample_count(wg_synth.hparams.sampling_rate, silence_sentences),))
        resulting_wavs.append(pause_samples)

    is_last_paragraph = paragraph_nr == len(paragraphs) - 1
    if silence_paragraphs > 0 and not is_last_paragraph:
      pause_samples = np.zeros(
        (get_sample_count(wg_synth.hparams.sampling_rate, silence_paragraphs),))
      resulting_wavs.append(pause_samples)

  if len(resulting_wavs) > 0:
    resulting_wav = np.concatenate(tuple(resulting_wavs), axis=-1)
    float_to_wav(resulting_wav, work_dir / "result.wav", sample_rate=wg_synth.hparams.sampling_rate)
    logger.info(f'Saved to: {work_dir / "result.wav"}')

  return True


def get_sample_count(sampling_rate: int, duration_s: float):
  return int(round(sampling_rate * duration_s, 0))


def save_obj(obj: Any, path: Path) -> None:
  assert isinstance(path, Path)
  assert path.parent.exists() and path.parent.is_dir()
  with open(path, mode="wb") as file:
    pickle.dump(obj, file)


def load_obj(path: Path) -> Any:
  assert isinstance(path, Path)
  assert path.is_file()
  with open(path, mode="rb") as file:
    return pickle.load(file)


def normalize_eng_text(text: str) -> str:
  text = execute_pipeline(
    text,
    (
      normalize_emails_and_at,
      remove_underscore_characters,
      remove_equal_sign,
      add_space_around_dashes,
      replace_ie_with_that_is,
      replace_eg_with_for_example,
      replace_etc_with_et_cetera,
      replace_nos_with_numbers,
      replace_no_with_number,
      geo_to_george,
      write_out_month_abbreviations,
      normalize_today_tomorrow_and_tonight,
      normalize_king_name_followed_by_roman_numeral,
      normalize_am_and_pm,
      normalize_pounds_shillings_and_pence,
      normalize_temperatures_general,
      normalize_degrees_minutes_and_seconds,
      normalize_all_units,
      normalize_percent,
      expand_and_a_half,
      replace_hyphen_between_numbers_with_to,
      normalize_second_and_third_when_abbr_with_d,
      normalize_numbers,
      expand_abbreviations,
      remove_dot_after_single_capital_letters,
      replace_and_sign_with_word_and,
      remove_double_hyphen_before_or_after_colon,
      normalize_three_and_four_dots,
      replace_four_hyphens_by_two,
      add_space_around_dashes,
      remove_sic,
      remove_stars,
      remove_whitespace_before_sentence_punctuation,
      strip,
      unidecode_expect_ascii,
    )
  )

  return text
