from logging import getLogger
from pathlib import Path
from typing import cast

import nltk
import torch
import wget
from pronunciation_dictionary import (DeserializationOptions, MultiprocessingOptions,
                                      PronunciationDict, load_dict)
from tacotron import CheckpointDict
from tacotron_cli import load_checkpoint
from waveglow import CheckpointWaveglow, convert_glow_files
from waveglow_cli import download_pretrained_model

from en_tts.io import load_obj, save_obj

LJS_DUR_DICT = "https://zenodo.org/record/7499098/files/pronunciations.dict"
CMU_IPA_DICT = "https://zenodo.org/record/7500805/files/pronunciations.dict"
TACO_CKP = "https://zenodo.org/records/10107104/files/101000.pt"


def get_ljs_dict(conf_dir: Path) -> PronunciationDict:
  logger = getLogger(__name__)
  conf_dir.mkdir(parents=True, exist_ok=True)
  ljs_dict_path = conf_dir / "ljs.dict"
  ljs_dict_pkl_path = conf_dir / "ljs.dict.pkl"

  ljs_dict: PronunciationDict
  if not ljs_dict_path.is_file():
    logger.info("Downloading LJS dictionary ...")
    wget.download(LJS_DUR_DICT, str(ljs_dict_path.absolute()))
    logger.info("Loading LJS dictionary ...")
    # 78k lines
    ljs_dict = load_dict(ljs_dict_path, "UTF-8", DeserializationOptions(
      False, False, False, True), MultiprocessingOptions(1, None, 100_000))
    save_obj(ljs_dict, ljs_dict_pkl_path)
  else:
    logger.info("Loading LJS dictionary ...")
    ljs_dict = cast(PronunciationDict, load_obj(ljs_dict_pkl_path))
  return ljs_dict


def get_cmu_dict(conf_dir: Path) -> PronunciationDict:
  logger = getLogger(__name__)
  conf_dir.mkdir(parents=True, exist_ok=True)
  cmu_dict_path = conf_dir / "cmu.dict"
  cmu_dict_pkl_path = conf_dir / "cmu.dict.pkl"

  cmu_dict: PronunciationDict
  if not cmu_dict_path.is_file():
    logger.info("Downloading CMU dictionary ...")
    wget.download(CMU_IPA_DICT, str(cmu_dict_path.absolute()))

    logger.info("Loading CMU dictionary ...")
    cmu_dict = load_dict(cmu_dict_path, "UTF-8", DeserializationOptions(
      False, False, False, False), MultiprocessingOptions(1, None, 100_000))
    save_obj(cmu_dict, cmu_dict_pkl_path)
  else:
    cmu_dict = cast(PronunciationDict, load_obj(cmu_dict_pkl_path))
  return cmu_dict


def get_wg_model(conf_dir: Path, device: torch.device) -> CheckpointWaveglow:
  logger = getLogger(__name__)
  conf_dir.mkdir(parents=True, exist_ok=True)
  wg_path = conf_dir / "waveglow.pt"
  wg_orig_path = conf_dir / "waveglow_orig.pt"

  if not wg_path.is_file():
    logger.info("Downloading WaveGlow checkpoint ...")
    download_pretrained_model(wg_orig_path, version=3)
    wg_checkpoint = convert_glow_files(wg_orig_path, wg_path, device, keep_orig=False)
    # wget.download(WG_CKP, str(wg_path.absolute()))
  else:
    logger.info("Loading WaveGlow checkpoint ...")  # from: {wg_path.absolute()} ...")
    wg_checkpoint = CheckpointWaveglow.load(wg_path, device)
  return wg_checkpoint


def get_taco_model(conf_dir: Path, device: torch.device) -> CheckpointDict:
  logger = getLogger(__name__)
  conf_dir.mkdir(parents=True, exist_ok=True)
  taco_path = conf_dir / "tacotron.pt"

  if not taco_path.is_file():
    logger.info("Downloading Tacotron checkpoint ...")
    wget.download(TACO_CKP, str(taco_path.absolute()))

  logger.info("Loading Tacotron checkpoint ...")  # from: {taco_path.absolute()} ...")
  checkpoint = load_checkpoint(taco_path, device)
  return checkpoint


def download_nltk_data():
  logger = getLogger(__name__)
  try:
    nltk.data.find('taggers/averaged_perceptron_tagger.zip')
  except LookupError:
    logger.info("Downloading 'averaged_perceptron_tagger' from nltk ...")
    nltk.download('averaged_perceptron_tagger')
  try:
    nltk.data.find('corpora/cmudict.zip')
  except LookupError:
    logger.info("Downloading 'cmudict' from nltk ...")
    nltk.download('cmudict')
