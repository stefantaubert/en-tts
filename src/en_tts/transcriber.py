
import re
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Generator, Optional

from dict_from_dict import create_dict_from_dict
from dict_from_g2pE import transcribe_with_g2pE
from english_text_normalization import *
from english_text_normalization.normalization_pipeline import (
  execute_pipeline, remove_whitespace_before_sentence_punctuation)
from ordered_set import OrderedSet
from pronunciation_dictionary import MultiprocessingOptions, PronunciationDict
from pronunciation_dictionary_utils import (merge_dictionaries, replace_symbols_in_pronunciations,
                                            select_single_pronunciation)
from pronunciation_dictionary_utils_cli.pronunciations_map_symbols_json import \
  identify_and_apply_mappings
from tacotron_cli import *
from txt_utils_cli import extract_vocabulary_from_text
from txt_utils_cli.replacement import replace_text
from txt_utils_cli.transcription import transcribe_text_using_dict
from unidecode import unidecode_expect_ascii

from en_tts.arpa_ipa_mapping import ARPA_IPA_MAPPING
from en_tts.globals import DEFAULT_CONF_DIR
from en_tts.logging import get_logger
from en_tts.resources import get_cmu_dict, get_ljs_dict

LJS_DUR_DICT = "https://zenodo.org/record/7499098/files/pronunciations.dict"
CMU_IPA_DICT = "https://zenodo.org/record/7500805/files/pronunciations.dict"
TACO_CKP = "https://zenodo.org/records/10107104/files/101000.pt"


class Transcriber():
  def __init__(
      self,
      conf_dir: Path = DEFAULT_CONF_DIR,
  ) -> None:
    self._conf_dir = conf_dir
    self._ljs_dict = get_ljs_dict(conf_dir)
    self._cmu_dict = get_cmu_dict(conf_dir)
    self._symbol_separator = "|"
    self._punctuation = {".", "!", "?", ",", ":", ";", "\"", "'", "[", "]", "(", ")", "-", "—"}
    self.text_normed: Optional[str] = None
    self.text_sentenced: Optional[str] = None
    self.vocabulary: OrderedSet[str] = OrderedSet()
    self.dict1: PronunciationDict = OrderedDict()
    self.oov1: Optional[OrderedSet[str]] = None
    self.dict1_single: Optional[PronunciationDict] = None
    self.dict2: Optional[PronunciationDict] = None
    self.dict2_single: Optional[PronunciationDict] = None
    self.oov2: Optional[OrderedSet[str]] = None
    self.dict1_2: Optional[PronunciationDict] = None
    self.dict3: Optional[PronunciationDict] = None
    self.dict3_single: Optional[PronunciationDict] = None
    self.oov3: Optional[OrderedSet[str]] = None
    self.dict1_2_3: Optional[PronunciationDict] = None
    self.dict4_arpa: Optional[PronunciationDict] = None
    self.dict4: Optional[PronunciationDict] = None
    self.dict4_single: Optional[PronunciationDict] = None
    self.dict1_2_3_4: Optional[PronunciationDict] = None
    self.text_ipa: str = ""
    self.text_ipa_readable: str = ""

  def _reset_locals(self) -> None:
    self.text_normed = None
    self.text_sentenced = None
    self.vocabulary = OrderedSet()
    self.dict1 = OrderedSet()
    self.oov1 = None
    self.dict1_single = None
    self.dict2 = None
    self.dict2_single = None
    self.oov2 = None
    self.dict1_2 = None
    self.dict3 = None
    self.dict3_single = None
    self.oov3 = None
    self.dict1_2_3 = None
    self.dict4_arpa = None
    self.dict4 = None
    self.dict4_single = None
    self.dict1_2_3_4 = None
    self.text_ipa = ""
    self.text_ipa_readable = ""

  def transcribe_to_ipa(self, text: str, skip_normalization: bool, skip_sentence_separation: bool) -> str:
    logger = get_logger()
    flogger = get_logger()
    self._reset_locals()

    if skip_normalization:
      flogger.info("Normalization was skipped.")
    else:
      text_normed = normalize_eng_text(text)
      if text_normed == text:
        flogger.info("No normalization applied.")
      else:
        self.text_normed = text_normed
        text = text_normed
        flogger.info("Normalization was applied.")

    if skip_sentence_separation:
      flogger.info("Sentence separation was skipped.")
    else:
      sentences = get_sentences(text)
      text_sentenced = "\n".join(sentences)
      if text == text_sentenced:
        flogger.info("No sentence separation applied.")
      else:
        self.text_sentenced = text_sentenced
        text = text_sentenced
        flogger.info("Sentence separation was applied.")

    vocabulary = extract_vocabulary_from_text(
      text, "\n", " ", False, 1, None, 2_000_000)
    self.vocabulary = vocabulary

    dict1, oov1 = create_dict_from_dict(vocabulary, self._ljs_dict, trim={
    }, split_on_hyphen=False, ignore_case=False, n_jobs=1, maxtasksperchild=None, chunksize=10_000)

    self.dict1 = deepcopy(dict1)
    if len(oov1) > 0:
      self.oov1 = oov1

    changed_word_count = select_single_pronunciation(dict1, mode="highest-weight", seed=None,
                                                     mp_options=MultiprocessingOptions(1, None, 1_000))

    self.dict1_single = None
    if changed_word_count > 0:
      self.dict1_single = deepcopy(dict1)

    oov2 = OrderedSet()
    if len(oov1) > 0:
      dict2, oov2 = create_dict_from_dict(oov1, self._ljs_dict, trim=self._punctuation, split_on_hyphen=True,
                                          ignore_case=True, n_jobs=1, maxtasksperchild=None, chunksize=10_000)
      self.dict2 = deepcopy(dict2)
      if len(oov2) > 0:
        self.oov2 = oov2

      changed_word_count = select_single_pronunciation(dict2, mode="highest-weight", seed=None,
                                                       mp_options=MultiprocessingOptions(1, None, 1_000))
      if changed_word_count > 0:
        self.dict2_single = deepcopy(dict2)

      merge_dictionaries(dict1, dict2, mode="add")

      self.dict1_2 = deepcopy(dict1)

    oov3 = OrderedSet()
    if len(oov2) > 0:
      dict3, oov3 = create_dict_from_dict(oov2, self._cmu_dict, trim=self._punctuation, split_on_hyphen=True,
                                          ignore_case=True, n_jobs=1, maxtasksperchild=None, chunksize=10_000)
      self.dict3 = deepcopy(dict3)
      if len(oov3) > 0:
        self.oov3 = oov3

      changed_word_count = select_single_pronunciation(dict3, mode="highest-weight", seed=None,
                                                       mp_options=MultiprocessingOptions(1, None, 1_000))
      if changed_word_count > 0:
        self.dict3_single = deepcopy(dict3)

      merge_dictionaries(dict1, dict3, mode="add")
      self.dict1_2_3 = deepcopy(dict1)

    if len(oov3) > 0:
      dict4 = transcribe_with_g2pE(oov3, weight=1, trim=self._punctuation,
                                   split_on_hyphen=True, n_jobs=1, maxtasksperchild=None, chunksize=100_000)
      self.dict4_arpa = deepcopy(dict4)

      identify_and_apply_mappings(logger, flogger, dict4, ARPA_IPA_MAPPING, partial_mapping=False,
                                  mp_options=MultiprocessingOptions(1, None, 100_000))
      replace_symbols_in_pronunciations(dict4, "( |ˈ|ˌ)(ə|ʌ|ɔ|ɪ|ɛ|ʊ) r", r"\1\2r",
                                        False, None, MultiprocessingOptions(1, None, 100_000))
      self.dict4 = deepcopy(dict4)

      changed_word_count = select_single_pronunciation(dict4, mode="highest-weight", seed=None,
                                                       mp_options=MultiprocessingOptions(1, None, 1_000))
      if changed_word_count > 0:
        self.dict4_single = deepcopy(dict4)

      merge_dictionaries(dict1, dict4, mode="add")

      self.dict1_2_3_4 = deepcopy(dict1)

    text_ipa = transcribe_text_using_dict(dict1, text, "\n", self._symbol_separator, " ", seed=None, ignore_missing=False,
                                          n_jobs=1, maxtasksperchild=None, chunksize=2_000_000)
    self.text_ipa = text_ipa
    self.text_ipa_readable = text_ipa.replace(self._symbol_separator, "")

    text_ipa = replace_text(text_ipa, " ", "SIL0", disable_regex=True)
    text_ipa = replace_text(text_ipa, f",{self._symbol_separator}SIL0",
                            f",{self._symbol_separator}SIL1", disable_regex=True)
    text_ipa = replace_text(text_ipa, r"(\.|\!|\?)",
                            rf"\1{self._symbol_separator}SIL2", disable_regex=False)
    text_ipa = replace_text(
      text_ipa, rf"(;|:){re.escape(self._symbol_separator)}SIL0", rf"\1{self._symbol_separator}SIL2", disable_regex=False)

    return text_ipa


def get_sentences(text: str) -> Generator[str, None, None]:
  pattern = re.compile(r"(\.|\?|\!) +")  # [^$]
  sentences = pattern.split(text)
  for i in range(0, len(sentences), 2):
    if i + 1 < len(sentences):
      sentence = sentences[i] + sentences[i + 1]
    else:
      sentence = sentences[i]
    if len(sentence) > 0:
      # yield from get_non_empty_lines(sentence)
      yield sentence


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
      remove_urls,
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
      change_minus,
    )
  )

  return text


def change_minus(text: str) -> str:
  # "- -" to "—"
  text = text.replace(" -- ", " — ")
  text = text.replace(" - ", " — ")
  return text


def remove_urls(text: str) -> str:
  pattern = re.compile(
    r"(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])")
  result = pattern.sub("U R L", text)
  return result
