import logging
import re
from collections import OrderedDict
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Generator, Optional

from dict_from_dict import create_dict_from_dict
from english_text_normalization import (add_space_around_dashes, execute_pipeline,
                                        expand_abbreviations, expand_and_a_half, geo_to_george,
                                        normalize_all_units, normalize_am_and_pm,
                                        normalize_degrees_minutes_and_seconds,
                                        normalize_emails_and_at,
                                        normalize_king_name_followed_by_roman_numeral,
                                        normalize_numbers, normalize_percent,
                                        normalize_pounds_shillings_and_pence,
                                        normalize_second_and_third_when_abbr_with_d,
                                        normalize_temperatures_general,
                                        normalize_three_and_four_dots,
                                        normalize_today_tomorrow_and_tonight,
                                        remove_dot_after_single_capital_letters,
                                        remove_double_hyphen_before_or_after_colon,
                                        remove_equal_sign, remove_sic, remove_stars,
                                        remove_underscore_characters,
                                        remove_whitespace_before_sentence_punctuation,
                                        replace_and_sign_with_word_and, replace_eg_with_for_example,
                                        replace_etc_with_et_cetera, replace_four_hyphens_by_two,
                                        replace_hyphen_between_numbers_with_to,
                                        replace_ie_with_that_is, replace_no_with_number,
                                        replace_nos_with_numbers, strip,
                                        write_out_month_abbreviations)
from ordered_set import OrderedSet
from pronunciation_dictionary import MultiprocessingOptions, PronunciationDict
from pronunciation_dictionary_utils import (map_symbols_dict, merge_dictionaries,
                                            replace_symbols_in_pronunciations,
                                            select_single_pronunciation)
from txt_utils import extract_vocabulary_from_text, replace_text, transcribe_text_using_dict
from unidecode import unidecode_expect_ascii

from en_tts.globals import DEFAULT_CONF_DIR
from en_tts.resources import download_nltk_data, get_cmu_dict, get_ljs_dict

ARPA_IPA_MAPPING = {
  "AO0": "ɔ",
  "AO1": "ˈɔ",
  "AO2": "ˌɔ",
  "EY0": "eɪ",
  "EY1": "ˈeɪ",
  "EY2": "ˌeɪ",
  "UW0": "u",
  "UW1": "ˈu",
  "UW2": "ˌu",
  "ER0": "ər",  # !,
  "ER1": "ˈʌr",  # !,
  "ER2": "ˌʌr",  # !,
  "IH0": "ɪ",
  "IH1": "ˈɪ",
  "IH2": "ˌɪ",
  "EH0": "ɛ",
  "EH1": "ˈɛ",
  "EH2": "ˌɛ",
  "IY0": "i",
  "IY1": "ˈi",
  "IY2": "ˌi",
  "AA0": "ɑ",
  "AA1": "ˈɑ",
  "AA2": "ˌɑ",
  "AE0": "æ",
  "AE1": "ˈæ",
  "AE2": "ˌæ",
  "OW0": "oʊ",
  "OW1": "ˈoʊ",
  "OW2": "ˌoʊ",
  "AY0": "aɪ",
  "AY1": "ˈaɪ",
  "AY2": "ˌaɪ",
  "AW0": "aʊ",
  "AW1": "ˈaʊ",
  "AW2": "ˌaʊ",
  "AH0": "ə",  # !,
  "AH1": "ˈʌ",
  "AH2": "ˌʌ",
  "OY0": "ɔɪ",
  "OY1": "ˈɔɪ",
  "OY2": "ˌɔɪ",
  "UH0": "ʊ",
  "UH1": "ˈʊ",
  "UH2": "ˌʊ",
  "SH": "ʃ",
  "B": "b",
  "D": "d",
  "M": "m",
  "Q": "ʔ",
  "L": "l",
  "K": "k",
  "R": "r",  # ɹ,
  "CH": "tʃ",
  "NG": "ŋ",
  "P": "p",
  "Z": "z",
  "ZH": "ʒ",
  "JH": "dʒ",
  "T": "t",
  "G": "ɡ",
  "N": "n",
  "F": "f",
  "TH": "θ",
  "V": "v",
  "HH": "h",
  "DH": "ð",
  "S": "s",
  "W": "w",
  "Y": "j",
}


class Transcriber():
  def __init__(
      self,
      conf_dir: Path = DEFAULT_CONF_DIR,
  ) -> None:
    logger = getLogger(__name__)
    tmp_logger = getLogger("english_text_normalization")
    tmp_logger.parent = logger
    tmp_logger.setLevel(logging.WARNING)

    tmp_logger = getLogger("dict_from_dict")
    tmp_logger.parent = logger
    tmp_logger.setLevel(logging.WARNING)

    tmp_logger = getLogger("dict_from_g2pE")
    tmp_logger.parent = logger
    tmp_logger.setLevel(logging.WARNING)

    tmp_logger = getLogger("pronunciation_dictionary")
    tmp_logger.parent = logger
    tmp_logger.setLevel(logging.WARNING)

    tmp_logger = getLogger("txt_utils")
    tmp_logger.parent = logger
    tmp_logger.setLevel(logging.WARNING)

    tmp_logger = getLogger("pronunciation_dictionary_utils")
    tmp_logger.parent = logger
    tmp_logger.setLevel(logging.WARNING)

    download_nltk_data()

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

  def transcribe_to_ipa(self, text: str, skip_normalization: bool = False, skip_sentence_separation: bool = False) -> str:
    logger = getLogger(__name__)
    self._reset_locals()

    if skip_normalization:
      logger.debug("Normalization was skipped.")
    else:
      logger.info("Normalizing ...")
      text_normed = normalize_eng_text(text)
      if text_normed == text:
        logger.debug("No normalization applied.")
      else:
        self.text_normed = text_normed
        text = text_normed
        logger.debug("Normalization was applied.")

    if skip_sentence_separation:
      logger.debug("Sentence separation was skipped.")
    else:
      logger.info("Separating sentences ...")
      sentences = get_sentences(text)
      text_sentenced = "\n".join(sentences)
      if text == text_sentenced:
        logger.debug("No sentence separation applied.")
      else:
        self.text_sentenced = text_sentenced
        text = text_sentenced
        logger.debug("Sentence separation was applied.")

    logger.debug("Extracting vocabulary ...")
    vocabulary = extract_vocabulary_from_text(text, n_jobs=1, chunksize=2_000_000, silent=True)
    self.vocabulary = vocabulary

    logger.info("Looking up vocabulary ...")
    dict1, oov1 = create_dict_from_dict(vocabulary, self._ljs_dict, trim={
    }, split_on_hyphen=False, ignore_case=False, n_jobs=1, maxtasksperchild=None, chunksize=10_000, silent=True)

    self.dict1 = deepcopy(dict1)
    if len(oov1) > 0:
      self.oov1 = oov1

    changed_word_count = select_single_pronunciation(dict1, mode="highest-weight", seed=None,
                                                     mp_options=MultiprocessingOptions(1, None, 1_000), silent=True)

    self.dict1_single = None
    if changed_word_count > 0:
      self.dict1_single = deepcopy(dict1)

    oov2: OrderedSet[str] = OrderedSet()
    if len(oov1) > 0:
      dict2, oov2 = create_dict_from_dict(oov1, self._ljs_dict, trim=self._punctuation, split_on_hyphen=True,
                                          ignore_case=True, n_jobs=1, maxtasksperchild=None, chunksize=10_000, silent=True)
      self.dict2 = deepcopy(dict2)
      if len(oov2) > 0:
        self.oov2 = oov2

      changed_word_count = select_single_pronunciation(dict2, mode="highest-weight", seed=None,
                                                       mp_options=MultiprocessingOptions(1, None, 1_000), silent=True)
      if changed_word_count > 0:
        self.dict2_single = deepcopy(dict2)

      merge_dictionaries(dict1, dict2, mode="add")

      self.dict1_2 = deepcopy(dict1)

    oov3: OrderedSet[str] = OrderedSet()
    if len(oov2) > 0:
      dict3, oov3 = create_dict_from_dict(oov2, self._cmu_dict, trim=self._punctuation, split_on_hyphen=True,
                                          ignore_case=True, n_jobs=1, maxtasksperchild=None, chunksize=10_000, silent=True)
      self.dict3 = deepcopy(dict3)
      if len(oov3) > 0:
        self.oov3 = oov3

      changed_word_count = select_single_pronunciation(dict3, mode="highest-weight", seed=None,
                                                       mp_options=MultiprocessingOptions(1, None, 1_000), silent=True)
      if changed_word_count > 0:
        self.dict3_single = deepcopy(dict3)

      merge_dictionaries(dict1, dict3, mode="add")
      self.dict1_2_3 = deepcopy(dict1)

    if len(oov3) > 0:
      from dict_from_g2pE import transcribe_with_g2pE
      dict4 = transcribe_with_g2pE(oov3, weight=1, trim=self._punctuation,
                                   split_on_hyphen=True, n_jobs=1, maxtasksperchild=None, chunksize=100_000, silent=True)
      self.dict4_arpa = deepcopy(dict4)

      map_symbols_dict(dict4, ARPA_IPA_MAPPING, partial_mapping=False,
                       mp_options=MultiprocessingOptions(1, None, 100_000), silent=True)
      replace_symbols_in_pronunciations(dict4, "( |ˈ|ˌ)(ə|ʌ|ɔ|ɪ|ɛ|ʊ) r", r"\1\2r",
                                        False, None, MultiprocessingOptions(1, None, 100_000), silent=True)
      self.dict4 = deepcopy(dict4)

      changed_word_count = select_single_pronunciation(dict4, mode="highest-weight", seed=None,
                                                       mp_options=MultiprocessingOptions(1, None, 1_000), silent=True)
      if changed_word_count > 0:
        self.dict4_single = deepcopy(dict4)

      merge_dictionaries(dict1, dict4, mode="add")

      self.dict1_2_3_4 = deepcopy(dict1)

    logger.debug("Transcribing to IPA ...")
    text_ipa = transcribe_text_using_dict(
      text, dict1,
      phoneme_sep=self._symbol_separator, n_jobs=1, chunksize=2_000_000, silent=True,
    )
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
