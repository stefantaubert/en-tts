
from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from logging import Logger
from pathlib import Path
from typing import Any, Callable, List
from typing import OrderedDict as ODType
from typing import cast

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm


from en_tts_cli.argparse_helper import (parse_existing_directory,
                                                       parse_existing_file, parse_path,
                                                       parse_positive_integer)
from en_tts_cli.types import ExecutionResult
                                                       
def init_from_mel_batch_parser(parser: ArgumentParser) -> Callable[[str, str], None]:
  parser.description = "Command-line interface for synthesizing English texts into speech."
  parser.add_argument("input", type=str, metavar="INPUT",
                      help="text input")
  return synthesize_ns

def synthesize_ns(ns: Namespace, logger: Logger, flogger: Logger) -> ExecutionResult:
  
  return True
