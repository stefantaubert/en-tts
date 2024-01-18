import argparse
import platform
import shutil
import sys
from argparse import ArgumentParser
from importlib.metadata import version
from logging import getLogger
from pathlib import Path
from pkgutil import iter_modules
from tempfile import gettempdir
from time import perf_counter
from typing import Callable, Generator, List, Tuple

from en_tts import LOGGER_NAME as EN_TTS_LOGGER_NAME
from en_tts_cli.globals import get_conf_dir, get_work_dir
from en_tts_cli.logging_configuration import (configure_root_logger, get_file_logger, get_logger,
                                              init_main_logger, try_init_file_buffer_logger)
from en_tts_cli.main import init_synthesize_eng_parser, init_synthesize_ipa_parser

__APP_NAME = "en-tts"

__version__ = version(__APP_NAME)

INVOKE_HANDLER_VAR = "invoke_handler"


def formatter(prog):
  return argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=40)


def print_features():
  parsers = get_parsers()
  for command, description, method in parsers:
    print(f"- `{command}`: {description}")


def get_parsers() -> Generator[Tuple[str, str, Callable], None, None]:
  yield from (
    ("synthesize", "synthesize English texts", init_synthesize_eng_parser),
    ("synthesize-ipa", "synthesize English texts transcribed in IPA", init_synthesize_ipa_parser),
  )


def _init_parser():
  main_parser = ArgumentParser(
    formatter_class=formatter,
    description="Command-line interface for synthesizing English texts into speech.",
  )
  main_parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)
  subparsers = main_parser.add_subparsers(help="description")

  for command, description, method in get_parsers():
    method_parser = subparsers.add_parser(
      command, help=description, formatter_class=formatter)
    # init parser
    invoke_method = method(method_parser)
    method_parser.set_defaults(**{
      INVOKE_HANDLER_VAR: invoke_method,
    })

    logging_group = method_parser.add_argument_group("logging arguments")
    # logging_group.add_argument("--work-directory", type=parse_path, metavar="DIRECTORY",
    #                            help="path to write the log", default=Path(gettempdir()) / "en-tts")
    logging_group.add_argument("--debug", action="store_true",
                               help="include debugging information in log")

  return main_parser


def reset_work_dir():
  root_logger = getLogger()

  work_dir = get_work_dir()

  try:
    if work_dir.is_dir():
      root_logger.debug("Deleting working directory ...")
      shutil.rmtree(work_dir)
    root_logger.debug("Creating working directory ...")
    work_dir.mkdir(parents=False, exist_ok=False)
  except Exception as ex:
    root_logger.exception("Working directory couldn't be resetted!", exc_info=ex, stack_info=True)
    sys.exit(1)


def ensure_conf_dir_exists():
  conf_dir = get_conf_dir()
  if not conf_dir.is_dir():
    root_logger = getLogger()
    root_logger.debug("Creating configuration directory ...")
    conf_dir.mkdir(parents=False, exist_ok=False)


def parse_args(args: List[str]) -> None:
  configure_root_logger()
  root_logger = getLogger()

  local_debugging = debug_file_exists()
  if local_debugging:
    root_logger.debug(f"Received arguments: {str(args)}")

  parser = _init_parser()

  try:
    ns = parser.parse_args(args)
  except SystemExit as error:
    error_code = error.args[0]
    # -v -> 0; invalid arg -> 2
    sys.exit(error_code)

  if local_debugging:
    root_logger.debug(f"Parsed arguments: {str(ns)}")

  if not hasattr(ns, INVOKE_HANDLER_VAR):
    parser.print_help()
    sys.exit(0)

  invoke_handler: Callable[..., bool] = getattr(ns, INVOKE_HANDLER_VAR)
  delattr(ns, INVOKE_HANDLER_VAR)

  ensure_conf_dir_exists()
  reset_work_dir()

  work_dir = get_work_dir()
  logfile = work_dir / "output.log"
  log_to_file = try_init_file_buffer_logger(logfile, local_debugging or ns.debug, 1)
  if not log_to_file:
    root_logger.error("Logging to file is not possible.")
    sys.exit(1)

  init_main_logger()

  core_logger = getLogger(EN_TTS_LOGGER_NAME)
  logger = get_logger()
  core_logger.parent = logger
  flogger = get_file_logger()

  if not local_debugging:
    sys_version = sys.version.replace('\n', '')
    flogger.debug(f"CLI version: {__version__}")
    flogger.debug(f"Python version: {sys_version}")
    flogger.debug("Modules: %s", ', '.join(sorted(p.name for p in iter_modules())))

    my_system = platform.uname()
    flogger.debug(f"System: {my_system.system}")
    flogger.debug(f"Node Name: {my_system.node}")
    flogger.debug(f"Release: {my_system.release}")
    flogger.debug(f"Version: {my_system.version}")
    flogger.debug(f"Machine: {my_system.machine}")
    flogger.debug(f"Processor: {my_system.processor}")

  flogger.debug(f"Received arguments: {str(args)}")
  flogger.debug(f"Parsed arguments: {str(ns)}")

  start = perf_counter()

  try:
    success = invoke_handler(ns)
  except ValueError as error:
    logger = get_logger()
    logger.debug(error)
    success = False

  exit_code = 0
  if success:
    flogger.info("Everything was successful!")
  else:
    exit_code = 1
    # cmd_logger.error(f"Validation error: {success.default_message}")
    if log_to_file:
      root_logger.error("Not everything was successful! See log for details.")
    else:
      root_logger.error("Not everything was successful!")
    flogger.error("Not everything was successful!")

  duration = perf_counter() - start
  flogger.debug(f"Total duration (s): {duration}")
  if log_to_file:
    # path not encapsulated in "" because it is only console out
    root_logger.info(f"Log: \"{logfile.absolute()}\"")
    root_logger.info("Writing remaining buffered log lines...")

  sys.exit(exit_code)


def run():
  arguments = sys.argv[1:]
  parse_args(arguments)


def run_prod():
  run()


def debug_file_exists():
  return (Path(gettempdir()) / f"{__APP_NAME}-debug").is_file()


def create_debug_file():
  if not debug_file_exists():
    (Path(gettempdir()) / f"{__APP_NAME}-debug").write_text("", "UTF-8")


if __name__ == "__main__":
  run_prod()
