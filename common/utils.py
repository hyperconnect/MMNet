import sys
import humanfriendly as hf
import contextlib
import argparse
import logging
import getpass
import shutil
import json
import time
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime

import tensorflow as tf
import click
from termcolor import colored


def check_external_paths(paths, log):
    for p in paths:
        if not Path(p).exists():
            log.info(colored("{} path does not exist!".format(p), "red"))
            time.sleep(1)


def get_model_variables(exclude_prefix):
    if exclude_prefix == [""]:
        return tf.contrib.framework.get_model_variables()
    assert "" not in exclude_prefix

    model_variables = []
    for var in tf.contrib.framework.get_model_variables():
        is_include = True
        for substring in exclude_prefix:
            if substring in var.name:
                is_include = False
        if is_include:
            model_variables.append(var)

    return model_variables


def dump_configuration(train_log_dir, config, filename="config.json"):
    if not Path(train_log_dir).exists():
        Path(train_log_dir).mkdir(parents=True)

    if isinstance(config, argparse.Namespace):
        config = vars(config)
    elif isinstance(config, dict):
        config = config
    else:
        raise ValueError("Unsupported type for configuration: {}".format(type(config)))

    with Path(train_log_dir, filename).open("w") as f:
        json.dump(config, f)


def update_train_dir(args):
    def replace_func(base_string, a, b):
        replaced_string = base_string.replace(a, b)
        print(colored("[update_train_dir] replace {} : {} -> {}".format(a, base_string, replaced_string),
                      "yellow"))
        return replaced_string

    def make_placeholder(s: str, circumfix: str="%"):
        return circumfix + s.upper() + circumfix

    placeholder_mapping = {
        make_placeholder("DATE"): datetime.now().strftime("%y%m%d%H%M%S"),
        make_placeholder("USER"): getpass.getuser(),
    }

    for key, value in placeholder_mapping.items():
        args.train_dir = replace_func(args.train_dir, key, value)

    unknown = "UNKNOWN"
    for key, value in vars(args).items():
        key_placeholder = make_placeholder(key)
        if key_placeholder in args.train_dir:
            replace_value = value
            if isinstance(replace_value, str):
                if "/" in replace_value:
                    replace_value = unknown
            elif isinstance(replace_value, list):
                replace_value = ",".join(map(str, replace_value))
            elif isinstance(replace_value, float) or isinstance(replace_value, int):
                replace_value = str(replace_value)
            elif isinstance(replace_value, bool):
                replace_value = str(replace_value)
            else:
                replace_value = unknown
            args.train_dir = replace_func(args.train_dir, key_placeholder, replace_value)

    print(colored("[update_train_dir] final train_dir {}".format(args.train_dir),
                  "yellow", attrs=["bold", "underline"]))


def exit_handler(train_dir):
    if click.confirm("Do you want to delete {}?".format(train_dir), abort=True):
        shutil.rmtree(train_dir)
        print("... delete {} done!".format(train_dir))


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def get_subparser_argument_list(parser, subparser_name):
    # Hack argparse and get subparser's arguments
    from argparse import _SubParsersAction
    argument_list = []
    for sub_parser_action in filter(lambda x: isinstance(x, _SubParsersAction), parser._subparsers._actions):
        for action in sub_parser_action.choices[subparser_name]._actions:
            arg = action.option_strings[-1].replace("--", "")
            if arg == "help":
                continue
            if arg.startswith("no-"):
                continue
            argument_list.append(arg)
    return argument_list


def get_logger(logger_name, log_file: Path=None, level=logging.DEBUG):
    # "log/data-pipe-{}.log".format(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logger = logging.getLogger(logger_name)

    if not logger.hasHandlers():
        formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s")

        logger.setLevel(level)

        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fileHandler = logging.FileHandler(log_file, mode="w")
            fileHandler.setFormatter(formatter)
            logger.addHandler(fileHandler)

        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        logger.addHandler(streamHandler)

    return logger


def format_timespan(duration):
    if duration < 10:
        readable_duration = "{:.1f} (ms)".format(duration * 1000)
    else:
        readable_duration = hf.format_timespan(duration)
    return readable_duration


@contextlib.contextmanager
def timer(name):
    st = time.time()
    yield
    print("<Timer> {} : {}".format(name, format_timespan(time.time() - st)))


def timeit(method):
    def timed(*args, **kw):
        hf_timer = hf.Timer()
        result = method(*args, **kw)
        print("<Timeit> {!r} ({!r}, {!r}) {}".format(method.__name__, args, kw, hf_timer.rounded))
        return result
    return timed


class Timer(object):
    def __init__(self, log):
        self.log = log

    @contextlib.contextmanager
    def __call__(self, name, log_func=None):
        """
        Example.
            timer = Timer(log)
            with timer("Some Routines"):
                routine1()
                routine2()
        """
        if log_func is None:
            log_func = self.log.info

        start = time.clock()
        yield
        end = time.clock()
        duration = end - start
        readable_duration = format_timespan(duration)
        log_func(f"{name} :: {readable_duration}")


class TextFormatter(object):
    def __init__(self, color, attrs):
        self.color = color
        self.attrs = attrs

    def __call__(self, string):
        return colored(string, self.color, attrs=self.attrs)


class LogFormatter(object):
    def __init__(self, log, color, attrs):
        self.log = log
        self.color = color
        self.attrs = attrs

    def __call__(self, string):
        return self.log(colored(string, self.color, attrs=self.attrs))


@contextlib.contextmanager
def format_text(color, attrs=None):
    yield TextFormatter(color, attrs)


def format_text_fun(color, attrs=None):
    return TextFormatter(color, attrs)


def format_log(log, color, attrs=None):
    return LogFormatter(log, color, attrs)


@contextlib.contextmanager
def smart_log(log, msg):
    log.info(f"{msg} started.")
    yield
    log.info(f"{msg} finished.")


def setup_step1_mode(args):
    args.step_evaluation = 1
    args.step_validation = 1
    args.step_minimum_save = 0
    args.step_save_checkpoint = 1
    args.step_save_summaries = 1

    log = get_logger("utils")
    log.info(colored("Update step_evaluation, step_validation, step_save_checkpoint ... to 1",
                     "yellow"))


def wait(message, stop_checker_closure):
    assert callable(stop_checker_closure)
    st = time.time()
    while True:
        try:
            time_pass = hf.format_timespan(int(time.time() - st))
            sys.stdout.write(colored((
                f"{message}. Do you wanna wait? If not, then ctrl+c! :: waiting time: {time_pass}\r"
            ), "yellow", attrs=["bold"]))
            sys.stdout.flush()
            time.sleep(1)
            if stop_checker_closure():
                break
        except KeyboardInterrupt:
            break


class MLNamespace(SimpleNamespace):
    def __init__(self, *args, **kwargs):
        for kwarg in kwargs.keys():
            assert kwarg not in dir(self)
        super().__init__(*args, **kwargs)

    def unordered_values(self):
        return list(vars(self).values())

    def __setitem__(self, key, value):
        setattr(self, key, value)
