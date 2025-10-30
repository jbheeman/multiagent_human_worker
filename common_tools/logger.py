import os
import sys
import re
from datetime import datetime

class AnsiStrippingFileWrapper:
    def __init__(self, file):
        self.file = file

    def write(self, text):
        cleaned_text = self.remove_ansi_codes(text)
        self.file.write(cleaned_text)

    def remove_ansi_codes(self, text):
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def __getattr__(self, attr):
        return getattr(self.file, attr)

class Logger:
    def __init__(self, log_dir="logs"):
        self.run_dir = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        os.makedirs(self.run_dir, exist_ok=True)
        self.step_revision_counters = {}
        self.overview_log_file_path = os.path.join(self.run_dir, "overview.log")
        self.overview_log_file = AnsiStrippingFileWrapper(open(self.overview_log_file_path, "w"))

    def get_step_suffix(self, step_number):
        revision_count = self.step_revision_counters.get(step_number, 0)
        if revision_count == 0:
            return ""
        else:
            return chr(ord('a') + revision_count)

    def get_log_file_path(self, agent_name, step_number=None):
        if step_number is not None:
            suffix = self.get_step_suffix(step_number)
            log_file_name = f"step_{step_number}{suffix}_{agent_name}.log"
        else:
            log_file_name = f"{agent_name}.log"
        return os.path.join(self.run_dir, log_file_name)

    def get_log_file(self, agent_name, step_number=None):
        log_file_path = self.get_log_file_path(agent_name, step_number)
        return AnsiStrippingFileWrapper(open(log_file_path, "w"))

    def mark_as_redundant(self, agent_name, step_number):
        log_file_path = self.get_log_file_path(agent_name, step_number)
        redundant_log_file_path = log_file_path.replace(".log", "_redundant.log")
        os.rename(log_file_path, redundant_log_file_path)
        self.step_revision_counters[step_number] = self.step_revision_counters.get(step_number, 0) + 1

    def log_overview(self, message):
        print(message, file=sys.__stdout__)
        self.overview_log_file.write(message + "\n")
        self.overview_log_file.flush()
