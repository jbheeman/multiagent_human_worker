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
    def __init__(self, run_dir, goal_id=None):
        self.run_dir = run_dir
        self.goal_id = goal_id
        os.makedirs(self.run_dir, exist_ok=True)
        
        self.step_revision_counters = {}
        self.stdout_log_file_path = os.path.join(self.run_dir, "stdout.log")
        # Use append mode for the shared stdout.log
        self.stdout_log_file = AnsiStrippingFileWrapper(open(self.stdout_log_file_path, "a"))
        
        self.goal_dir = None
        self.goal_overview_log_file = None

        if self.goal_id:
            self.goal_dir = os.path.join(self.run_dir, self.goal_id)
            os.makedirs(self.goal_dir, exist_ok=True)
            goal_overview_log_file_path = os.path.join(self.goal_dir, "overview.log")
            self.goal_overview_log_file = AnsiStrippingFileWrapper(open(goal_overview_log_file_path, "w"))

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
        
        if self.goal_dir:
            return os.path.join(self.goal_dir, log_file_name)
        return os.path.join(self.run_dir, log_file_name)

    def get_log_file(self, agent_name, step_number=None):
        log_file_path = self.get_log_file_path(agent_name, step_number)
        # Use write mode 'w' for individual agent logs
        return AnsiStrippingFileWrapper(open(log_file_path, "w"))

    def mark_as_redundant(self, agent_name, step_number):
        log_file_path = self.get_log_file_path(agent_name, step_number)
        redundant_log_file_path = log_file_path.replace(".log", "_redundant.log")
        os.rename(log_file_path, redundant_log_file_path)
        self.step_revision_counters[step_number] = self.step_revision_counters.get(step_number, 0) + 1

    def log_overview(self, message, to_stdout=False):
        if to_stdout:
            # Prepend goal_id if available and message is not empty
            if self.goal_id and message.strip():
                message = f"[{self.goal_id}] {message}"
            print(message, file=sys.__stdout__)
            # Only write to the main stdout.log if it's also going to console
            self.stdout_log_file.write(message + "\n")
            self.stdout_log_file.flush()
        
        # If it's a goal-specific logger, always write to the goal's overview.log
        if self.goal_overview_log_file:
            self.goal_overview_log_file.write(message + "\n")
            self.goal_overview_log_file.flush()
