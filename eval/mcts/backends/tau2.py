"""Tau2 backend: subprocess `tau2 run` mirroring run_all_personas_unique.py."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from eval.mcts.backends.base import to_fs_label
from eval.mcts.backends import RolloutRequest, RolloutResult
from eval.mcts.terminal import map_tau2_terminal


class Tau2Backend:
    name = "tau2"

    def __init__(
        self,
        repo_root: Path,
        simulations_dir: Path | None = None,
        user_llm: str | None = None,
        max_retries: int = 9,
        task_split_name: str = "base",
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.simulations_dir = (simulations_dir or (self.repo_root / "data" / "simulations")).resolve()
        self.user_llm = user_llm  # overridden per-request by sim_model if set
        self.max_retries = max_retries
        self.task_split_name = task_split_name

    def run(self, req: RolloutRequest) -> RolloutResult:
        agent_llm = f"openai/{req.model}" if "/" not in req.model else req.model
        model_label = to_fs_label(req.model)
        # Always key artifacts by persona_id so fixed_prompt paired rows do not collide.
        persona_key = to_fs_label(req.persona_id) if req.persona_yaml_path is not None else f"baseline__{to_fs_label(req.persona_id)}"
        # Include task + arm + block so artifacts don't collide across the matrix.
        save_to = (
            f"mcts_{model_label}_{persona_key}_{req.arm}_{req.domain}_"
            f"b{req.block}_{req.task_id}"
        )
        # tau2 --save-to writes data/simulations/<save_to>.json
        run_save_path = self.simulations_dir / f"{save_to}.json"
        model_dir = req.eval_dir / model_label / req.arm
        dest_path = model_dir / f"{persona_key}_{model_label}_{req.domain}_{req.task_id}.json"

        user_llm = req.sim_model
        if user_llm and "/" not in user_llm:
            user_llm = f"openai/{user_llm}"
        if self.user_llm and not req.sim_model:
            user_llm = self.user_llm

        # Clamp seed to signed 32-bit for LLM APIs that reject huge ints.
        seed = int(req.sim_seed) % (2**31 - 1)

        cmd = [
            "tau2",
            "run",
            "--domain",
            req.domain,
            "--agent-llm",
            agent_llm,
            "--user-llm",
            user_llm,
            "--num-trials",
            "1",
            "--num-tasks",
            "1",
            "--task-ids",
            str(req.task_id),
            "--task-split-name",
            self.task_split_name,
            "--seed",
            str(seed),
            "--save-to",
            save_to,
        ]

        env = os.environ.copy()
        if req.persona_yaml_path is None:
            env["TAU2_NO_PERSONA"] = "1"
            env.pop("TAU2_PERSONA_FILE", None)
        else:
            env.pop("TAU2_NO_PERSONA", None)
            env["TAU2_PERSONA_FILE"] = str(req.persona_yaml_path.resolve())

        if req.dry_run:
            print(f"[tau2 dry-run] TAU2_PERSONA_FILE={env.get('TAU2_PERSONA_FILE', '')} TAU2_NO_PERSONA={env.get('TAU2_NO_PERSONA', '')}")
            print(f"[tau2 dry-run] {' '.join(cmd)}")
            print(f"[tau2 dry-run] -> {dest_path}")
            return RolloutResult(
                terminal_state="failure_no_transfer",
                task_success=False,
                transfer=False,
                n_turns=0,
                full_transcript=[],
                artifact_path=dest_path,
                raw={"dry_run": True, "cmd": cmd, "seed_clamped": seed},
            )

        model_dir.mkdir(parents=True, exist_ok=True)
        self.simulations_dir.mkdir(parents=True, exist_ok=True)

        exit_code = self._run_with_retry(cmd, env)
        if exit_code != 0:
            return RolloutResult(
                terminal_state="sim_error",
                task_success=False,
                transfer=False,
                n_turns=0,
                full_transcript=[],
                artifact_path=None,
                error=f"tau2 run exit={exit_code}",
            )

        if not run_save_path.exists():
            return RolloutResult(
                terminal_state="sim_error",
                task_success=False,
                transfer=False,
                n_turns=0,
                full_transcript=[],
                artifact_path=None,
                error=f"missing output {run_save_path}",
            )

        shutil.copy2(run_save_path, dest_path)
        return self._parse_artifact(dest_path)

    def _run_with_retry(self, cmd: list[str], env: dict) -> int:
        delay = 2.0
        stdin_input = "y\ny\n"
        for attempt in range(self.max_retries):
            result = subprocess.run(
                cmd,
                env=env,
                cwd=self.repo_root,
                input=stdin_input,
                text=True,
            )
            if result.returncode == 0:
                return 0
            if attempt == self.max_retries - 1:
                return result.returncode
            print(
                f"  tau2 attempt {attempt + 1}/{self.max_retries} failed "
                f"(exit {result.returncode}); retry in {delay:.1f}s",
                file=sys.stderr,
            )
            time.sleep(delay)
            delay = min(delay * 2.0, 120.0)
        return 1

    def _parse_artifact(self, path: Path) -> RolloutResult:
        data = json.loads(path.read_text(encoding="utf-8"))
        sims = data.get("simulations") or []
        if not sims:
            return RolloutResult(
                terminal_state="sim_error",
                task_success=False,
                transfer=False,
                n_turns=0,
                full_transcript=[],
                artifact_path=path,
                error="no simulations in artifact",
                raw=data,
            )
        sim = sims[0]
        reward_info = sim.get("reward_info") or {}
        reward = reward_info.get("reward")
        termination_reason = sim.get("termination_reason")
        messages = sim.get("messages") or []
        terminal, success, transfer = map_tau2_terminal(
            reward=reward,
            termination_reason=termination_reason,
            messages=messages,
        )
        return RolloutResult(
            terminal_state=terminal,
            task_success=success,
            transfer=transfer,
            n_turns=len(messages),
            full_transcript=messages,
            artifact_path=path,
            raw={
                "reward": reward,
                "termination_reason": termination_reason,
                "seed": sim.get("seed"),
            },
        )
