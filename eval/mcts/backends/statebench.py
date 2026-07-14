"""STATE-Bench backend: in-process run_one_trajectory (no satisfaction critic)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from eval.mcts.backends import RolloutRequest, RolloutResult
from eval.mcts.backends.base import to_fs_label
from eval.mcts.terminal import map_statebench_terminal


class StateBenchBackend:
    name = "statebench"

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self._agent_clients: dict[str, Any] = {}
        self._sim_clients: dict[str, Any] = {}
        self._agent_class = None
        self._client_class = None
        self._seed_gap_logged = False

    def _ensure_imports(self) -> None:
        if self._client_class is not None:
            return
        root = str(self.repo_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        from dotenv import load_dotenv

        load_dotenv(self.repo_root / ".env")
        from state_bench.agents.loader import load_root_agent_class, load_root_client_class

        self._client_class = load_root_client_class("NautilusClient")
        self._agent_class = load_root_agent_class("NautilusAgent")

    def _get_clients(self, model: str, sim_model: str):
        self._ensure_imports()
        if model not in self._agent_clients:
            self._agent_clients[model] = self._client_class.from_env(model=model)
        if sim_model not in self._sim_clients:
            self._sim_clients[sim_model] = self._client_class.from_env(model=sim_model)
        return self._agent_clients[model], self._sim_clients[sim_model]

    def run(self, req: RolloutRequest) -> RolloutResult:
        if not self._seed_gap_logged:
            print(
                "WARNING: STATE-Bench harness has no sim-seed knob; "
                "logging intended sim_seed in core JSONL only.",
                file=sys.stderr,
            )
            self._seed_gap_logged = True

        model_label = to_fs_label(req.model)
        persona_key = (
            to_fs_label(req.persona_id)
            if req.persona_yaml_path is not None
            else f"baseline__{to_fs_label(req.persona_id)}"
        )
        model_dir = req.eval_dir / model_label / req.arm
        dest_path = model_dir / f"{persona_key}_{model_label}_{req.domain}_{req.task_id}.json"

        if req.dry_run:
            print(f"[statebench dry-run] model={req.model} persona={persona_key} task={req.task_id}")
            print(f"[statebench dry-run] -> {dest_path}")
            return RolloutResult(
                terminal_state="failure_no_transfer",
                task_success=False,
                transfer=False,
                n_turns=0,
                full_transcript=[],
                artifact_path=dest_path,
                raw={"dry_run": True, "sim_seed_unsupported": True},
            )

        self._ensure_imports()
        from state_bench.paths import domain_tasks_dir
        from state_bench.schemas import TaskDefinition
        from state_bench.scripts.persona_injection import load_persona_yaml, persona_id_from_yaml
        from state_bench.scripts.run_unofficial import run_one_trajectory

        tasks_dir = domain_tasks_dir(req.domain)
        task_path = tasks_dir / f"{req.task_id}.json"
        if not task_path.exists():
            return RolloutResult(
                terminal_state="sim_error",
                task_success=False,
                transfer=False,
                n_turns=0,
                full_transcript=[],
                error=f"task not found: {task_path}",
            )
        task = TaskDefinition.load(task_path)
        if not task.user_id:
            return RolloutResult(
                terminal_state="sim_error",
                task_success=False,
                transfer=False,
                n_turns=0,
                full_transcript=[],
                error=f"task {req.task_id} has no user_id",
            )

        if req.persona_yaml_path is None:
            persona_yaml = None
            persona_id = None
        else:
            persona_yaml = load_persona_yaml(req.persona_yaml_path)
            persona_id = persona_id_from_yaml(persona_yaml, fallback=persona_key)

        agent_client, sim_client = self._get_clients(req.model, req.sim_model)
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            traj = run_one_trajectory(
                domain_name=req.domain,
                task=task,
                persona_yaml=persona_yaml,
                persona_key=persona_key,
                persona_id=persona_id,
                persona_path=req.persona_yaml_path,
                agent_client=agent_client,
                sim_client=sim_client,
                agent_class=self._agent_class,
                satisfaction_client=None,  # Phase 1: never run critic
                sim_model=req.sim_model,
                agent_model=req.model,
            )
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            payload = {
                "task_id": req.task_id,
                "error": err,
                "terminal_state": "error",
                "conversation": [],
            }
            dest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return RolloutResult(
                terminal_state="sim_error",
                task_success=False,
                transfer=False,
                n_turns=0,
                full_transcript=[],
                artifact_path=dest_path,
                error=err,
                raw={"sim_seed_unsupported": True},
            )

        sim = traj.to_dict()
        sim["task_id"] = req.task_id
        envelope = {
            "info": {
                "model": req.model,
                "sim_model": req.sim_model,
                "domain": req.domain,
                "arm": req.arm,
                "persona_id": req.persona_id,
                "block": req.block,
                "intended_sim_seed": req.sim_seed,
            },
            "tasks": [{"task_id": req.task_id}],
            "simulations": [sim],
        }
        dest_path.write_text(json.dumps(envelope, indent=2, default=str), encoding="utf-8")

        raw_terminal = (traj.metadata or {}).get("terminal_state")
        met = traj.state_requirements_score.score if traj.state_requirements_score else None
        terminal, success, transfer = map_statebench_terminal(
            raw_terminal=raw_terminal,
            error=traj.error,
            state_requirements_met=met,
        )
        conversation = sim.get("conversation") or getattr(traj, "conversation", []) or []
        return RolloutResult(
            terminal_state=terminal,
            task_success=success,
            transfer=transfer,
            n_turns=len(conversation),
            full_transcript=conversation,
            artifact_path=dest_path,
            error=traj.error,
            raw={
                "statebench_terminal": raw_terminal,
                "state_requirements_met": met,
                "sim_seed_unsupported": True,
            },
        )
