"""STATE-Bench backend: in-process run_one_trajectory (no satisfaction critic)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from eval.mcts.api_errors import is_external_api_error
from eval.mcts.backends import RolloutRequest, RolloutResult
from eval.mcts.backends.base import artifact_path, to_fs_label
from eval.mcts.terminal import map_statebench_terminal


class StateBenchBackend:
    name = "statebench"

    def __init__(self, repo_root: Path, satisfaction_model: str = "qwen3-small") -> None:
        self.repo_root = repo_root.resolve()
        self.satisfaction_model = satisfaction_model
        self._agent_clients: dict[str, Any] = {}
        self._sim_clients: dict[str, Any] = {}
        self._satisfaction_clients: dict[str, Any] = {}
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
        try:
            from state_bench.agents.loader import load_root_agent_class, load_root_client_class
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", None) or str(exc)
            raise SystemExit(
                "STATE-Bench import failed "
                f"({missing}). Run MCTS statebench via its uv env, e.g.\n"
                "  cd ~/multiagent_human_worker\n"
                "  uv run --project STATE-Bench python eval/mcts/run_mcts.py "
                "--bench statebench ... --no-satisfaction --confirm\n"
                "Do not use the repo-root .venv for --bench statebench "
                "(it lacks azure-identity / STATE-Bench deps)."
            ) from exc

        self._client_class = load_root_client_class("NautilusClient", root=self.repo_root)
        self._agent_class = load_root_agent_class("NautilusAgent", root=self.repo_root)

    def _get_clients(self, model: str, sim_model: str, *, with_satisfaction: bool):
        self._ensure_imports()
        if model not in self._agent_clients:
            self._agent_clients[model] = self._client_class.from_env(model=model, role="agent")
        if sim_model not in self._sim_clients:
            self._sim_clients[sim_model] = self._client_class.from_env(model=sim_model)
        satisfaction_client = None
        if with_satisfaction:
            sat = self.satisfaction_model
            if sat not in self._satisfaction_clients:
                self._satisfaction_clients[sat] = self._client_class.from_env(model=sat)
            satisfaction_client = self._satisfaction_clients[sat]
        return self._agent_clients[model], self._sim_clients[sim_model], satisfaction_client

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
        dest_path = artifact_path(
            req.eval_dir,
            model=req.model,
            domain=req.domain,
            arm=req.arm,
            persona_key=persona_key,
            task_id=req.task_id,
        )
        model_dir = dest_path.parent

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

        agent_client, sim_client, satisfaction_client = self._get_clients(
            req.model,
            req.sim_model,
            with_satisfaction=not req.no_satisfaction,
        )
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
                satisfaction_client=satisfaction_client,
                sim_model=req.sim_model,
                agent_model=req.model,
            )
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            if is_external_api_error(err):
                # Do not persist — leave slot free for --skip-existing rerun.
                return RolloutResult(
                    terminal_state="sim_error",
                    task_success=False,
                    transfer=False,
                    n_turns=0,
                    full_transcript=[],
                    artifact_path=None,
                    error=err,
                    raw={"sim_seed_unsupported": True, "skipped_persist": True},
                )
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

        if is_external_api_error(traj.error):
            return RolloutResult(
                terminal_state="sim_error",
                task_success=False,
                transfer=False,
                n_turns=0,
                full_transcript=[],
                artifact_path=None,
                error=traj.error,
                raw={"sim_seed_unsupported": True, "skipped_persist": True},
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
