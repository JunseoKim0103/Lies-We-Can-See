"""
Aria — Config-driven ablation agent for Among Us (MineLand).

Drop-in replacement for Steve/Alex with identical run() signature.
All experimental axes are controlled by AriaConfig.
Interrupt conditions are inherited from Steve (RUBRIC R2).
"""

from __future__ import annotations

import copy
import json
import os
import random
import time
from typing import Optional

from ..sim.data.action import Action
from .config import AriaConfig
from .env_adapter import AmongUsEnvAdapter, AmongUsGameState
from .logger import AriaLogger


# ──────────────────────────────────────────────────────────────────────
# Lazy imports to avoid circular dependencies
# ──────────────────────────────────────────────────────────────────────

def _create_llm(config: AriaConfig, max_tokens: int = None, temperature: float = None):
    """Create LLM instance."""
    from .llm import create_chat_openai
    _eff = "minimal" if getattr(config, "role", None) == "crewmate" else None  # RQ2 crew-off: crew non-reasoning
    return create_chat_openai(
        reasoning_effort=_eff,
        model_name=config.llm_model,
        max_tokens=max_tokens if max_tokens is not None else config.max_tokens,
        temperature=temperature if temperature is not None else config.temperature,
        response_format={"type": "json_object"},
        api_base=config.api_base,
    )


SAVE_INTERVAL_TICKS = 50


class Aria:
    """Config-driven Among Us agent with ablation support.

    Matches Steve.run() / Alex.run() signature for drop-in compatibility.
    """

    def __init__(self, config: Optional[AriaConfig] = None, **kwargs):
        if config is None:
            config = AriaConfig(**kwargs)
        self.config = config
        self.bot_name = config.bot_name
        self.role = config.role

        self.save_path = os.path.join(
            config.save_path,
            f"aria_{config.bot_name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
        )
        os.makedirs(self.save_path, exist_ok=True)

        # Route this agent's prompt-trace output (trace.json + images/) into
        # its own storage folder, alongside memory/state JSON.
        from .trace_logger import register_agent_trace_dir
        register_agent_trace_dir(self.bot_name, config.save_path)

        # ── Logger (always-on failure tracking) ──────────────────────
        self.logger = AriaLogger(bot_name=config.bot_name)

        # ── Environment adapter ──────────────────────────────────────
        self.env_adapter = AmongUsEnvAdapter(
            bot_name=config.bot_name,
            player_id_map=config.player_id_map,
        )

        # ── State builder ────────────────────────────────────────────
        from .state import create_state_builder
        self.state_builder = create_state_builder(config)

        # ── Memory ───────────────────────────────────────────────────
        from .memory import create_memory
        self.memory = create_memory(config)

        # ── Skill memory ─────────────────────────────────────────────
        self.skill_memory = None
        if config.use_skill_memory and config.reflection_mode != "none":
            from .skill_memory import SkillMemory
            self.skill_memory = SkillMemory(
                max_size=config.skill_memory_size,
                sources=config.skill_memory_sources,
            )

        # ── Planner ──────────────────────────────────────────────────
        from .planner import create_planner
        self.planner = create_planner(config)

        # ── Reflector ────────────────────────────────────────────────
        from .reflection import create_reflector
        self.reflector = create_reflector(config)

        # ── Inject LLMs where needed ──────────────────────────────────
        if config.planning_policy == "llm" and hasattr(self.planner, "set_llm"):
            llm_plan = _create_llm(config, max_tokens=256, temperature=0.3)
            self.planner.set_llm(llm_plan)
        if config.reflection_mode != "none" and hasattr(self.reflector, "set_llm"):
            llm_reflect = _create_llm(config, max_tokens=256, temperature=0.3)
            self.reflector.set_llm(llm_reflect)
        if hasattr(self.memory, "set_llm"):
            llm_mem = _create_llm(config, max_tokens=256, temperature=0.2)
            self.memory.set_llm(llm_mem)

        # ── Action modules ───────────────────────────────────────────
        llm = _create_llm(config)
        llm_meeting = _create_llm(config, max_tokens=256, temperature=0.7)
        llm_vote = _create_llm(config, max_tokens=256, temperature=0.2)

        from .modules.meeting_module import MeetingModule
        from .modules.vote_module import VoteModule
        from .modules.kill_module import KillModule
        from .modules.report_module import ReportModule
        from .modules.move_module import MoveModule
        from .modules.mission_module import MissionModule
        from .modules.emergency_module import EmergencyModule
        from .modules.surveillance_module import SurveillanceModule

        self.meeting_module = MeetingModule(
            llm=llm_meeting, bot_name=config.bot_name, role=config.role,
            prompt_style=config.prompt_style,
            personal_message=config.personal_message,
            all_players=config.all_players,
        )
        self.vote_module = VoteModule(
            llm=llm_vote, bot_name=config.bot_name, role=config.role,
            prompt_style=config.prompt_style,
            player_id_map=config.player_id_map or {},
            personal_message=config.personal_message,
            all_players=config.all_players,
            teammate=config.teammate_imposter,
        )
        self.kill_module = KillModule(
            bot_name=config.bot_name,
            detection_range=config.kill_detection_range,
            policy=config.module_policy,
            teammate=config.teammate_imposter,
            all_players=config.all_players,
            stalk_giveup_distance=config.stalk_giveup_distance,
            vlm_scan_distance=config.vlm_scan_distance,
            use_vision=config.use_vision,
        )
        self.kill_module.vlm_fallback_threshold = config.vlm_kill_fallback
        self.report_module = ReportModule(
            bot_name=config.bot_name,
            role=config.role,
            policy=config.module_policy,
            use_vision=config.use_vision,
        )
        self.emergency_module = EmergencyModule(
            bot_name=config.bot_name, role=config.role,
            policy=config.module_policy,
            use_vision=config.use_vision,
        )
        self.surveillance_module = SurveillanceModule(
            bot_name=config.bot_name, role=config.role,
            policy=config.module_policy,
            use_vision=config.use_vision,
        )
        if config.module_policy == "vlm":
            vlm_module = _create_llm(config, max_tokens=256, temperature=0.2)
            self.kill_module.set_vlm(vlm_module)
            self.report_module.set_vlm(vlm_module)
            self.emergency_module.set_vlm(vlm_module)
            self.surveillance_module.set_vlm(vlm_module)
        self.move_module = MoveModule(
            llm=llm, bot_name=config.bot_name,
            role=config.role, prompt_style=config.prompt_style,
            save_path=self.save_path,
            personal_message=config.personal_message,
            codegen_model=config.llm_model,
            codegen_max_tokens=1024,
            codegen_temperature=0.2,
            api_base=config.api_base,
        )
        self.mission_module = MissionModule(
            llm=llm, bot_name=config.bot_name, role=config.role,
            save_path=self.save_path,
            personal_message=config.personal_message,
            codegen_model=config.llm_model,
            codegen_max_tokens=1024,
            codegen_temperature=0.2,
            api_base=config.api_base,
            coords_lookup=config.mission_coords_lookup,
        )

        # ── Internal state ───────────────────────────────────────────
        self._last_save_tick = -1
        self._prev_phase = 0
        self._was_running = False
        self._last_mode = ""
        self._needs_body_scan = False
        self._prev_game_state: Optional[AmongUsGameState] = None
        self._vlm_kill_deferred: bool = False  # VLM module deferred → next step explore
        self._current_target: Optional[str] = None
        self._state_timeline: dict = {}  # env step -> state snapshot (per-step history)

        if config.load_memory:
            self.state_builder.load_from_json(
                os.path.join(self.save_path, "state.json")
            )
            self.memory.load_from_json(
                os.path.join(self.save_path, "memory.json")
            )

        print(f"[Aria:{self.bot_name}] Ready. {config.summary()}")

    # ──────────────────────────────────────────────────────────────────
    # Main interface (same signature as Steve.run / Alex.run)
    # ──────────────────────────────────────────────────────────────────

    def run(
        self,
        obs,
        code_info=None,
        done=None,
        task_info=None,
        verbose: bool = False,
    ) -> Optional[Action]:
        """Called every environment step. Returns Action or None."""

        if done:
            self._save_all()
            print(self.logger.summary())
            return None

        # ── Parse obs via env_adapter (RUBRIC R1) ────────────────────
        game_state = self.env_adapter.parse_obs(obs)

        # No-vision (text-only) ablation: strip the RGB screenshot so every
        # module's `if game_state.rgb_base64:` guard is False → no image is
        # attached to any LLM call. KILL/SURVEILLANCE use distance-based text
        # reasoning instead (see their use_vision paths).
        if not self.config.use_vision:
            game_state.rgb_base64 = None

        # Tag this agent's LLM calls for token accounting + prompt tracing.
        # The env step index is set by the runner (run_agent_reasoning); here we
        # only ensure the agent name is attributed even outside that runner.
        from .llm import set_current_agent
        set_current_agent(self.bot_name)

        # DEBUG: entity visibility check (temporary)
        if verbose:
            te = getattr(obs, "target_entities", None)
            if te is None and hasattr(obs, "get"):
                te = obs.get("target_entities")
            te_count = len(te) if te else 0
            player_names = [p.name for p in game_state.visible_players]
            if te_count > 0 or player_names:
                print(f"[Aria:{self.bot_name}] ENTITY_CHECK raw_te={te_count} "
                      f"visible={player_names} tick={game_state.tick}")


        if game_state.is_ghost:
            # Refresh state_builder even while a ghost so mission_status
            # reflects ghost-completed missions. Without this, the agent's
            # pending_missions() freezes at the death-time snapshot, and the
            # orchestrator's sync_global_mission_progress under-counts
            # progress by the number of ghost-completed missions (visible
            # to imposters as a stale "Crew mission progress: N/M").
            self.state_builder.update(game_state)
            if game_state.phase == 1:
                return Action(type=Action.NEW, code=AmongUsEnvAdapter.build_interrupt_js())
            if self.role == "imposter":
                return Action(type=Action.RESUME, code="")
            is_running = self._is_code_running(code_info)
            if is_running:
                return Action(type=Action.RESUME, code="")
            result = self.mission_module.run(
                obs, game_state, self.state_builder, self.memory, verbose,
                logger=self.logger,
            )
            code = result["code"] if result and result.get("code", "").strip() else ""
            if verbose and code:
                print(f"[Aria:{self.bot_name}] ghost → mission")
            self._last_mode = "MISSION"
            action_type = Action.NEW if code.strip() else Action.RESUME
            return Action(type=action_type, code=code)

        # ── Update state builder ─────────────────────────────────────
        self.state_builder.update(game_state)
        self._record_state_step()

        phase = game_state.phase
        in_meeting = phase == 1

        # ── INTERRUPT: corpse nearby (before is_running) — RUBRIC R2 ─
        if (
            phase == 0
            and self._last_mode not in ("REPORT",)
            and self.env_adapter.has_nearby_corpse(game_state)
        ):
            if self.role == "imposter" and self._last_mode == "KILL":
                if random.random() >= self.config.self_report_probability:
                    if verbose:
                        print(f"[Aria:{self.bot_name}] post-kill → flee")
                    self._last_mode = "EXPLORE"
                    if game_state.own_position:
                        code = AmongUsEnvAdapter.build_flee_js(
                            game_state.own_position, role=self.role,
                        )
                    else:
                        code = ""
                    self._maybe_save()
                    return Action(type=Action.NEW, code=code)

            if verbose:
                print(f"[Aria:{self.bot_name}] corpse detected → REPORT")
            self._needs_body_scan = False
            self._last_mode = "REPORT"
            result = self.report_module.run(game_state, self.state_builder, self.memory, verbose)
            self._maybe_save()
            return Action(type=Action.NEW, code=result["code"] if result else "")

        # ── INTERRUPT: meeting start (before is_running) — RUBRIC R2 ─
        phase_just_entered = phase == 1 and self._prev_phase != 1
        phase_just_left = phase == 0 and self._prev_phase == 1
        self._prev_phase = phase


        if phase_just_entered:
            # Tell SemanticMemory to defer auto belief-update to post-reflection
            if hasattr(self.memory, '_meeting_belief_pending'):
                self.memory._meeting_belief_pending = True
            # Flush events to memory BEFORE returning, otherwise meeting_start
            # event (emitted on phase 0→1 transition) is lost. The standard
            # memory.update at line ~500 is unreachable on this code path.
            self.memory.update(
                tick=game_state.tick,
                action_summary="",
                state_summary=f"phase={phase}",
                events=[
                    {"tick": game_state.tick,
                     "message": getattr(e, "message", str(e)),
                     "type": getattr(e, "type", "")}
                    for e in (game_state.events or [])
                ] or None,
            )
            # Fire meeting-start reflection before early return
            if self.reflector.should_reflect(game_state, self._prev_game_state, False):
                if verbose:
                    print(f"[Aria:{self.config.bot_name}] REFLECTION triggered (meeting-start)")
                state_ctx = self.state_builder.summarize_for_context("reflection")
                memory_ctx = self.memory.get_context("reflection")
                refl = self.reflector.reflect(
                    state_ctx, memory_ctx, game_state, self.role, self.config.prompt_style,
                )
                if refl.insights:
                    self.memory.add_reflection(refl.insights)
                    if verbose:
                        print(f"[Aria:{self.config.bot_name}] REFLECTION insight: {refl.insights[:100]}")
            if verbose:
                print(f"[Aria:{self.bot_name}] Phase→Meeting: INTERRUPT")
            self._last_mode = "INTERRUPT"
            self._maybe_save()
            return Action(type=Action.NEW, code=AmongUsEnvAdapter.build_interrupt_js())

        # ── Code running check ───────────────────────────────────────
        is_running = self._is_code_running(code_info)
        if is_running:
            self._was_running = True
            if in_meeting and self._last_mode not in ("INTERRUPT", "MEETING", "VOTE"):
                self._last_mode = "INTERRUPT"
                return Action(type=Action.NEW, code=AmongUsEnvAdapter.build_interrupt_js())
            return Action(type=Action.RESUME, code="")

        # ── Code just completed → body scan ──────────────────────────
        # action_completed: True whenever Aria.run() is invoked AFTER having
        # already dispatched at least one real action. The _was_running disjunct
        # is dead code in this script's main loop (it processes resume internally
        # so is_running is always False on entry, see amongus_*.py:553-568) but
        # is kept for robustness against alternative loops. The _last_mode
        # disjunct is what actually fires post_action reflection — it stays
        # False for the very first step (_last_mode="") and for INTERRUPT
        # steps (which abort an action, not complete one).
        action_completed = self._was_running or self._last_mode not in ("", "INTERRUPT")
        if self._last_mode not in ("REPORT", "INTERRUPT", ""):
            self._needs_body_scan = True
        self._was_running = False

        # Inline body scan (Steve commit 3 pattern)
        if (
            self._needs_body_scan
            and phase == 0
            and not (self.role == "imposter" and self._last_mode == "KILL")
        ):
            if self.env_adapter.has_nearby_corpse(game_state):
                self._needs_body_scan = False
                self._last_mode = "REPORT"
                result = self.report_module.run(game_state, self.state_builder, self.memory, verbose)
                self._maybe_save()
                return Action(type=Action.NEW, code=result["code"] if result else "")
            else:
                self._needs_body_scan = False

        # ── Reflection (if triggered) ────────────────────────────────
        if self.reflector.should_reflect(game_state, self._prev_game_state, action_completed):
            if verbose:
                print(f"[Aria:{self.config.bot_name}] REFLECTION triggered (mode={self.config.reflection_mode})")
            state_ctx = self.state_builder.summarize_for_context("reflection")
            memory_ctx = self.memory.get_context("reflection")
            reflection_result = self.reflector.reflect(
                state_ctx, memory_ctx, game_state, self.role, self.config.prompt_style,
            )
            if reflection_result.insights:
                self.memory.add_reflection(reflection_result.insights)
                if verbose:
                    print(f"[Aria:{self.config.bot_name}] REFLECTION insight: {reflection_result.insights[:100]}")
            for player, delta in reflection_result.belief_updates.items():
                self.state_builder.record_suspicion(player, delta)
            # Update SemanticMemory beliefs
            if hasattr(self.memory, 'player_beliefs') and hasattr(self.memory, 'belief_update'):
                if self.memory.belief_update == "llm" and self.memory._chain is not None:
                    # LLM mode: feed full meeting chat → LLM belief chain
                    chat = self.state_builder.meeting_log or self.state_builder.last_meeting_chat
                    if chat:
                        self.memory._pending_events.extend(
                            [{"tick": game_state.tick, "message": m, "type": "chat"} for m in chat]
                        )
                        self.memory._llm_update_beliefs()
                    self.memory._meeting_belief_pending = False
                elif self.memory.belief_update == "rule":
                    # Rule mode: sync reflector's deltas into player_beliefs
                    for player, delta in reflection_result.belief_updates.items():
                        if player not in self.memory.player_beliefs:
                            self.memory.player_beliefs[player] = {"suspicion": 0.0, "notes": []}
                        cur = self.memory.player_beliefs[player].get("suspicion", 0.0)
                        self.memory.player_beliefs[player]["suspicion"] = max(0.0, min(1.0, cur + delta))
            # Sync meeting summary to SemanticMemory
            if hasattr(self.memory, 'meeting_summaries'):
                sb_summaries = self.state_builder.meeting_summaries
                if sb_summaries and len(sb_summaries) > len(self.memory.meeting_summaries):
                    self.memory.meeting_summaries = list(sb_summaries)
            if self.skill_memory and reflection_result.skill_candidates:
                from .skill_memory import SkillEntry
                for sc in reflection_result.skill_candidates:
                    self.skill_memory.add(SkillEntry(**sc))
                    if verbose:
                        print(f"[Aria:{self.config.bot_name}] SKILL_MEMORY saved: {sc.get('behavior', '')[:80]}")

        # ── Surveillance: multi-step scan (crewmate only, triggered after action completion) ──
        # Imposter has kill module's own scan; surveillance is for crewmate observation.
        # If scan is in progress, continue it (takes priority over planning)
        if self.surveillance_module.is_scanning and self.role == "crewmate":
            scan_result = self.surveillance_module.step(
                game_state, self.state_builder, self.memory, verbose,
            )
            if scan_result and scan_result.get("code", "").strip():
                self._last_mode = "SURVEILLANCE"
                return Action(type=Action.NEW, code=scan_result["code"])
            # scan_result is None → scan just completed, fall through to planning
            # Check emergency only if surveillance found something notable
            observation = self.surveillance_module.get_last_observation()
            if observation and game_state.phase == 0:
                emergency_result = self.emergency_module.run(
                    game_state, self.state_builder, self.memory, verbose,
                )
                if emergency_result and emergency_result.get("code", "").strip():
                    if verbose:
                        print(f"[Aria:{self.bot_name}] EMERGENCY MEETING CALLED!")
                    self._last_mode = "EMERGENCY"
                    self._maybe_save()
                    return Action(type=Action.NEW, code=emergency_result["code"])

        # Start surveillance scan after action completion (new position).
        # Crewmate only — imposter uses kill module's own scan.
        prev_was_action = self._last_mode in (
            "EXPLORE", "MISSION", "KILL",
        )
        if (
            self.role == "crewmate"
            and prev_was_action
            and phase == 0
            and not is_running
            and not self.surveillance_module.is_scanning
        ):
            self.surveillance_module.start_scan()
            scan_result = self.surveillance_module.step(
                game_state, self.state_builder, self.memory, verbose,
            )
            if scan_result and scan_result.get("code", "").strip():
                self._last_mode = "SURVEILLANCE"
                return Action(type=Action.NEW, code=scan_result["code"])

        # ── Planning ─────────────────────────────────────────────────
        state_ctx = self.state_builder.summarize_for_context("planning")
        memory_ctx = self.memory.get_context("planning")
        skill_ctx = self.skill_memory.get_context() if self.skill_memory else ""

        if verbose:
            for line in state_ctx.split("\n"):
                if "mission progress" in line.lower() or "WARNING" in line:
                    print(f"[Aria:{self.bot_name}] {line.strip()}")
            if skill_ctx:
                print(f"[Aria:{self.bot_name}] SKILL_CTX: {skill_ctx[:120]}")

        plan = self.planner.plan(
            state_context=state_ctx,
            memory_context=memory_ctx,
            skill_context=skill_ctx,
            game_state=game_state,
            state_builder=self.state_builder,
            role=self.role,
            prompt_style=self.config.prompt_style,
        )

        # VLM kill deferred last step → override kill with explore (go find a target)
        if self._vlm_kill_deferred and plan.mode == "kill":
            plan = plan.__class__(
                mode="explore",
                reasoning="VLM deferred kill (no target) → exploring to find players",
                short_term_plan="Move around to find an isolated target",
            )
            self._vlm_kill_deferred = False
            if verbose:
                print(f"[Aria:{self.bot_name}] VLM defer override → explore")

        if verbose:
            print(f"[Aria:{self.bot_name}] plan: mode={plan.mode} | {plan.reasoning}")
            # PLAN_LINE: smoke-test marker for fix #3 (planning_mode ablation).
            # Shows what the planner produced in fields and what state_builder
            # will inject into module prompts via summarize_for_context.
            _lt = plan.long_term_plan if plan.long_term_plan else ""
            _st = plan.short_term_plan if plan.short_term_plan else ""
            print(
                f"[Aria:{self.bot_name}] PLAN_LINE mode={plan.mode} "
                f"long_term={_lt!r} short_term={_st!r}"
            )

        # ── doing_mission: skip mission codegen (already waiting at button) ──
        # Only honor doing_mission during gameplay (phase==0), not during meeting
        if game_state.doing_mission and plan.mode == "mission" and game_state.phase == 0:
            if verbose:
                print(f"[Aria:{self.bot_name}] doing_mission → waiting (skip codegen)")
            self._last_mode = "MISSION"
            self._maybe_save()
            return Action(type=Action.RESUME, code="")

        # ── Agent-level target management ──────────────────────────
        _teammate = getattr(self.kill_module, 'teammate', None)
        if plan.target and plan.target != _teammate:
            self._current_target = plan.target
        if self._current_target:
            alive = self.state_builder.alive_player_names()
            if self._current_target not in alive:
                self._current_target = None
        if plan.mode in ("kill", "explore") and not plan.target and self._current_target:
            plan.target = self._current_target

        # ── Dispatch to module ───────────────────────────────────────
        # Inject plan into state_builder so module prompts (which all consume
        # state_builder.summarize_for_context) see short/long-term plan text.
        self.state_builder.set_current_plan(plan)
        result = self._dispatch(plan.mode, game_state, obs, verbose, target=plan.target, plan=plan)

        # Update agent target from module result — but only if it is a real
        # living player name. KillModule returns sentinel strings like
        # "nearest", "scanning", "none", "nearest_player" in its metadata,
        # which would otherwise pollute _current_target on the next step.
        meta_target = result.get("metadata", {}).get("target") if result else None
        if meta_target:
            alive = self.state_builder.alive_player_names() if hasattr(self.state_builder, "alive_player_names") else set()
            if meta_target in alive:
                self._current_target = meta_target

        # Track VLM kill defer/stalk for next step
        action_type = result.get("metadata", {}).get("action", "") if result else ""
        if action_type in ("vlm_deferred", "vlm_stalk", "kill_defer_far_target"):
            # kill_defer_far_target is included so the LLM doesn't pick the
            # same too-far target again next step (it gets forced to explore
            # for one step, breaking same-target DEFER loops).
            self._vlm_kill_deferred = True
        else:
            self._vlm_kill_deferred = False

        # Safety net: empty result → wander
        if not result or not result.get("code", "").strip():
            if not in_meeting and game_state.own_position:
                code = AmongUsEnvAdapter.build_wander_js(
                    game_state.own_position, role=self.role,
                )
                result = {"code": code}
            else:
                result = {"code": ""}

        self._last_mode = plan.mode.upper()
        self._prev_game_state = game_state

        self.memory.update(
            tick=game_state.tick,
            action_summary=f"mode={plan.mode}: {plan.short_term_plan[:80] if plan.short_term_plan else ''}",
            state_summary=f"phase={game_state.phase} alive={len(game_state.visible_players)}",
            events=[
                {
                    "tick": game_state.tick,
                    "message": getattr(e, "message", str(e)),
                    "type": getattr(e, "type", ""),
                }
                for e in (game_state.events or [])
            ] or None,
        )

        self._maybe_save()

        code = result["code"]

        # Ego mode: inject JS wrapper to limit bot.nearestEntity range
        if self.config.state_mode == "ego" and code and code.strip():
            ego_range = self.config.ego_visible_range
            ego_wrapper = (
                f"// Ego mode: limit entity detection to {ego_range} blocks\n"
                f"if (!bot._egoWrapped) {{\n"
                f"  const _origNE = bot.nearestEntity.bind(bot);\n"
                f"  bot.nearestEntity = (filter) => _origNE(\n"
                f"    e => filter(e) && e.position.distanceTo(bot.entity.position) < {ego_range}\n"
                f"  );\n"
                f"  bot._egoWrapped = true;\n"
                f"}}\n"
            )
            code = ego_wrapper + code

        # DEBUG: inject JS to log bot.entities player count (temporary)
        if verbose and code and code.strip():
            ego_range = self.config.ego_visible_range if self.config.state_mode == "ego" else "unlimited"
            debug_js = (
                "// DEBUG entity check\n"
                "const _dbgPlayers = Object.values(bot.entities).filter("
                "e => e.type === 'player' && e.username !== bot.username);\n"
                f"const _dbgInRange = _dbgPlayers.filter("
                f"e => e.position.distanceTo(bot.entity.position) < "
                f"{self.config.ego_visible_range if self.config.state_mode == 'ego' else 9999});\n"
                "console.log('[JS_ENTITY_DEBUG] bot=' + bot.username + "
                f"' mode={self.config.state_mode}' + "
                "' all=' + _dbgPlayers.length + "
                "' inRange=' + _dbgInRange.length + "
                "' names=' + _dbgPlayers.map(e => e.username + '(' + "
                "Math.round(e.position.distanceTo(bot.entity.position)) + 'b)').join(','));\n"
            )
            code = debug_js + code

        # Strip bot.chat() from codegen output during explore phase (RUBRIC R1)
        if not in_meeting and code and "bot.chat" in code:
            code = self.env_adapter.strip_chat_from_js(code)

        action_type = Action.NEW if code.strip() else Action.RESUME
        return Action(type=action_type, code=code)

    # ──────────────────────────────────────────────────────────────────
    # Module dispatch
    # ──────────────────────────────────────────────────────────────────

    def _dispatch(self, mode: str, game_state: AmongUsGameState, raw_obs, verbose: bool,
                  target: Optional[str] = None, plan=None) -> Optional[dict]:
        try:
            if mode == "vote":
                return self.vote_module.run(
                    game_state, self.state_builder, self.memory, verbose,
                )
            elif mode == "meeting":
                return self.meeting_module.run(
                    self.state_builder, self.memory, verbose,
                )
            elif mode == "kill":
                return self.kill_module.run(
                    game_state, self.state_builder, self.memory, verbose,
                    target=target,
                )
            elif mode == "report":
                return self.report_module.run(
                    game_state, self.state_builder, self.memory, verbose,
                )
            elif mode == "mission":
                return self.mission_module.run(
                    raw_obs, game_state, self.state_builder, self.memory, verbose,
                    logger=self.logger,
                )
            else:  # explore
                return self.move_module.run(
                    raw_obs, game_state, self.state_builder, self.memory, verbose,
                    logger=self.logger,
                )
        except Exception as e:
            print(f"[Aria:{self.bot_name}] Module error ({mode}): {e}")
            return None

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_code_running(code_info) -> bool:
        if code_info is None:
            return False
        if isinstance(code_info, dict):
            return bool(code_info.get("is_running", False))
        return bool(getattr(code_info, "is_running", False))

    def _maybe_save(self) -> None:
        tick = self.state_builder.get_tick()
        if tick - self._last_save_tick >= SAVE_INTERVAL_TICKS:
            self._save_all()
            self._last_save_tick = tick

    def _record_state_step(self) -> None:
        """Append a per-step state snapshot to state_timeline.json (keyed by env
        step, like trace.json) so analysis can reconstruct each agent's state
        over time — position, suspicion, dead players, etc."""
        from .llm import get_current_step
        step = get_current_step()
        key = str(step) if isinstance(step, int) else "unknown"
        # deepcopy: to_dict() returns live references for mutable fields
        # (suspicion, mission_status, chat_log, ...); without copying, every
        # step entry would alias the same objects and show only final values.
        self._state_timeline[key] = copy.deepcopy(self.state_builder.to_dict())
        try:
            with open(os.path.join(self.save_path, "state_timeline.json"),
                      "w", encoding="utf-8") as f:
                json.dump({"agent": self.bot_name, "steps": self._state_timeline},
                          f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass  # logging must never crash a real run

    def _save_all(self) -> None:
        self.state_builder.save_to_json(
            os.path.join(self.save_path, "state.json")
        )
        self.memory.save_to_json(
            os.path.join(self.save_path, "memory.json")
        )
        if self.skill_memory:
            self.skill_memory.save_to_json(
                os.path.join(self.save_path, "skill_memory.json")
            )
        self.planner.save_to_json(
            os.path.join(self.save_path, "planner.json")
        )
        self.reflector.save_to_json(
            os.path.join(self.save_path, "reflector.json")
        )
