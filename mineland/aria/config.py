"""
AriaConfig — Config-driven ablation for the Aria Among Us agent.

Every experimental axis is a field in this dataclass. Validation at
construction time catches impossible combinations (ValueError) and
flags not-recommended ones (warnings.warn).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class AriaConfig:
    """Central configuration for the Aria agent.

    Experimental axes
    -----------------
    1. state_mode       — ego (current obs only) vs privileged (stored positions)
    2. memory_mode      — none / window / episodic / semantic
    3. use_skill_memory — list-based skill storage (requires reflection)
    4. planning_mode    — reactive / hierarchical (shortterm is deactivated)
    5. reflection_mode  — none / post_action / periodic / meeting
    6. prompt_style     — deterministic (prescriptive) / minimal (emergent)
    """

    # ── Role ────────────────────────────────────────────────────────────
    role: Literal["imposter", "crewmate"] = "crewmate"

    # ── 1. State representation ─────────────────────────────────────────
    state_mode: Literal["ego", "privileged"] = "privileged"
    ego_visible_range: float = 36.0  # ego: filter target_entities to this range (~6 chunks)
    # privileged: optional perception-radius cap for coordinate ingestion. None =
    # unlimited (mineflayer server view ~96 blocks, the default privileged cheat).
    # Set to e.g. 36.0 to shrink the acquisition radius to ego's while KEEPING the
    # other privileged cheats (no LoS/frustum filter, persistence of last-known
    # positions). Used for the distance-only leave-one-out ablation vs privileged.
    privileged_visible_range: Optional[float] = None
    # No-vision (text-only) ablation. When False, RGB screenshots are stripped
    # from EVERY LLM call, and the KILL / SURVEILLANCE modules replace their
    # camera scans with distance-based text reasoning over player positions
    # (witness = anyone close enough by Euclidean distance; walls/LoS unknown,
    # just like a sightless agent). Default True = unchanged vision behavior.
    use_vision: bool = True

    # ── 2. Memory ───────────────────────────────────────────────────────
    memory_mode: Literal["none", "window", "episodic", "semantic"] = "semantic"
    memory_window_size: int = 10
    memory_belief_update: Literal["rule", "llm"] = "rule"

    # ── 3. Skill memory ────────────────────────────────────────────────
    use_skill_memory: bool = False
    skill_memory_sources: List[Literal["self", "observed"]] = field(
        default_factory=lambda: ["self"],
    )
    skill_memory_size: int = 50

    # ── 4. Planning ─────────────────────────────────────────────────────
    planning_mode: Literal["reactive", "shortterm", "hierarchical"] = "reactive"
    planning_policy: Literal["rule", "llm"] = "rule"
    longterm_plan_interval: int = 5

    # ── 5. Reflection ───────────────────────────────────────────────────
    reflection_mode: Literal["none", "post_action", "periodic", "meeting"] = "none"
    reflection_interval: int = 10

    # ── 6. Prompt style ─────────────────────────────────────────────────
    # Scope: this axis affects only the *text-LLM* modules whose prompts are
    # loaded via prompt_template/{deterministic,minimal}/  — currently
    # MeetingModule, VoteModule, MoveModule — plus MeetingReflector and
    # ShorttermPlanner (LLM mode), and MeetingModule's dead-section guidance.
    # VLM modules (KillModule, ReportModule, EmergencyModule, SurveillanceModule),
    # MissionModule's codegen, ReactivePlanner, and Post/PeriodicReflectors use
    # hardcoded prompts and ignore this axis. See docs/aria_ko.md §6.
    prompt_style: Literal["deterministic", "minimal"] = "deterministic"

    # ── 7. Module policy (kill/report execution strategy) ───────────────
    # rule: nearest entity → immediate action (Steve-style)
    # vlm:  RGB + context → LLM/VLM judges safety, target selection, timing
    module_policy: Literal["rule", "vlm"] = "rule"

    # ── 8. VLM kill fallback ──────────────────────────────────────────
    # After N consecutive VLM defer/stalk, force rule-based kill (<0 = no fallback)
    vlm_kill_fallback: int = 1

    # ── LLM settings ───────────────────────────────────────────────────
    llm_model: str = "gpt-4.1-mini"
    vlm_model: str = "gpt-4.1-mini"
    max_tokens: int = 512
    temperature: float = 0.3
    api_base: Optional[str] = None

    # ── Agent identity ──────────────────────────────────────────────────
    bot_name: str = "Aria"
    personal_message: Optional[str] = None
    save_path: str = "./storage"
    load_memory: bool = False

    # ── Among Us specifics (passed through, not agent logic) ───────────
    all_players: Optional[list] = None
    player_id_map: Optional[dict] = None
    self_report_probability: float = 0.1
    kill_detection_range: float = 10
    # Named-target stalk give-up distance. If the planner-chosen kill target
    # is farther than this, KillModule returns DEFER (empty code) so the
    # planner can re-pick a target or switch modes next step, instead of
    # silently swapping to nearestEntity or stalking forever.
    stalk_giveup_distance: float = 10.0

    # VLM scan-start threshold. Inside _vlm_kill, if the named target is
    # farther than this, the imposter stalks (approaches) before initiating
    # the witness/forward scan rotation.
    vlm_scan_distance: float = 8.0

    # ── Mission coords lookup (LLM hallucination workaround) ────────────
    # When True (default), MissionModule overrides the LLM-generated
    # `objective` text with deterministic (x, y, z) parsed from
    # personal_message's "Mission Location" section, keyed by mission_key.
    #
    # Why: the decision-stage LLM picks the right mission_key but often
    # hallucinates the wrong coords (e.g. mission2 → "(91, 65, 116)" which
    # is mission8's location). Without this override the bot navigates to
    # the wrong spot, the server never registers a Mission Started, and
    # the game stalls (see seq#3 stagnant case where Ethan retried 4×
    # before finally getting coords right).
    #
    # When False (default): the LLM's objective text is used verbatim, so
    # coord accuracy is part of the LLM capability being measured. Useful
    # for ablations isolating "instruction-following / lookup" capability.
    # Game progression may suffer (more stagnant runs) if LLM hallucinates
    # coords frequently. Pair with a strengthened mission decision prompt
    # to mitigate hallucination at the prompt level.
    mission_coords_lookup: bool = False

    # ── Multi-imposter coordination ─────────────────────────────────────
    # Name of this imposter's teammate (other imposter). Used to prevent
    # team-kills and team-votes. None for crewmates or single-imposter games.
    teammate_imposter: Optional[str] = None

    # ── Imposter-only: mission progress visibility toggle ───────────────
    # When True, imposter sees exact "done/total (k crewmates × 3)" in context.
    # When False, imposter only sees a coarse bucket (early/mid/late).
    # Set False for realism (standard Among Us imposters see only a bar).
    show_mission_progress: bool = True

    # ──────────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        self._validate_impossible()
        self._validate_warnings()

    # -- impossible combos: raise ValueError --------------------------

    def _validate_impossible(self) -> None:
        # 1) reflection without any output target
        if (
            self.reflection_mode != "none"
            and self.memory_mode == "none"
            and not self.use_skill_memory
        ):
            raise ValueError(
                f"reflection_mode={self.reflection_mode!r} requires at least one "
                f"output target: memory_mode != 'none' or use_skill_memory=True. "
                f"Currently memory_mode='none' and use_skill_memory=False — "
                f"reflection output has nowhere to go."
            )

        # 2) hierarchical + rule policy is meaningless
        if self.planning_mode == "hierarchical" and self.planning_policy == "rule":
            raise ValueError(
                "planning_mode='hierarchical' with planning_policy='rule' is "
                "invalid. Hierarchical planning inherently requires LLM reasoning. "
                "Use planning_mode='reactive' instead, or set planning_policy='llm'."
            )

        # 3) shortterm is deactivated — collapses onto reactive (single-LLM
        #    mode+target decision). Use 'reactive' for the single-call axis,
        #    'hierarchical' for the long-term + short-term two-call axis.
        if self.planning_mode == "shortterm":
            raise ValueError(
                "planning_mode='shortterm' is deactivated. It is functionally "
                "equivalent to 'reactive' (both are single-LLM mode+target "
                "decisions). Use planning_mode='reactive' or 'hierarchical'."
            )

    # -- not-recommended combos: emit warnings -----------------------

    def _validate_warnings(self) -> None:
        # 1) ego + deterministic
        if self.state_mode == "ego" and self.prompt_style == "deterministic":
            warnings.warn(
                "state_mode='ego' + prompt_style='deterministic': deterministic "
                "prompts prescribe tactics like 'stay near crewmate' that require "
                "stored player positions which ego mode does not provide. Consider "
                "prompt_style='minimal' with ego mode.",
                UserWarning,
                stacklevel=3,
            )

        # 2) hierarchical + no memory
        if self.planning_mode == "hierarchical" and self.memory_mode == "none":
            warnings.warn(
                "planning_mode='hierarchical' + memory_mode='none': long-term "
                "strategy is regenerated from scratch each interval with no "
                "historical context. 'Long-term' planning has little meaning "
                "without persistent memory.",
                UserWarning,
                stacklevel=3,
            )

        # 3) deterministic + no memory
        if self.prompt_style == "deterministic" and self.memory_mode == "none":
            warnings.warn(
                "prompt_style='deterministic' + memory_mode='none': deterministic "
                "prompts prescribe tracking alibis and cross-referencing claims, "
                "but the agent has no memory to perform these actions.",
                UserWarning,
                stacklevel=3,
            )

        # 4) skill_memory without reflection → auto-disable
        if self.use_skill_memory and self.reflection_mode == "none":
            warnings.warn(
                "use_skill_memory=True but reflection_mode='none': skill memory "
                "requires reflection as its storage trigger. Skill memory will "
                "be effectively disabled (no entries will be saved).",
                UserWarning,
                stacklevel=3,
            )

        # 5) vlm modules without memory to record observations into
        if self.module_policy == "vlm" and self.memory_mode == "none":
            warnings.warn(
                "module_policy='vlm' + memory_mode='none': VLM surveillance "
                "observations have nowhere to be stored, and emergency-meeting "
                "decisions lose context across steps. Consider memory_mode='window' "
                "or 'semantic' when running VLM modules.",
                UserWarning,
                stacklevel=3,
            )

    # ──────────────────────────────────────────────────────────────────
    # Convenience
    # ──────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """One-line config summary for logging."""
        return (
            f"Aria[{self.role}] "
            f"state={self.state_mode} mem={self.memory_mode} "
            f"skill={'on' if self.use_skill_memory else 'off'} "
            f"plan={self.planning_mode}/{self.planning_policy} "
            f"reflect={self.reflection_mode} prompt={self.prompt_style} "
            f"module={self.module_policy} temp={self.temperature}"
        )


# ──────────────────────────────────────────────────────────────────────
# Preset configs
# ──────────────────────────────────────────────────────────────────────

def steve_equivalent(**overrides) -> AriaConfig:
    """AriaConfig that reproduces Steve's architecture."""
    defaults = dict(
        state_mode="privileged",
        memory_mode="semantic",
        memory_belief_update="rule",
        use_skill_memory=False,
        planning_mode="reactive",
        planning_policy="rule",
        reflection_mode="none",
        prompt_style="deterministic",
    )
    defaults.update(overrides)
    return AriaConfig(**defaults)


def baseline(**overrides) -> AriaConfig:
    """Minimal baseline config for lower-bound measurement."""
    defaults = dict(
        state_mode="ego",
        memory_mode="none",
        use_skill_memory=False,
        planning_mode="reactive",
        planning_policy="llm",
        reflection_mode="none",
        prompt_style="minimal",
    )
    defaults.update(overrides)
    return AriaConfig(**defaults)


def full(**overrides) -> AriaConfig:
    """Full-featured config for upper-bound measurement."""
    defaults = dict(
        state_mode="privileged",
        memory_mode="semantic",
        memory_belief_update="llm",
        use_skill_memory=True,
        skill_memory_sources=["self", "observed"],
        planning_mode="hierarchical",
        planning_policy="llm",
        reflection_mode="meeting",
        prompt_style="minimal",
    )
    defaults.update(overrides)
    return AriaConfig(**defaults)
