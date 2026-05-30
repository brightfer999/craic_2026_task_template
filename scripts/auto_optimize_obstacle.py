#!/usr/bin/env python3
"""Phase 4: Auto-optimize obstacle planner params (DWA + gap-follow).

Runs repeated scene1 obstacle trials, records per-frame planner diagnostics
to planner.jsonl, scores each trial with a multi-term reward/loss function,
and updates params with staged exploration plus a lightweight quasi-Newton step.

Parameter groups:
  Gap-follow   — planner_min_reward, gap_safety_radius, gap_lookahead, ...
  DWA weights  — goal_heading, clearance_reward, memory_cost, oscillation, ...
  Safety       — front_stop, danger_dist, caution_dist
  Inflation    — obstacle_inflation_radius
  Speed        — v_max, w_max
  Escape       — min_acceptable_score, stuck_time

Workflow:
  seed → record planner.jsonl → compute reward/loss → update params → next round
"""

import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# ============================================================================
# Parameter space — all tunable knobs
# ============================================================================

BASE_PARAMS = {
    # -- Gap-follow (retained) --
    "planner_min_reward": -2.5,
    "obstacle_gap_safety_radius": 0.40,
    "obstacle_gap_lookahead": 2.20,
    "obstacle_yaw_kp": 0.70,
    "front_stop_distance": 0.55,       # raised: stop earlier before walls
    "front_backoff_distance": 0.35,    # raised: back off earlier
    "cone_min_points": 3,
    "wall_score_threshold": 0.50,
    # -- DWA weights --
    "dwa_goal_heading_weight": 2.2,
    "dwa_clearance_reward_gain": 0.35,
    "dwa_memory_cost_weight": 0.65,
    "dwa_oscillation_weight": 1.8,
    "dwa_goal_proximity_reward": 1.0,
    "dwa_forward_progress_weight": 1.15,
    "dwa_caution_turn_reward": 0.65,
    "dwa_danger_extra_cost": 4.0,
    "dwa_global_yaw_weight": 2.0,
    "dwa_global_yaw_soft_limit_deg": 45.0,
    "dwa_global_yaw_hard_limit_deg": 95.0,
    "dwa_uturn_cost": 50.0,
    # -- Safety --
    "obstacle_danger_dist": 0.35,      # raised: treat anything < 0.35m as danger
    "obstacle_caution_dist": 0.55,     # raised: wider caution zone
    # -- Inflation --
    "obstacle_inflation_radius": 0.22, # raised: inflate obstacles more
    # -- Speed --
    "v_max": 0.20,                     # lowered: safer default speed
    "w_max": 0.45,
    # -- Escape --
    "dwa_min_acceptable_score": -4.0,
    "escape_stuck_time": 8.0,
    # -- Map --
    "local_map_cell_size": 0.20,
    "local_map_decay": 0.985,
    # -- Wall-following (was NOT in optimization — root cause of crashing) --
    "wall_follow_distance": 1.05,      # stay farther from right wall
    "wall_follow_kp_lat": 0.15,        # gentler lateral correction
    "wall_follow_kp_ang": 0.12,        # gentler angular correction
    "corridor_bias_startup": 0.05,     # much weaker rightward bias at start
    # -- Goal-manager (was NOT in optimization) --
    "traversability_min_clearance": 0.65,  # sectors need more clearance
    "goal_side_bias_gain": 0.15,       # weaker corridor bias in goal gen
    # -- Wall guard (new proactive safety layer) --
    "wall_guard_trigger_dist": 0.55,   # distance at which wall guard overrides steering
    "wall_guard_turn_gain": 0.70,      # how strongly to turn away from near wall
}

PARAM_BOUNDS = {
    "planner_min_reward": (-5.0, 0.0),
    "obstacle_gap_safety_radius": (0.32, 0.55),
    "obstacle_gap_lookahead": (1.60, 3.00),
    "obstacle_yaw_kp": (0.35, 1.15),
    "front_stop_distance": (0.38, 0.75),
    "front_backoff_distance": (0.25, 0.50),
    "cone_min_points": (2, 7),
    "wall_score_threshold": (0.25, 0.75),
    "dwa_goal_heading_weight": (1.0, 4.0),
    "dwa_clearance_reward_gain": (0.10, 0.70),
    "dwa_memory_cost_weight": (0.20, 1.20),
    "dwa_oscillation_weight": (0.5, 3.5),
    "dwa_goal_proximity_reward": (0.3, 2.0),
    "dwa_forward_progress_weight": (0.3, 2.5),
    "dwa_caution_turn_reward": (0.0, 1.8),
    "dwa_danger_extra_cost": (1.0, 9.0),
    "dwa_global_yaw_weight": (0.8, 7.0),
    "dwa_global_yaw_soft_limit_deg": (25.0, 75.0),
    "dwa_global_yaw_hard_limit_deg": (70.0, 130.0),
    "dwa_uturn_cost": (35.0, 140.0),
    "obstacle_danger_dist": (0.22, 0.50),
    "obstacle_caution_dist": (0.40, 0.80),
    "obstacle_inflation_radius": (0.12, 0.35),
    "v_max": (0.12, 0.28),
    "w_max": (0.30, 0.65),
    "dwa_min_acceptable_score": (-7.0, -1.0),
    "escape_stuck_time": (4.0, 15.0),
    "local_map_cell_size": (0.12, 0.32),
    "local_map_decay": (0.94, 0.997),
    # -- New: wall-following --
    "wall_follow_distance": (0.70, 1.50),
    "wall_follow_kp_lat": (0.05, 0.35),
    "wall_follow_kp_ang": (0.05, 0.30),
    "corridor_bias_startup": (-0.10, 0.20),
    # -- New: goal-manager --
    "traversability_min_clearance": (0.40, 1.00),
    "goal_side_bias_gain": (0.05, 0.45),
    # -- New: wall guard --
    "wall_guard_trigger_dist": (0.40, 0.85),
    "wall_guard_turn_gain": (0.40, 1.20),
}

MUTATION_SIGMA = {
    "planner_min_reward": 0.45,
    "obstacle_gap_safety_radius": 0.035,
    "obstacle_gap_lookahead": 0.18,
    "obstacle_yaw_kp": 0.10,
    "front_stop_distance": 0.035,
    "front_backoff_distance": 0.025,
    "cone_min_points": 1.0,
    "wall_score_threshold": 0.07,
    "dwa_goal_heading_weight": 0.35,
    "dwa_clearance_reward_gain": 0.07,
    "dwa_memory_cost_weight": 0.12,
    "dwa_oscillation_weight": 0.35,
    "dwa_goal_proximity_reward": 0.20,
    "dwa_forward_progress_weight": 0.25,
    "dwa_caution_turn_reward": 0.22,
    "dwa_danger_extra_cost": 0.8,
    "dwa_global_yaw_weight": 0.75,
    "dwa_global_yaw_soft_limit_deg": 6.0,
    "dwa_global_yaw_hard_limit_deg": 7.0,
    "dwa_uturn_cost": 12.0,
    "obstacle_danger_dist": 0.030,
    "obstacle_caution_dist": 0.040,
    "obstacle_inflation_radius": 0.030,
    "v_max": 0.020,
    "w_max": 0.05,
    "dwa_min_acceptable_score": 0.60,
    "escape_stuck_time": 1.2,
    "local_map_cell_size": 0.025,
    "local_map_decay": 0.006,
    # -- New: wall-following --
    "wall_follow_distance": 0.10,
    "wall_follow_kp_lat": 0.035,
    "wall_follow_kp_ang": 0.030,
    "corridor_bias_startup": 0.040,
    # -- New: goal-manager --
    "traversability_min_clearance": 0.07,
    "goal_side_bias_gain": 0.05,
    # -- New: wall guard --
    "wall_guard_trigger_dist": 0.06,
    "wall_guard_turn_gain": 0.10,
}

INT_PARAMS = {"cone_min_points"}


AGGRESSIVE_TARGET_PARAMS = {
    "planner_min_reward": -4.6,
    "obstacle_gap_safety_radius": 0.34,
    "obstacle_gap_lookahead": 2.75,
    "obstacle_yaw_kp": 0.98,
    "front_stop_distance": 0.43,
    "front_backoff_distance": 0.28,
    "cone_min_points": 2,
    "wall_score_threshold": 0.36,
    "dwa_goal_heading_weight": 1.45,
    "dwa_clearance_reward_gain": 0.48,
    "dwa_memory_cost_weight": 0.92,
    "dwa_oscillation_weight": 1.10,
    "dwa_goal_proximity_reward": 1.45,
    "dwa_forward_progress_weight": 2.10,
    "dwa_caution_turn_reward": 1.45,
    "dwa_danger_extra_cost": 2.60,
    "dwa_global_yaw_weight": 3.6,
    "dwa_global_yaw_soft_limit_deg": 38.0,
    "dwa_global_yaw_hard_limit_deg": 88.0,
    "dwa_uturn_cost": 92.0,
    "obstacle_danger_dist": 0.28,
    "obstacle_caution_dist": 0.46,
    "obstacle_inflation_radius": 0.16,
    "v_max": 0.26,
    "w_max": 0.62,
    "dwa_min_acceptable_score": -6.4,
    "escape_stuck_time": 5.2,
    "local_map_cell_size": 0.16,
    "local_map_decay": 0.972,
    "wall_follow_distance": 0.92,
    "wall_follow_kp_lat": 0.24,
    "wall_follow_kp_ang": 0.20,
    "corridor_bias_startup": 0.02,
    "traversability_min_clearance": 0.48,
    "goal_side_bias_gain": 0.09,
    "wall_guard_trigger_dist": 0.47,
    "wall_guard_turn_gain": 0.98,
}


# ============================================================================
# Helpers
# ============================================================================


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def normalize_param(name: str, value: float):
    lo, hi = PARAM_BOUNDS[name]
    value = clamp(float(value), lo, hi)
    if name in INT_PARAMS:
        return int(round(value))
    return round(value, 5)


def param_to_unit(name: str, value: float) -> float:
    lo, hi = PARAM_BOUNDS[name]
    return clamp((float(value) - lo) / max(1e-9, hi - lo), 0.0, 1.0)


def unit_to_param(name: str, unit_value: float):
    lo, hi = PARAM_BOUNDS[name]
    return normalize_param(name, lo + clamp(unit_value, 0.0, 1.0) * (hi - lo))


def blend_params(left: Dict[str, float], right: Dict[str, float], alpha: float) -> Dict[str, float]:
    alpha = clamp(alpha, 0.0, 1.0)
    out = {}
    for name in BASE_PARAMS:
        out[name] = normalize_param(
            name,
            float(left[name]) * (1.0 - alpha) + float(right.get(name, left[name])) * alpha,
        )
    return out


def optimizer_phase(trial_idx: int, iterations: int, args) -> dict:
    ratio = trial_idx / max(1.0, float(iterations - 1))
    aggressive_end = clamp(float(args.aggressive_fraction), 0.05, 0.80)
    if trial_idx == 0:
        return {
            "name": "baseline",
            "score_phase": "converge" if args.score_profile == "safety" else "explore",
            "explore_scale": 0.0,
            "mutation_probability": 0.0,
            "aggressive_bias": 0.0,
            "wide_jump_probability": 0.0,
            "direction_gain": 0.0,
        }
    if ratio <= aggressive_end:
        local = ratio / max(1e-6, aggressive_end)
        scale = args.early_explore_scale * (1.0 - 0.35 * local)
        return {
            "name": "explore",
            "score_phase": "explore" if args.score_profile == "staged" else "converge",
            "explore_scale": scale,
            "mutation_probability": 0.96,
            "aggressive_bias": 0.75 - 0.35 * local,
            "wide_jump_probability": 0.22,
            "direction_gain": 0.0,
        }
    if ratio <= 0.75:
        local = (ratio - aggressive_end) / max(1e-6, 0.75 - aggressive_end)
        return {
            "name": "refine",
            "score_phase": "refine" if args.score_profile == "staged" else "converge",
            "explore_scale": 1.05 - 0.40 * local,
            "mutation_probability": 0.78,
            "aggressive_bias": 0.18 * (1.0 - local),
            "wide_jump_probability": 0.08,
            "direction_gain": 0.36,
        }
    local = (ratio - 0.75) / 0.25
    return {
        "name": "converge",
        "score_phase": "converge",
        "explore_scale": max(args.final_explore_scale, 0.55 - 0.20 * local),
        "mutation_probability": 0.58,
        "aggressive_bias": 0.0,
        "wide_jump_probability": 0.02,
        "direction_gain": 0.22,
    }


def mutate_params(
    best: Dict[str, float],
    rng: random.Random,
    explore_scale: float,
    mutation_probability: float = 0.72,
    direction: Dict[str, float] = None,
    direction_gain: float = 0.0,
) -> Dict[str, float]:
    params = dict(best)
    for name, sigma in MUTATION_SIGMA.items():
        if rng.random() < mutation_probability:
            step = rng.gauss(0.0, sigma * explore_scale)
            if direction:
                lo, hi = PARAM_BOUNDS[name]
                step += direction_gain * direction.get(name, 0.0) * (hi - lo)
            params[name] = normalize_param(name, params[name] + step)
    return params


def broad_explore_params(best: Dict[str, float], rng: random.Random, phase: dict) -> Dict[str, float]:
    center = blend_params(best, AGGRESSIVE_TARGET_PARAMS, phase.get("aggressive_bias", 0.0))
    params = dict(center)
    for name, sigma in MUTATION_SIGMA.items():
        if rng.random() < phase.get("wide_jump_probability", 0.0):
            lo, hi = PARAM_BOUNDS[name]
            params[name] = normalize_param(name, rng.uniform(lo, hi))
        elif rng.random() < phase.get("mutation_probability", 0.9):
            params[name] = normalize_param(
                name,
                params[name] + rng.gauss(0.0, sigma * phase.get("explore_scale", 1.0)),
            )
    return params


def empirical_direction(best: Dict[str, float], history: List[dict]) -> Dict[str, float]:
    valid = [
        h for h in history
        if "params" in h and math.isfinite(float(h.get("score", -float("inf"))))
    ]
    if len(valid) < 4:
        return {}
    ranked = sorted(valid, key=lambda item: float(item.get("score", -float("inf"))), reverse=True)
    top = ranked[:min(4, len(ranked))]
    bottom = ranked[-min(4, len(ranked)):]
    best_score = float(top[0].get("score", 0.0))
    worst_score = float(bottom[-1].get("score", best_score - 1.0))
    span = max(1.0, abs(best_score - worst_score))
    direction = {}
    for name in BASE_PARAMS:
        center = param_to_unit(name, best[name])
        top_pull = 0.0
        bottom_push = 0.0
        for item in top:
            weight = max(0.05, (float(item.get("score", 0.0)) - worst_score) / span)
            top_pull += weight * (param_to_unit(name, item["params"][name]) - center)
        for item in bottom:
            weight = max(0.05, (best_score - float(item.get("score", 0.0))) / span)
            bottom_push += weight * (center - param_to_unit(name, item["params"][name]))
        direction[name] = clamp((top_pull + 0.45 * bottom_push) / max(1.0, len(top)), -0.40, 0.40)
    return direction


def quasi_newton_params(best: Dict[str, float], history: List[dict],
                        rng: random.Random, phase: dict) -> Dict[str, float]:
    direction = empirical_direction(best, history)
    if not direction:
        return mutate_params(
            best,
            rng,
            phase.get("explore_scale", 0.8),
            phase.get("mutation_probability", 0.65),
        )
    return mutate_params(
        best,
        rng,
        phase.get("explore_scale", 0.8),
        phase.get("mutation_probability", 0.65),
        direction=direction,
        direction_gain=phase.get("direction_gain", 0.25),
    )


def propose_params(best: Dict[str, float], history: List[dict],
                   rng: random.Random, phase: dict, optimizer: str) -> Tuple[Dict[str, float], str]:
    if optimizer == "gaussian":
        return mutate_params(
            best,
            rng,
            phase.get("explore_scale", 1.0),
            phase.get("mutation_probability", 0.72),
        ), "gaussian"
    if phase["name"] == "explore":
        return broad_explore_params(best, rng, phase), "broad_explore"
    if optimizer == "hybrid" and phase["name"] in ("refine", "converge"):
        return quasi_newton_params(best, history, rng, phase), "quasi_newton"
    return mutate_params(
        best,
        rng,
        phase.get("explore_scale", 1.0),
        phase.get("mutation_probability", 0.72),
    ), "gaussian"


def ros_private_args(params: Dict[str, float]) -> List[str]:
    args = []
    for name, value in sorted(params.items()):
        args.append("_%s:=%s" % (name, value))
    return args


def read_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def planner_log_progress(path: Path) -> Tuple[int, float, float]:
    """Return (frame_count, last_wall_time, max_stage_distance)."""
    if not path.exists():
        return 0, 0.0, 0.0
    frame_count = 0
    last_wall_time = 0.0
    max_distance = 0.0
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") != "frame":
                    continue
                frame_count += 1
                last_wall_time = max(last_wall_time, float(rec.get("wall_time", 0.0)))
                max_distance = max(max_distance, float(rec.get("stage_distance", 0.0)))
    except OSError:
        return 0, 0.0, 0.0
    return frame_count, last_wall_time, max_distance


def _descendant_pids(pid: int) -> List[int]:
    """Return all descendant PIDs recursively (crosses session boundaries)."""
    result: List[int] = []
    try:
        children_path = "/proc/%d/task/%d/children" % (pid, pid)
        if os.path.exists(children_path):
            with open(children_path, "r") as fh:
                child_pids = [int(p) for p in fh.read().strip().split()]
            for child_pid in child_pids:
                result.append(child_pid)
                result.extend(_descendant_pids(child_pid))
    except (OSError, ValueError):
        pass
    return result


def _kill_recursive(pids: Iterable[int], sig: int, grace: float) -> bool:
    """Send signal to every pid; return True when none remain alive after grace."""
    for pid in pids:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
    if sig == signal.SIGKILL:
        return True
    deadline = time.time() + grace
    while time.time() < deadline:
        alive = [p for p in pids if _pid_alive(p)]
        if not alive:
            return True
        time.sleep(0.15)
    return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def terminate_process_tree(proc: subprocess.Popen, grace_sec: float = 5.0) -> None:
    """Terminate the trial process and all descendants (crossing session boundaries).

    After killing, sleeps briefly to let the OS release ports (e.g. roscore 11311).
    """
    if proc.poll() is not None:
        return
    try:
        all_pids = [proc.pid] + _descendant_pids(proc.pid)
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            if not _kill_recursive(all_pids, signal.SIGTERM, grace_sec):
                _kill_recursive(all_pids, signal.SIGKILL, 1.0)
            # Let the OS release TCP ports (roscore 11311) before next trial
            time.sleep(1.5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_monitored_trial_process(cmd: List[str], cwd: str, stdout, planner_log: Path, args) -> Tuple[int, bool, str]:
    """Run one scene script and stop it when sim/planner progress becomes stale."""
    creationflags = 0
    popen_kwargs = {}
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=stdout,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
        **popen_kwargs,
    )

    start = time.time()
    last_frame_count = 0
    last_progress_wall = start
    timeout_hit = False
    monitor_reason = "process_exit"

    while True:
        ret = proc.poll()
        now = time.time()
        frame_count, last_frame_wall, max_distance = planner_log_progress(planner_log)
        if frame_count > last_frame_count:
            last_frame_count = frame_count
            last_progress_wall = now
            stdout.write(
                "[watchdog] frames=%d max_dist=%.3f\n" % (frame_count, max_distance)
            )
            stdout.flush()
        elif last_frame_wall > 0.0:
            # If the child wrote wall_time but the file watcher missed a frame count
            # change, still treat the file timestamp as fresh planner progress.
            last_progress_wall = max(last_progress_wall, last_frame_wall)

        if ret is not None:
            return ret, timeout_hit, monitor_reason

        if now - start >= args.timeout:
            timeout_hit = True
            monitor_reason = "trial_timeout"
            stdout.write("[watchdog] trial timeout, terminating process tree\n")
            stdout.flush()
            terminate_process_tree(proc, args.kill_grace)
            return 124, timeout_hit, monitor_reason

        if not args.disable_watchdog:
            if frame_count == 0 and now - start >= args.startup_frame_timeout:
                monitor_reason = "no_planner_frames"
                stdout.write("[watchdog] no planner frames, restarting trial\n")
                stdout.flush()
                terminate_process_tree(proc, args.kill_grace)
                return 125, timeout_hit, monitor_reason
            if frame_count > 0 and now - last_progress_wall >= args.sim_stale_timeout:
                monitor_reason = "planner_stale"
                stdout.write("[watchdog] planner stale, restarting trial\n")
                stdout.flush()
                terminate_process_tree(proc, args.kill_grace)
                return 125, timeout_hit, monitor_reason

        time.sleep(max(0.2, args.watchdog_poll))


# ============================================================================
# Trial scoring (Phase 4 enhanced)
# ============================================================================


def analyze_failure(frames: List[dict], returncode: int, timeout_hit: bool,
                    target_distance: float, max_distance: float) -> dict:
    """Classify the failure mode of a trial from planner.jsonl frames."""
    reasons = []
    details = {}

    if timeout_hit:
        reasons.append("timeout")
        details["timeout"] = True

    if returncode != 0 and not timeout_hit:
        reasons.append("process_error")
        details["returncode"] = returncode
        if returncode == 125:
            reasons.append("sim_or_planner_stale")

    if not frames:
        reasons.append("no_planner_frames")
        return {"failure_modes": reasons, "details": details}

    # Wall-hit: very few frames, minimal progress, near-zero clearance
    min_clearance = min(float(f.get("nearest_clearance", f.get("path_clearance", 99.0))) for f in frames)
    if len(frames) < 30 and max_distance < 0.60 and min_clearance < 0.12:
        reasons.append("wall_hit")
        details["wall_hit"] = True
        details["min_clearance"] = round(min_clearance, 4)
        return {"failure_modes": reasons, "details": details}

    # Early crash: short run with very low clearance
    if len(frames) < 50 and max_distance < 1.20 and min_clearance < 0.18:
        reasons.append("early_crash")
        details["early_crash"] = True
        details["min_clearance"] = round(min_clearance, 4)
    if max_distance < 0.30 * target_distance:
        reasons.append("minimal_progress")
        details["max_distance"] = max_distance

    # Collision check
    clearances = [float(f.get("nearest_clearance", f.get("path_clearance", 99.0))) for f in frames]
    near_collision = sum(1 for c in clearances if c < 0.12)
    if near_collision > 3:
        reasons.append("near_collision")
        details["near_collision_frames"] = near_collision

    # Stuck check
    distances = [float(f.get("stage_distance", 0.0)) for f in frames]
    final_distance = distances[-1] if distances else 0.0
    progress_retreat = max(0.0, max_distance - final_distance)
    details["final_distance"] = round(final_distance, 4)
    details["progress_retreat"] = round(progress_retreat, 4)
    if progress_retreat > 0.45 and max_distance > 0.80:
        reasons.append("retreated_after_progress")
    if len(distances) >= 60:
        start_dist = max(distances[:10]) if distances[:10] else 0.0
        end_dist = max(distances[-50:]) if distances[-50:] else 0.0
        progress = end_dist - start_dist
        details["late_progress"] = round(progress, 4)
        if progress < 0.20 and max_distance < 0.60 * target_distance:
            reasons.append("stuck")
            details["stuck_progress"] = round(progress, 4)

    # DWA fallback degradation check
    dwa_used = [f for f in frames if f.get("dwa", {}).get("dwa_used") is True]
    dwa_total = sum(1 for f in frames if "dwa" in f)
    if dwa_total > 20:
        fallback_rate = 1.0 - len(dwa_used) / dwa_total
        details["dwa_fallback_rate"] = round(fallback_rate, 4)
        if fallback_rate > 0.80:
            reasons.append("dwa_degraded")
        elif fallback_rate > 0.50:
            reasons.append("dwa_partial")

    # Revisiting loop check
    visited = [f.get("memory_visited", 0) for f in frames]
    if len(visited) >= 30:
        early = sum(visited[:10]) / max(1, len([v for v in visited[:10] if v > 0]))
        late = sum(visited[-20:]) / max(1, len([v for v in visited[-20:] if v > 0]))
        details["visited_early"] = round(early, 1)
        details["visited_late"] = round(late, 1)
        if late > early * 2.5 and early > 5:
            reasons.append("revisiting_loop")

    # Grid occupancy high
    grid_pcts = [f.get("grid_map", {}).get("occupancy_pct", 0.0) for f in frames if "grid_map" in f]
    if grid_pcts:
        avg_occ = sum(grid_pcts) / len(grid_pcts)
        details["avg_occupancy_pct"] = round(avg_occ, 2)
        if avg_occ > 60.0:
            reasons.append("high_occupancy")

    # Oscillation
    angles = [float(f.get("selected_angle", 0.0)) for f in frames]
    yaws = [abs(float(f.get("yaw_delta", 0.0))) for f in frames]
    if yaws:
        details["max_abs_yaw"] = round(max(yaws), 4)
        details["final_abs_yaw"] = round(yaws[-1], 4)
        if max(yaws) > 1.45 or yaws[-1] > 1.20:
            reasons.append("yaw_turnback")
    if len(angles) >= 20:
        oscillation = sum(abs(angles[i] - angles[i - 1]) for i in range(1, len(angles)))
        details["oscillation"] = round(oscillation, 4)
        if oscillation > len(angles) * 0.15:
            reasons.append("excessive_oscillation")

    if not reasons:
        reasons.append("partial_or_unknown")

    return {"failure_modes": reasons, "details": details}


def score_profile_for_phase(score_phase: str) -> dict:
    if score_phase == "explore":
        return {
            "min_progress_penalty": 3200.0,
            "low_progress_penalty": 900.0,
            "wall_hit_penalty": 15000.0,
            "early_crash_penalty": 5200.0,
            "progress_gain": 190.0,
            "completed_bonus": 500.0,
            "caution_reward": 14.0,
            "clear_reward": 1.5,
            "planner_score_gain": 8.0,
            "dwa_use_gain": 24.0,
            "dwa_frame_gain": 4.0,
            "danger_frame_penalty": 38.0,
            "danger_growth": 0.07,
            "soft_clearance_penalty": 260.0,
            "stopped_penalty": 2.0,
            "oscillation_penalty": 7.0,
            "retreat_penalty": 260.0,
            "yaw_penalty": 85.0,
        }
    if score_phase == "refine":
        return {
            "min_progress_penalty": 5600.0,
            "low_progress_penalty": 1400.0,
            "wall_hit_penalty": 15000.0,
            "early_crash_penalty": 6000.0,
            "progress_gain": 155.0,
            "completed_bonus": 650.0,
            "caution_reward": 10.0,
            "clear_reward": 2.5,
            "planner_score_gain": 6.0,
            "dwa_use_gain": 18.0,
            "dwa_frame_gain": 3.5,
            "danger_frame_penalty": 58.0,
            "danger_growth": 0.11,
            "soft_clearance_penalty": 560.0,
            "stopped_penalty": 3.0,
            "oscillation_penalty": 10.0,
            "retreat_penalty": 520.0,
            "yaw_penalty": 150.0,
        }
    return {
        "min_progress_penalty": 8000.0,
        "low_progress_penalty": 2000.0,
        "wall_hit_penalty": 15000.0,
        "early_crash_penalty": 6000.0,
        "progress_gain": 130.0,
        "completed_bonus": 650.0,
        "caution_reward": 6.0,
        "clear_reward": 3.0,
        "planner_score_gain": 5.0,
        "dwa_use_gain": 15.0,
        "dwa_frame_gain": 3.0,
        "danger_frame_penalty": 80.0,
        "danger_growth": 0.15,
        "soft_clearance_penalty": 900.0,
        "stopped_penalty": 4.0,
        "oscillation_penalty": 12.0,
        "retreat_penalty": 850.0,
        "yaw_penalty": 240.0,
    }


def score_trial(returncode: int, log_path: Path, timeout_hit: bool,
                target_distance: float, score_phase: str = "converge") -> dict:
    """Compute a multi-term score from a planner.jsonl file.

    Phase 4 additions:
      - DWA vs fallback utilisation
      - Grid-map occupancy penalty
      - DWA trajectory quality
      - Revisiting penalty
      - Escape / stuck penalty
    """
    frames = [r for r in read_jsonl(log_path) if r.get("event") == "frame"]
    if not frames:
        return {
            "score": -10000.0,
            "score_phase": score_phase,
            "completed": False,
            "reason": "no_planner_frames",
            "frame_count": 0,
            "max_distance": 0.0,
            "min_clearance": 0.0,
        }

    # Basic metrics
    distances = [float(f.get("stage_distance", 0.0)) for f in frames]
    clearances = [float(f.get("nearest_clearance", f.get("path_clearance", 0.0))) for f in frames]
    nearest_obstacles = [float(f.get("nearest_obstacle", 99.0)) for f in frames]
    angles = [float(f.get("selected_angle", 0.0)) for f in frames]
    cmd_v = [float(f.get("cmd", {}).get("linear_x", 0.0)) for f in frames]
    planner_scores = [float(f.get("score", 0.0)) for f in frames]

    max_distance = max(distances) if distances else 0.0
    final_distance = distances[-1] if distances else 0.0
    progress_retreat = max(0.0, max_distance - final_distance)
    min_clearance = min(clearances) if clearances else 0.0
    min_obstacle = min(nearest_obstacles) if nearest_obstacles else 0.0
    yaw_values = [abs(float(f.get("yaw_delta", 0.0))) for f in frames]
    max_abs_yaw = max(yaw_values) if yaw_values else 0.0
    final_abs_yaw = yaw_values[-1] if yaw_values else 0.0

    # Frame classification
    danger_frames = sum(1 for c in clearances if c < 0.30)
    caution_frames = sum(1 for c in clearances if 0.30 <= c <= 0.40)
    clear_frames = sum(1 for c in clearances if c > 0.75)
    stopped_frames = sum(1 for v in cmd_v if abs(v) < 0.015)
    backward_frames = sum(1 for v in cmd_v if v < -0.001)
    oscillation = sum(abs(angles[i] - angles[i - 1]) for i in range(1, len(angles)))
    avg_planner_score = sum(planner_scores) / max(1, len(planner_scores))

    # Phase 4: DWA metrics
    dwa_frames = [f for f in frames if "dwa" in f]
    dwa_used = [f for f in dwa_frames if f.get("dwa", {}).get("dwa_used") is True]
    dwa_count = len(dwa_frames)
    dwa_used_count = len(dwa_used)
    dwa_fallback_rate = 1.0 - dwa_used_count / max(1, dwa_count) if dwa_count > 0 else 1.0
    dwa_avg_cost = 0.0
    if dwa_used:
        costs = [f["dwa"]["selected"]["cost"] for f in dwa_used if "selected" in f.get("dwa", {})]
        dwa_avg_cost = sum(costs) / max(1, len(costs))

    # Phase 4: Grid-map occupancy
    grid_occs = [f.get("grid_map", {}).get("occupancy_pct", 0.0) for f in frames if "grid_map" in f]
    avg_grid_occ = sum(grid_occs) / max(1, len(grid_occs))

    # Phase 4: Revisiting
    visited_counts = [f.get("memory_visited", 0) for f in frames]
    late_visited = sum(visited_counts[-20:]) / max(1, len([v for v in visited_counts[-20:] if v > 0])) if len(visited_counts) >= 20 else 0
    early_visited = sum(visited_counts[:10]) / max(1, len([v for v in visited_counts[:10] if v > 0])) if len(visited_counts) >= 10 else 0
    revisiting_ratio = late_visited / max(1.0, early_visited) if early_visited > 0 else 1.0

    # Completion check
    completed = (returncode == 0
                 and final_distance >= 0.92 * target_distance
                 and progress_retreat <= 0.35
                 and danger_frames == 0)

    # ---- Wall-hit detection ----
    # A trial that ends very quickly with extremely low clearance almost
    # certainly crashed into a wall at startup.
    wall_hit = (
        len(frames) < 30
        and max_distance < 0.60
        and min_clearance < 0.12
    )
    early_crash = (
        len(frames) < 50
        and max_distance < 1.20
        and min_clearance < 0.18
    )
    critical_clearance_frames = sum(1 for c in clearances if c < 0.12)

    # ---- Score assembly ----
    profile = score_profile_for_phase(score_phase)
    score = 0.0

    # Minimum-progress gate: trials that barely move cannot get positive score.
    if max_distance < 0.50:
        score -= profile["min_progress_penalty"]
    elif max_distance < 1.00:
        score -= profile["low_progress_penalty"] * (1.00 - max_distance)
    if final_distance < 0.30 * target_distance:
        score -= 1200.0 * (0.30 * target_distance - final_distance)
    elif final_distance < 0.55 * target_distance:
        score -= 320.0 * (0.55 * target_distance - final_distance)

    # Wall-hit instant death penalty
    if wall_hit:
        score -= profile["wall_hit_penalty"]
    elif early_crash:
        score -= profile["early_crash_penalty"]

    # Critical clearance: touching obstacles
    score -= 500.0 * critical_clearance_frames
    if min_clearance < 0.08:
        score -= 5000.0 * (0.08 - min_clearance)

    # Progress reward
    effective_distance = 0.35 * max_distance + 0.65 * final_distance
    score += profile["progress_gain"] * effective_distance
    score += profile["completed_bonus"] if completed else 0.0

    # Clearance rewards (diminished — collision avoidance is now the priority)
    score += profile["caution_reward"] * min(caution_frames, 80)
    score += profile["clear_reward"] * min(clear_frames, 80)
    score += 12.0 * max(0.0, min_clearance - 0.30)
    score += profile["planner_score_gain"] * avg_planner_score

    # DWA quality
    if dwa_count > 10:
        score += profile["dwa_use_gain"] * (1.0 - dwa_fallback_rate)
        score -= 4.0 * max(0.0, dwa_avg_cost)
        score += profile["dwa_frame_gain"] * min(dwa_used_count, 80)

    # Penalties — exponential for danger frames to strongly prefer zero collisions
    if danger_frames > 0:
        score -= profile["danger_frame_penalty"] * danger_frames * (
            1.0 + profile["danger_growth"] * danger_frames
        )
    score -= 35.0 * backward_frames
    score -= profile["stopped_penalty"] * stopped_frames
    score -= profile["oscillation_penalty"] * oscillation
    score -= profile["retreat_penalty"] * progress_retreat
    if max_abs_yaw > 0.95:
        score -= profile["yaw_penalty"] * (max_abs_yaw - 0.95) ** 2
    if final_abs_yaw > 0.75:
        score -= 0.6 * profile["yaw_penalty"] * (final_abs_yaw - 0.75) ** 2

    # Phase 4 penalties
    if avg_grid_occ > 40.0:
        score -= 2.0 * (avg_grid_occ - 40.0)
    if revisiting_ratio > 2.0:
        score -= 15.0 * (revisiting_ratio - 2.0)
    if dwa_fallback_rate > 0.70 and dwa_count > 15:
        score -= 25.0 * (dwa_fallback_rate - 0.70)

    # Timeout / clearance penalties
    if timeout_hit:
        score -= 180.0
    if returncode != 0:
        score -= 220.0
    if min_clearance < 0.30:
        score -= profile["soft_clearance_penalty"] * (0.30 - min_clearance)
    if min_obstacle < 0.08:
        score -= 2000.0

    return {
        "score": round(score, 4),
        "score_phase": score_phase,
        "completed": completed,
        "reason": "ok" if completed else "partial_or_failed",
        "frame_count": len(frames),
        "max_distance": round(max_distance, 4),
        "final_distance": round(final_distance, 4),
        "progress_retreat": round(progress_retreat, 4),
        "max_abs_yaw": round(max_abs_yaw, 4),
        "final_abs_yaw": round(final_abs_yaw, 4),
        "min_clearance": round(min_clearance, 4),
        "min_obstacle": round(min_obstacle, 4),
        "danger_frames": danger_frames,
        "caution_frames": caution_frames,
        "clear_frames": clear_frames,
        "stopped_frames": stopped_frames,
        "backward_frames": backward_frames,
        "oscillation": round(oscillation, 4),
        "avg_planner_score": round(avg_planner_score, 4),
        "dwa_count": dwa_count,
        "dwa_used_count": dwa_used_count,
        "dwa_fallback_rate": round(dwa_fallback_rate, 4),
        "dwa_avg_cost": round(dwa_avg_cost, 4),
        "avg_grid_occupancy_pct": round(avg_grid_occ, 2),
        "revisiting_ratio": round(revisiting_ratio, 2),
        "returncode": returncode,
        "timeout_hit": timeout_hit,
        "wall_hit": wall_hit,
        "early_crash": early_crash,
        "critical_clearance_frames": critical_clearance_frames,
    }


# ============================================================================
# Trial runner
# ============================================================================


def _cleanup_orphan_ros() -> None:
    """Kill leftover roslaunch / roscore / MuJoCo processes from a previous trial."""
    import subprocess as _sp
    for pattern in ["roslaunch", "rosmaster", "roscore", "mujoco"]:
        _sp.run(
            ["pkill", "-f", pattern],
            stdout=_sp.DEVNULL,
            stderr=_sp.DEVNULL,
            check=False,
        )
    time.sleep(0.8)


def run_trial(
    trial_idx: int,
    params: Dict[str, float],
    args,
    scene_script: Path,
    out_dir: Path,
    optimizer_meta: dict = None,
) -> Tuple[dict, Path]:
    optimizer_meta = optimizer_meta or {}
    # Always clean up leftover ROS processes before the first trial.
    # --pre-kill extends this to every trial for extra safety.
    if trial_idx == 0 or args.pre_kill:
        _cleanup_orphan_ros()
    trial_dir = out_dir / ("trial_%03d" % trial_idx)
    trial_dir.mkdir(parents=True, exist_ok=True)
    planner_log = trial_dir / "planner.jsonl"
    stdout_log = trial_dir / "stdout.log"
    params_path = trial_dir / "params.json"
    summary_path = trial_dir / "summary.json"
    failure_path = trial_dir / "failure_reason.json"

    params_path.write_text(json.dumps(params, indent=2, sort_keys=True), encoding="utf-8")

    cmd = [
        sys.executable,
        str(scene_script),
        "--seed",
        str(args.seed),
        "--planner-log",
        str(planner_log),
        "--settle-time",
        str(args.settle_time),
        "--params-file",
        str(params_path),
    ]
    if args.startup_scan:
        cmd.append("--startup-scan")
    cmd.extend(ros_private_args(params))

    timeout_hit = False
    monitor_reason = "not_started"
    start = time.time()
    with stdout_log.open("w", encoding="utf-8") as stdout:
        stdout.write("COMMAND: %s\n" % " ".join(cmd))
        stdout.flush()
        returncode, timeout_hit, monitor_reason = run_monitored_trial_process(
            cmd=cmd,
            cwd=str(scene_script.parents[1]),
            stdout=stdout,
            planner_log=planner_log,
            args=args,
        )

    summary = score_trial(
        returncode,
        planner_log,
        timeout_hit,
        args.target_distance,
        optimizer_meta.get("score_phase", "converge"),
    )

    # Phase 4: failure analysis
    frames = [r for r in read_jsonl(planner_log) if r.get("event") == "frame"]
    failure = analyze_failure(frames, returncode, timeout_hit,
                              args.target_distance, summary["max_distance"])
    failure_path.write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")

    summary.update({
        "trial": trial_idx,
        "duration_wall": round(time.time() - start, 3),
        "params": params,
        "planner_log": str(planner_log),
        "stdout_log": str(stdout_log),
        "failure_reason": str(failure_path),
        "failure_modes": failure["failure_modes"],
        "monitor_reason": monitor_reason,
        "optimizer_phase": optimizer_meta.get("name", "manual"),
        "candidate_source": optimizer_meta.get("candidate_source", "manual"),
        "explore_scale": optimizer_meta.get("explore_scale", 0.0),
    })
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary, trial_dir


# ============================================================================
# Parameter loading
# ============================================================================


def load_start_params(path: str) -> Dict[str, float]:
    if not path:
        return dict(BASE_PARAMS)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    params = dict(BASE_PARAMS)
    params.update(data)
    return {name: normalize_param(name, value)
            for name, value in params.items() if name in BASE_PARAMS}


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 4: Auto-optimize DWA + gap-follow obstacle planner params"
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--iterations", type=int, default=12, help="trial count")
    parser.add_argument("--seed", type=int, default=0, help="simulation seed")
    parser.add_argument("--random-seed", type=int, default=20260507, help="optimizer RNG seed")
    parser.add_argument("--timeout", type=float, default=260.0, help="seconds per trial")
    parser.add_argument("--settle-time", type=float, default=18.0, help="scene1 startup settle seconds")
    parser.add_argument("--startup-frame-timeout", type=float, default=70.0,
                        help="restart trial if planner frames never appear within this many seconds")
    parser.add_argument("--sim-stale-timeout", type=float, default=8.0,
                        help="restart trial if planner frames stop updating after this many seconds")
    parser.add_argument("--watchdog-poll", type=float, default=1.0,
                        help="seconds between watchdog checks")
    parser.add_argument("--kill-grace", type=float, default=5.0,
                        help="seconds to wait after SIGTERM before force killing a stale trial")
    parser.add_argument("--disable-watchdog", action="store_true",
                        help="disable stale planner-frame restart detection")
    parser.add_argument("--target-distance", type=float, default=5.0, help="obstacle-stage target distance")
    parser.add_argument("--output", default=str(script_dir.parent / "data" / "auto_optimize"),
                        help="output directory")
    parser.add_argument("--scene-script", default=str(script_dir / "scene1_patrol.py"),
                        help="scene1 script path")
    parser.add_argument("--start-params", default="", help="optional JSON params to start from")
    parser.add_argument("--optimizer", choices=("hybrid", "gaussian"), default="hybrid",
                        help="hybrid uses broad early exploration plus empirical quasi-Newton refinement")
    parser.add_argument("--score-profile", choices=("staged", "safety"), default="staged",
                        help="staged relaxes soft penalties early, safety uses final conservative scoring")
    parser.add_argument("--aggressive-fraction", type=float, default=0.38,
                        help="fraction of trials reserved for broad/aggressive exploration")
    parser.add_argument("--early-explore-scale", type=float, default=2.4,
                        help="mutation scale multiplier during early exploration")
    parser.add_argument("--final-explore-scale", type=float, default=0.35,
                        help="minimum mutation scale multiplier during convergence")
    parser.add_argument("--startup-scan", action="store_true", help="enable scene1 startup lidar scan")
    parser.add_argument("--dry-run", action="store_true",
                        help="write initial params and exit without sim")
    parser.add_argument("--single-trial", action="store_true",
                        help="run one trial with current best params and exit")
    parser.add_argument("--pre-kill", action="store_true",
                        help="pkill leftover roslaunch/roscore/mujoco before each trial")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_script = Path(args.scene_script).resolve()
    rng = random.Random(args.random_seed)
    best_params = load_start_params(args.start_params)
    best_summary = {"score": -float("inf")}
    history = []

    (out_dir / "optimizer_config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if args.dry_run:
        dry = out_dir / "dry_run_params.json"
        dry.write_text(json.dumps(best_params, indent=2, sort_keys=True), encoding="utf-8")
        aggressive = out_dir / "dry_run_aggressive_probe_params.json"
        aggressive.write_text(
            json.dumps(blend_params(best_params, AGGRESSIVE_TARGET_PARAMS, 0.75),
                       indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print("dry-run params:", dry)
        print("dry-run aggressive probe:", aggressive)
        return 0

    if args.single_trial:
        phase = optimizer_phase(0, 1, args)
        phase["candidate_source"] = "single_trial"
        phase["score_phase"] = "converge"
        summary, trial_dir = run_trial(0, best_params, args, scene_script, out_dir, phase)
        print("single-trial score=%.2f dir=%s" % (summary["score"], trial_dir))
        if summary.get("failure_modes"):
            print("failure modes:", summary["failure_modes"])
        return 0

    for trial_idx in range(args.iterations):
        phase = optimizer_phase(trial_idx, args.iterations, args)
        if trial_idx == 0:
            params = dict(best_params)
            candidate_source = "baseline"
        else:
            params, candidate_source = propose_params(best_params, history, rng, phase, args.optimizer)
        phase["candidate_source"] = candidate_source

        try:
            summary, trial_dir = run_trial(trial_idx, params, args, scene_script, out_dir, phase)
        except Exception as exc:
            print("trial=%03d  CRASHED: %s" % (trial_idx, exc))
            import traceback
            traceback.print_exc()
            # Record a sentinel entry so history stays aligned
            summary = {
                "score": -20000.0,
                "completed": False,
                "reason": "trial_crash",
                "frame_count": 0,
                "max_distance": 0.0,
                "min_clearance": 0.0,
                "failure_modes": ["trial_crash"],
                "trial": trial_idx,
                "duration_wall": 0.0,
                "params": params,
                "monitor_reason": "exception",
                "optimizer_phase": phase.get("name", "unknown"),
                "candidate_source": candidate_source,
                "score_phase": phase.get("score_phase", "converge"),
                "crash": str(exc),
            }
            crash_dir = out_dir / ("trial_%03d" % trial_idx)
            crash_dir.mkdir(parents=True, exist_ok=True)
            (crash_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
            (crash_dir / "crash_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            trial_dir = crash_dir

        history.append(summary)
        print(
            "trial=%03d  phase=%s/%s  source=%s  score=%.2f  completed=%s  "
            "max=%.2f final=%.2f yaw=%.2f clear=%.2f  fallback=%.1f%%  failure=%s  dir=%s"
            % (
                trial_idx,
                summary.get("optimizer_phase", phase.get("name", "?")),
                summary.get("score_phase", phase.get("score_phase", "?")),
                summary.get("candidate_source", candidate_source),
                summary["score"],
                summary["completed"],
                summary["max_distance"],
                summary.get("final_distance", summary["max_distance"]),
                summary.get("final_abs_yaw", 0.0),
                summary["min_clearance"],
                100.0 * summary.get("dwa_fallback_rate", 0.0),
                ",".join(summary.get("failure_modes", [])),
                trial_dir,
            )
        )

        if summary["score"] > best_summary["score"]:
            best_summary = summary
            best_params = dict(params)
            (out_dir / "best_params.json").write_text(
                json.dumps(best_params, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            (out_dir / "best_summary.json").write_text(
                json.dumps(best_summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )

        (out_dir / "history.json").write_text(
            json.dumps(history, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (out_dir / "optimizer_state.json").write_text(
            json.dumps({
                "best_score": best_summary["score"],
                "best_params": best_params,
                "last_phase": phase,
                "trial_count": len(history),
            }, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    print("\n=== Best params ===")
    for name, value in sorted(best_params.items()):
        print("  %s=%s" % (name, value))
    print("best score=%.2f" % best_summary["score"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
