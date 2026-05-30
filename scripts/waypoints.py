#!/usr/bin/env python3.8
"""Semantic scene-1 directives for perception-driven patrol.

This file intentionally avoids absolute route coordinates. It only describes
task order, AprilTag meaning, relative travel thresholds, and controller choice.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class TagAnchor:
    tag_id: int
    name: str
    stage_hint: str
    preferred_heading: float = 0.0


@dataclass(frozen=True)
class StageDirective:
    name: str
    controller: str
    nominal_distance: float
    timeout: float
    speed: float
    target_tags: Tuple[int, ...] = ()
    completion_tags: Tuple[int, ...] = ()
    corridor_bias: float = 0.0
    description: str = ""
    use_dwa: bool = False  # Phase 2: enable DWA local planner for this stage


@dataclass(frozen=True)
class PatrolDirective:
    allowed_scope: str = "src/craic_task_template"
    allow_ground_truth: bool = False
    allow_absolute_world_route: bool = False
    use_apriltag_anchors: bool = True
    use_lidar_local_map: bool = True
    use_relative_odom: bool = True


GLOBAL_DIRECTIVE = PatrolDirective()


APRILTAG_ANCHORS: Dict[int, TagAnchor] = {
    1: TagAnchor(1, "slope_entry", "slope"),
    2: TagAnchor(2, "slope_mid", "slope"),
    3: TagAnchor(3, "slope_exit", "slope"),
    4: TagAnchor(4, "stairs_entry", "stairs"),
    5: TagAnchor(5, "stairs_mid", "stairs"),
    6: TagAnchor(6, "stairs_exit", "goal"),
}


FORWARD_STAGES = [
    StageDirective(
        name="leave_start",
        controller="mpc",
        nominal_distance=1.8,
        timeout=35.0,
        speed=0.18,
        corridor_bias=0.15,
        description="Leave the start area by following the clearest lidar corridor.",
    ),
    StageDirective(
        name="obstacle",
        controller="mpc",
        nominal_distance=5.0,
        timeout=85.0,
        speed=0.20,
        description="Cross the obstacle area with local lidar avoidance.",
        use_dwa=True,
    ),
    StageDirective(
        name="operator_1",
        controller="mpc",
        nominal_distance=0.8,
        timeout=35.0,
        speed=0.12,
        description="Approach and press the first operation panel.",
    ),
    StageDirective(
        name="terrain",
        controller="amp",
        nominal_distance=5.8,
        timeout=90.0,
        speed=0.16,
        target_tags=(1,),
        completion_tags=(1,),
        corridor_bias=-0.05,
        description="Traverse the complex terrain until the slope anchor is found or distance is reached.",
    ),
    StageDirective(
        name="operator_2",
        controller="mpc",
        nominal_distance=0.8,
        timeout=35.0,
        speed=0.12,
        description="Approach and press the second operation panel.",
    ),
    StageDirective(
        name="slope",
        controller="amp",
        nominal_distance=5.4,
        timeout=80.0,
        speed=0.13,
        target_tags=(1, 2, 3),
        completion_tags=(3, 4),
        description="Follow the local corridor over the slope anchors.",
    ),
    StageDirective(
        name="stairs",
        controller="amp",
        nominal_distance=3.0,
        timeout=95.0,
        speed=0.09,
        target_tags=(4, 5, 6),
        completion_tags=(6,),
        description="Climb the stair section at low speed using lidar corridor checks.",
    ),
    StageDirective(
        name="goal",
        controller="mpc",
        nominal_distance=1.2,
        timeout=40.0,
        speed=0.12,
        completion_tags=(6,),
        description="Enter the finish area and stop.",
    ),
]


SEED0_OBSTACLE_STAGES = [
    StageDirective(
        name="leave_start",
        controller="mpc",
        nominal_distance=1.8,
        timeout=35.0,
        speed=0.16,
        corridor_bias=0.15,
        description="Leave the start area with lidar corridor following.",
    ),
    StageDirective(
        name="obstacle",
        controller="mpc",
        nominal_distance=5.0,
        timeout=95.0,
        speed=0.16,
        description="Cross the seed0 obstacle area using robot-local lidar gap following.",
        use_dwa=True,
    ),
]


REVERSE_STAGES = [
    StageDirective("reverse_stairs", "amp", 3.0, 95.0, 0.08, target_tags=(6, 5, 4), completion_tags=(4,)),
    StageDirective("reverse_slope", "amp", 5.4, 80.0, 0.12, target_tags=(3, 2, 1), completion_tags=(1,)),
    StageDirective("reverse_terrain", "amp", 5.8, 90.0, 0.14),
    StageDirective("reverse_obstacle", "mpc", 5.0, 85.0, 0.18),
    StageDirective("return_start", "mpc", 1.8, 40.0, 0.14),
]


BUTTON_PANELS = {
    "operator_1": {
        "arm": "right",
        "approach_distance": 0.55,
        "approach_timeout": 18.0,
    },
    "operator_2": {
        "arm": "left",
        "approach_distance": 0.55,
        "approach_timeout": 18.0,
    },
}
