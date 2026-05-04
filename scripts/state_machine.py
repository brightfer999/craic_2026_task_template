#!/usr/bin/env python3.8

import time
from dataclasses import dataclass
from typing import Callable, List, Tuple

import rospy

from controller_manager import ControllerManager
from localization import Localizer
from terrain_traverser import TerrainTraverser
from waypoints import FORWARD_STAGES, GLOBAL_DIRECTIVE, REVERSE_STAGES, StageDirective


@dataclass
class PatrolResult:
    completed: bool
    elapsed_time: float
    failed_states: List[str]
    fall_count: int
    obstacle_hit_count: int
    pose_is_valid: bool
    anchor_is_valid: bool


class PatrolFSM:
    """Finite-state patrol runner for scene 1."""

    WATCHDOG_SECONDS = 9.5 * 60.0

    def __init__(self, localizer: Localizer, controller: ControllerManager):
        self.localizer = localizer
        self.controller = controller
        self.traverser = TerrainTraverser(localizer, controller)
        self._start_time = time.time()
        self._failed_states = []
        self._fall_count = 0
        self._validate_directive()

    def run(self) -> PatrolResult:
        rospy.loginfo("PatrolFSM: running forward task with hybrid perception directives")
        completed = self._run_stage_directives(FORWARD_STAGES, reverse=False)
        return self._result(completed)

    def run_reverse(self) -> PatrolResult:
        rospy.loginfo("PatrolFSM: running reverse extra task with relative directives")
        completed = self._run_stage_directives(REVERSE_STAGES, reverse=True)
        return self._result(completed)

    def should_attempt_extra(self, result: PatrolResult) -> bool:
        return (
            result.completed
            and result.elapsed_time < 6.0 * 60.0
            and result.fall_count == 0
            and result.pose_is_valid
            and result.obstacle_hit_count <= 2
        )

    def _run_stage_directives(self, stages: List[StageDirective], reverse: bool) -> bool:
        states: List[Tuple[str, Callable[[], bool]]] = [
            (stage.name, lambda stage=stage: self.traverser.run_stage(stage, reverse=reverse))
            for stage in stages
        ]
        return self._run_states(states)

    def _run_states(self, states: List[Tuple[str, Callable[[], bool]]]) -> bool:
        for state_name, action in states:
            if self._watchdog_expired():
                rospy.logwarn("PatrolFSM watchdog expired before %s", state_name)
                return False

            rospy.loginfo("PatrolFSM state: %s", state_name)
            try:
                ok = bool(action())
            except Exception as exc:
                rospy.logerr("PatrolFSM state %s failed with exception: %s", state_name, exc)
                ok = False

            if not ok:
                self._failed_states.append(state_name)
                rospy.logwarn("PatrolFSM state failed, continuing: %s", state_name)

        return len(self._failed_states) == 0

    def _watchdog_expired(self) -> bool:
        return time.time() - self._start_time > self.WATCHDOG_SECONDS

    def _result(self, completed: bool) -> PatrolResult:
        self.traverser.stop(0.5)
        return PatrolResult(
            completed=completed,
            elapsed_time=time.time() - self._start_time,
            failed_states=list(self._failed_states),
            fall_count=self._fall_count,
            obstacle_hit_count=self.traverser.obstacle_hit_count,
            pose_is_valid=self.localizer.pose_is_valid,
            anchor_is_valid=self.localizer.anchor_is_valid,
        )

    @staticmethod
    def _validate_directive():
        if GLOBAL_DIRECTIVE.allow_ground_truth:
            raise RuntimeError("Ground truth is disabled for scene1 patrol")
        if GLOBAL_DIRECTIVE.allow_absolute_world_route:
            raise RuntimeError("Absolute world route modeling is disabled")
