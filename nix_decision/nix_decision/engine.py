from __future__ import annotations

import time

from .clock import SystemClock
from .evaluator import Evaluator
from .state import WorldState


class NixDecisionEngine:

    def __init__(
        self,
        clock=None,
        evaluator=None,
        interval=2.0,
        on_decision_change=None,
    ):

        self.clock = clock or SystemClock()

        self.world = WorldState()

        self.evaluator = evaluator or Evaluator()

        self.interval = interval

        self.on_decision_change = on_decision_change

        self.current_decision = None

        self.running = False

    # ---------------------------------------------------------
    # OBSERVATION INPUT
    # ---------------------------------------------------------

    def ingest(self, observation):

        self.world.ingest(observation)

        return self.evaluate()

    # ---------------------------------------------------------
    # COGNITION
    # ---------------------------------------------------------

    def evaluate(self):

        now = self.clock.now()

        new_decision = self.evaluator.evaluate(
            self.world,
            now,
        )

        # First decision.
        if self.current_decision is None:

            previous = None

            self.current_decision = new_decision

            self._decision_changed(
                previous,
                new_decision,
            )

            return new_decision

        # Determine whether cognition actually changed.
        if (
            new_decision.identity()
            != self.current_decision.identity()
        ):

            previous = self.current_decision

            self.current_decision = new_decision

            self._decision_changed(
                previous,
                new_decision,
            )

        return self.current_decision

    # ---------------------------------------------------------
    # DECISION TRANSITION
    # ---------------------------------------------------------

    def _decision_changed(
        self,
        previous,
        current,
    ):

        if self.on_decision_change:

            self.on_decision_change(
                previous,
                current,
            )

    # ---------------------------------------------------------
    # CONTINUOUS COGNITION LOOP
    # ---------------------------------------------------------

    def run(self):

        self.running = True

        while self.running:

            self.evaluate()

            time.sleep(self.interval)

    def stop(self):

        self.running = False
