from __future__ import annotations

from .decisions import Decision


class Evaluator:
    """
    Cheap continuous cognition.

    This is NOT the expensive Nix Core.

    Its job is to evaluate relationships between current
    observations, historical context, time, and world state.
    """

    def evaluate(self, world, now) -> Decision:

        phone = world.phone_location
        presence = world.presence

        # ---------------------------------------------------------
        # No meaningful presence information.
        # ---------------------------------------------------------

        if not phone.value and not presence.room:

            return Decision(
                state="MONITORING",
                action="NONE",
                attention="LOW",
                reason="Insufficient presence context.",
                timestamp=now,
            )

        # ---------------------------------------------------------
        # Determine whether phone recently reported arrival.
        # ---------------------------------------------------------

        recent_arrival = self._recent_event(
            world,
            kind="phone.location",
            seconds=5 * 60,
            now=now,
            value="home",
            field="location",
        )

        arrived_home = phone.value == "home"

        # ---------------------------------------------------------
        # Look for recently completed relevant activity.
        # ---------------------------------------------------------

        recent_calendar_event = None

        for event in reversed(world.calendar_events):

            if event.end is None:
                continue

            age = (
                now - event.end
            ).total_seconds()

            if 0 <= age <= 90 * 60:

                recent_calendar_event = event
                break

        # ---------------------------------------------------------
        # RELATIONSHIP DETECTION
        #
        # This is the important part.
        #
        # We aren't doing:
        #
        #     if home: speak
        #
        # We are asking whether multiple observations
        # form a meaningful relationship.
        # ---------------------------------------------------------

        if (
            arrived_home
            and recent_arrival
            and recent_calendar_event
        ):

            return Decision(
                state="CONSIDER",
                action="EVALUATE_CONTEXT",
                attention="MEDIUM",
                reason=(
                    f"Home arrival occurred shortly after "
                    f"'{recent_calendar_event.name}' ended."
                ),
                timestamp=now,
            )

        # ---------------------------------------------------------
        # Nothing currently significant.
        # ---------------------------------------------------------

        if arrived_home:

            return Decision(
                state="MONITORING",
                action="NONE",
                attention="LOW",
                reason="User appears to be home.",
                timestamp=now,
            )

        return Decision(
            state="MONITORING",
            action="NONE",
            attention="LOW",
            reason="No significant relationship detected.",
            timestamp=now,
        )

    @staticmethod
    def _recent_event(
        world,
        kind,
        seconds,
        now,
        value=None,
        field=None,
    ):

        for event in reversed(world.recent_events):

            if event["kind"] != kind:
                continue

            age = (
                now - event["timestamp"]
            ).total_seconds()

            if not 0 <= age <= seconds:
                continue

            if value is not None and field is not None:

                if event["data"].get(field) != value:
                    continue

            return True

        return False
