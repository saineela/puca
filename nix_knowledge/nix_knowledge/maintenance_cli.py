from __future__ import annotations

import argparse
import json
import sys
import time

from .engine import KnowledgeEngine
from .maintenance import TemporalMaintenance

DEFAULT_DB = "knowledge.db"
DEFAULT_TZ = "America/Chicago"


def run_once(
    database: str,
    timezone: str,
) -> dict:
    engine = KnowledgeEngine(database)

    try:
        maintenance = TemporalMaintenance(engine, timezone=timezone)
        return maintenance.expire_due()
    finally:
        engine.close()


def loop(
    database: str,
    timezone: str,
) -> int:
    """
    Midnight runner: expire all context-stale events exactly when the
    calendar day rolls over, then sleep until the next midnight.
    """
    engine = KnowledgeEngine(database)

    try:
        maintenance = TemporalMaintenance(engine, timezone=timezone)

        print(
            f"nix-knowledge maintenance looping "
            f"(db={database}, tz={timezone}); Ctrl-C to stop"
        )

        while True:
            summary = maintenance.expire_due()

            print(
                json.dumps(summary, ensure_ascii=False, default=str)
            )

            nap = maintenance.seconds_until_next_midnight()

            print(
                f"next run at local midnight in {nap / 3600:.2f}h"
            )

            time.sleep(nap + 1)  # +1s so we wake just after midnight

    except KeyboardInterrupt:
        print()
        return 0

    finally:
        engine.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nix-knowledge-maintenance",
        description=(
            "Expire calendar events whose temporal context is over. "
            "Runs continuously at midnight, or once with --once."
        ),
    )
    parser.add_argument(
        "--database",
        default=DEFAULT_DB,
        help=f"Knowledge database path (default {DEFAULT_DB})",
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TZ,
        help=f"Timezone name (default {DEFAULT_TZ})",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="run one pass and exit (cron friendly)",
    )

    args = parser.parse_args(argv)

    if args.once:
        summary = run_once(args.database, args.timezone)
        print(json.dumps(summary, ensure_ascii=False, default=str))
        return 0

    return loop(args.database, args.timezone)


if __name__ == "__main__":
    sys.exit(main())
