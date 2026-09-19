from __future__ import annotations

from datetime import datetime

from .operations import KnowledgeOperation


class KnowledgeValidator:

    def validate(self, operation: KnowledgeOperation) -> None:
        operation.validate()

        if operation.valid_from:
            self._validate_datetime(
                operation.valid_from,
                "valid_from",
            )

        if operation.valid_until:
            self._validate_datetime(
                operation.valid_until,
                "valid_until",
            )

        if (
            operation.valid_from
            and operation.valid_until
        ):
            start = datetime.fromisoformat(
                operation.valid_from
            )
            end = datetime.fromisoformat(
                operation.valid_until
            )

            if end <= start:
                raise ValueError(
                    "valid_until must be later than valid_from"
                )

    @staticmethod
    def _validate_datetime(
        value: str,
        field_name: str,
    ) -> None:
        try:
            datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} must be a valid ISO-8601 datetime"
            ) from exc
