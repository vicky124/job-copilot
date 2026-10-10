"""A/B prompt router: sticky assignment of a user to a prompt variant."""

from __future__ import annotations

import hashlib

from copilot.config.settings import ExperimentConfig


class ABRouter:
    def __init__(self, experiments: dict[str, ExperimentConfig]) -> None:
        self._experiments = experiments

    @staticmethod
    def bucket(user_id: str, experiment: str) -> float:
        """Stable value in [0, 1) for a user within one experiment."""
        digest = hashlib.sha256(f"{experiment}:{user_id}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64

    def variant(self, experiment: str, user_id: str) -> str:
        """Prompt label for this user. Unknown experiments get 'production'."""
        config = self._experiments.get(experiment)
        if config is None:
            return "production"
        point = self.bucket(user_id, experiment)
        cumulative = 0.0
        for label, weight in sorted(config.variants.items()):
            cumulative += weight
            if point < cumulative:
                return label
        return sorted(config.variants)[-1]
