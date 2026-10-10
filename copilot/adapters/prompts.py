"""File-backed prompt store: prompts/<name>.yaml with labelled versions.

Each file seeds Langfuse prompt management; the same name@label is fetched from
Langfuse in staging/prod by an adapter implementing the same PromptStore port.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from copilot.ports.prompts import Prompt, PromptNotFoundError


class FilePromptStore:
    def __init__(self, directory: Path) -> None:
        self._dir = directory
        self._cache: dict[str, dict[str, Prompt]] = {}

    def _load(self, name: str) -> dict[str, Prompt]:
        if name not in self._cache:
            path = self._dir / f"{name}.yaml"
            if not path.exists():
                raise PromptNotFoundError(f"no prompt file {path}")
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            versions = raw.get("versions", {})
            labels: dict[str, Prompt] = {}
            for version, body in versions.items():
                for label in body.get("labels", []):
                    labels[label] = Prompt(
                        name=name,
                        version=str(version),
                        label=label,
                        system=body["system"],
                        user=body["user"],
                    )
            self._cache[name] = labels
        return self._cache[name]

    def get(self, name: str, label: str = "production") -> Prompt:
        labels = self._load(name)
        if label not in labels:
            if "production" in labels:
                return labels["production"]
            raise PromptNotFoundError(f"prompt {name} has no label '{label}'")
        return labels[label]
