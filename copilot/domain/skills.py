"""Skill taxonomy: canonicalise synonyms so "k8s" and "Kubernetes" match."""

from __future__ import annotations

import re
from collections.abc import Iterable

SKILL_SYNONYMS: dict[str, str] = {
    "k8s": "kubernetes",
    "kube": "kubernetes",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "golang": "go",
    "postgres": "postgresql",
    "psql": "postgresql",
    "mongo": "mongodb",
    "gcp": "google cloud",
    "aws": "aws",
    "amazon web services": "aws",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "llm": "large language models",
    "llms": "large language models",
    "genai": "generative ai",
    "gen ai": "generative ai",
    "rag": "retrieval augmented generation",
    "ci/cd": "ci/cd",
    "cicd": "ci/cd",
    "react.js": "react",
    "reactjs": "react",
    "node.js": "node",
    "nodejs": "node",
    "next.js": "nextjs",
    "fast api": "fastapi",
    "sklearn": "scikit-learn",
    "tf": "tensorflow",
}

# Vocabulary used to spot skills inside free text (JD requirements).
KNOWN_SKILLS: frozenset[str] = frozenset(
    {
        "python", "java", "go", "rust", "c++", "c#", "javascript", "typescript", "scala",
        "kotlin", "swift", "sql", "postgresql", "mysql", "mongodb", "redis", "kafka",
        "spark", "airflow", "dbt", "snowflake", "databricks", "aws", "azure", "google cloud",
        "docker", "kubernetes", "terraform", "ci/cd", "linux", "react", "nextjs", "node",
        "fastapi", "django", "flask", "spring", "graphql", "rest", "microservices",
        "machine learning", "deep learning", "natural language processing", "pytorch",
        "tensorflow", "scikit-learn", "large language models", "generative ai",
        "retrieval augmented generation", "langchain", "langgraph", "vector databases",
        "qdrant", "elasticsearch", "system design", "data structures", "algorithms",
        "pandas", "numpy", "tableau", "power bi", "excel", "git", "agile",
    }
)


# Short terms that are also ordinary words or ambiguous in prose ("go to market").
# They still canonicalise when given as explicit skills, but are not extracted from text.
_AMBIGUOUS_IN_TEXT: frozenset[str] = frozenset({"go", "ts", "js", "py", "ml", "dl", "tf", "kube"})


def canonical_skill(skill: str) -> str:
    key = skill.strip().lower()
    return SKILL_SYNONYMS.get(key, key)


def canonical_skills(skills: Iterable[str]) -> set[str]:
    return {canonical_skill(s) for s in skills if s.strip()}


def extract_skills(text: str) -> list[str]:
    """Known skills mentioned in free text, canonicalised, in first-seen order."""
    lowered = f" {text.lower()} "
    found: list[str] = []
    candidates = [
        t for t in (*KNOWN_SKILLS, *SKILL_SYNONYMS) if t not in _AMBIGUOUS_IN_TEXT
    ]
    for term in sorted(candidates, key=len, reverse=True):
        pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
        if re.search(pattern, lowered):
            canon = canonical_skill(term)
            if canon not in found:
                found.append(canon)
    return found
