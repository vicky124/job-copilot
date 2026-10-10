"""Write JDs, candidate achievements and interview questions into the vector store."""

from __future__ import annotations

from copilot.config.settings import CollectionsConfig
from copilot.domain.models import CandidateProfile, Chunk, InterviewQuestion, JobPosting
from copilot.ports.embeddings import Embedder
from copilot.ports.vector_store import VectorStore
from copilot.services.chunking import chunk_job


class CollectionNames:
    """Physical names carry the embedding suffix, e.g. job_descriptions_v1_nomic768."""

    def __init__(self, logical: CollectionsConfig, suffix: str) -> None:
        self.job_descriptions = f"{logical.job_descriptions}_{suffix}"
        self.interview_questions = f"{logical.interview_questions}_{suffix}"
        self.candidate_profile = f"{logical.candidate_profile}_{suffix}"

    def all(self) -> list[str]:
        return [self.job_descriptions, self.interview_questions, self.candidate_profile]


class IndexingService:
    def __init__(self, store: VectorStore, embedder: Embedder, names: CollectionNames) -> None:
        self._store = store
        self._embedder = embedder
        self.names = names
        for name in names.all():
            store.ensure_collection(name, embedder.dimensions)

    def _write(self, collection: str, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self._embedder.embed_documents([c.text for c in chunks])
        self._store.upsert(collection, chunks, vectors)
        return len(chunks)

    def index_job(self, posting: JobPosting) -> int:
        """Re-indexing a job replaces its previous chunks (group = job_id)."""
        self._store.delete_group(self.names.job_descriptions, "job_id", posting.job_id)
        return self._write(self.names.job_descriptions, chunk_job(posting))

    def index_profile(self, user_id: str, profile: CandidateProfile) -> int:
        self._store.delete_group(self.names.candidate_profile, "user_id", user_id)
        chunks: list[Chunk] = []
        for r_idx, role in enumerate(profile.experience):
            for b_idx, bullet in enumerate(role.bullets):
                chunks.append(
                    Chunk(
                        id=f"{user_id}:exp{r_idx}:{b_idx}",
                        text=f"{bullet} (as {role.title} at {role.company})",
                        metadata={"user_id": user_id, "role": role.title, "company": role.company},
                    )
                )
        if profile.skills:
            chunks.append(
                Chunk(
                    id=f"{user_id}:skills",
                    text="Skills: " + ", ".join(profile.skills),
                    metadata={"user_id": user_id, "role": "skills", "company": ""},
                )
            )
        return self._write(self.names.candidate_profile, chunks)

    def index_questions(self, questions: list[InterviewQuestion], bank: str = "default") -> int:
        chunks = [
            Chunk(
                id=f"{bank}:{i}",
                text=q.question,
                metadata={
                    "bank": bank,
                    "round_type": q.round_type.value,
                    "skill": q.skill,
                    "difficulty": q.difficulty,
                },
            )
            for i, q in enumerate(questions)
        ]
        return self._write(self.names.interview_questions, chunks)
