"""Interview pack: retrieve questions by role and skills, draft STAR answers."""

from __future__ import annotations

from pydantic import BaseModel

from copilot.domain.models import (
    CandidateProfile,
    InterviewPack,
    InterviewQuestion,
    JobPosting,
    RoundType,
)
from copilot.ports.llm import LLMClient
from copilot.ports.prompts import PromptStore
from copilot.services.retrieval import Retriever
from copilot.services.structured import complete_structured

# Every Indian tech loop asks these; used when the bank has nothing better.
BASELINE_QUESTIONS: list[InterviewQuestion] = [
    InterviewQuestion(question="Walk me through your current role and team.", round_type=RoundType.HR),
    InterviewQuestion(
        question="What is your current CTC, expected CTC and notice period?",
        round_type=RoundType.HR,
    ),
    InterviewQuestion(
        question="Describe a production issue you owned end to end.",
        round_type=RoundType.MANAGERIAL,
    ),
]


class _Answers(BaseModel):
    answers: dict[str, str]


class InterviewPrepService:
    PROMPT = "interview_prep"

    def __init__(self, retriever: Retriever, llm: LLMClient, prompts: PromptStore) -> None:
        self._retriever = retriever
        self._llm = llm
        self._prompts = prompts

    def build_pack(
        self, profile: CandidateProfile, posting: JobPosting, max_questions: int = 12
    ) -> InterviewPack:
        query = f"{posting.title} {' '.join(posting.required_skills[:8])}"
        questions = [
            InterviewQuestion(
                question=c.text,
                round_type=RoundType(str(c.metadata.get("round_type", "technical"))),
                skill=str(c.metadata["skill"]) if c.metadata.get("skill") else None,
                difficulty=int(c.metadata.get("difficulty") or 2),
            )
            for c in self._retriever.interview_questions(query, limit=max_questions)
        ]
        seen = {q.question for q in questions}
        questions += [q for q in BASELINE_QUESTIONS if q.question not in seen]
        questions = questions[: max_questions + len(BASELINE_QUESTIONS)]

        prompt = self._prompts.get(self.PROMPT)
        safe = profile.redacted()
        system, user = prompt.render(
            job_title=posting.title,
            company=posting.company,
            questions="\n".join(f"- {q.question}" for q in questions),
            experience="\n".join(
                f"{r.title} at {r.company}: " + "; ".join(r.bullets) for r in safe.experience
            ),
        )
        drafted = complete_structured(
            self._llm,
            "medium",
            system,
            user,
            _Answers,
            tags={"node": "interview_prep", "job_id": posting.job_id},
        )
        answers = {q.question: drafted.answers.get(q.question, "") for q in questions}
        return InterviewPack(job_id=posting.job_id, questions=questions, answers=answers)
