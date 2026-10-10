// Typed client for the FastAPI backend.
// Regenerate exact types from the OpenAPI spec with: npm run gen:api

export type Decision = "approve" | "edit" | "reject";

export interface FitScore {
  job_id: string;
  score: number;
  matched: string[];
  gaps: string[];
  evidence_ids: string[];
  rationale: string;
}

export interface DraftForReview {
  job_id: string;
  draft: { summary: string; bullets: string[]; change_log: string[] };
  violations: { bullet: string; reason: string }[];
}

export interface RunResponse {
  thread_id: string;
  status: "awaiting_review" | "completed" | "failed";
  path: string[];
  review: { type: "review_drafts"; drafts: DraftForReview[] } | null;
  scored: FitScore[];
  artifacts: Record<string, string>;
  errors: string[];
  error: string | null;
}

export interface SearchPreferences {
  roles: string[];
  locations: string[];
  work_modes: ("onsite" | "hybrid" | "remote-india")[];
  min_ctc_lpa?: number;
}

export interface ReviewDecision {
  job_id: string;
  decision: Decision;
  edited_summary?: string;
  edited_bullets?: string[];
  comment?: string;
}

const BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function call<T>(path: string, token: string | undefined, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...init?.headers,
    },
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`${init?.method ?? "GET"} ${path} failed: ${res.status} ${await res.text()}`);
  }
  return (await res.json()) as T;
}

export const api = {
  startRun: (token: string | undefined, resumeText: string, preferences: SearchPreferences) =>
    call<RunResponse>("/runs", token, {
      method: "POST",
      body: JSON.stringify({ resume_text: resumeText, preferences }),
    }),
  getRun: (token: string | undefined, threadId: string) =>
    call<RunResponse>(`/runs/${threadId}`, token),
  review: (token: string | undefined, threadId: string, decisions: ReviewDecision[]) =>
    call<RunResponse>(`/runs/${threadId}/review`, token, {
      method: "POST",
      body: JSON.stringify({ decisions }),
    }),
};
