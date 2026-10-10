"use client";

// Starter flow: paste resume -> start run -> review drafts -> approve.
// Matches, tracker and interview-prep screens come next (see README roadmap).
import { useState } from "react";
import { api, type ReviewDecision, type RunResponse } from "@/lib/api";

export default function Home() {
  const [resume, setResume] = useState("");
  const [roles, setRoles] = useState("backend engineer");
  const [cities, setCities] = useState("Bengaluru, Pune");
  const [run, setRun] = useState<RunResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function start() {
    setBusy(true);
    setError(null);
    try {
      setRun(
        await api.startRun(undefined, resume, {
          roles: roles.split(",").map((r) => r.trim()).filter(Boolean),
          locations: cities.split(",").map((c) => c.trim()).filter(Boolean),
          work_modes: [],
        }),
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function decide(decision: ReviewDecision["decision"]) {
    if (!run?.review) return;
    const decisions = run.review.drafts.map((d) => ({ job_id: d.job_id, decision }));
    setRun(await api.review(undefined, run.thread_id, decisions));
  }

  return (
    <>
      <h1>Job Application Copilot</h1>
      <label>
        Resume text
        <textarea value={resume} onChange={(e) => setResume(e.target.value)} rows={10} style={{ width: "100%" }} />
      </label>
      <p>
        <input value={roles} onChange={(e) => setRoles(e.target.value)} placeholder="Roles" />{" "}
        <input value={cities} onChange={(e) => setCities(e.target.value)} placeholder="Cities" />{" "}
        <button onClick={start} disabled={busy || resume.length < 50}>
          {busy ? "Running…" : "Find and tailor"}
        </button>
      </p>
      {error && <p role="alert">{error}</p>}

      {run && (
        <section>
          <h2>Matches</h2>
          <ul>
            {run.scored.map((s) => (
              <li key={s.job_id}>
                <strong>{s.score}</strong> {s.job_id} — {s.rationale}
              </li>
            ))}
          </ul>

          {run.status === "awaiting_review" && run.review && (
            <>
              <h2>Review drafts</h2>
              {run.review.drafts.map((d) => (
                <article key={d.job_id}>
                  <h3>{d.job_id}</h3>
                  <p>{d.draft.summary}</p>
                  <ul>{d.draft.bullets.map((b) => <li key={b}>{b}</li>)}</ul>
                  {d.violations.length > 0 && (
                    <p role="alert">Check: {d.violations.map((v) => v.reason).join("; ")}</p>
                  )}
                </article>
              ))}
              <button onClick={() => decide("approve")}>Approve all</button>{" "}
              <button onClick={() => decide("reject")}>Reject all</button>
            </>
          )}

          {run.status === "failed" && <p role="alert">Run failed: {run.error ?? "unknown error"}</p>}
          {run.errors.length > 0 && <p role="alert">Warnings: {run.errors.join("; ")}</p>}

          {run.status === "completed" && (
            <p>Done. Resumes: {Object.values(run.artifacts).join(", ") || "none"}</p>
          )}
          <p style={{ fontSize: 12, color: "#666" }}>Jobs by Adzuna where shown.</p>
        </section>
      )}
    </>
  );
}
