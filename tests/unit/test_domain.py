import unittest

from pydantic import ValidationError

from copilot.domain.india import (
    infer_work_mode,
    is_india_location,
    normalise_cities,
    parse_ctc_lpa,
    parse_experience_band,
    parse_notice_period_days,
)
from copilot.domain.matching import matches_query
from copilot.domain.models import (
    Decision,
    InterviewPack,
    ReviewDecision,
    WorkMode,
)
from copilot.domain.skills import canonical_skill, extract_skills
from tests.fakes import make_job, make_profile, make_query


class CitiesTest(unittest.TestCase):
    def test_aliases_normalised(self) -> None:
        self.assertEqual(normalise_cities("Bangalore / Gurgaon (Hybrid)"), ["Bengaluru", "Gurugram"])

    def test_longest_alias_wins(self) -> None:
        self.assertEqual(normalise_cities("Navi Mumbai"), ["Mumbai"])

    def test_no_city(self) -> None:
        self.assertEqual(normalise_cities("London, UK"), [])

    def test_india_location(self) -> None:
        self.assertTrue(is_india_location("India - Remote"))
        self.assertTrue(is_india_location("Bangalore,India"))
        self.assertFalse(is_india_location("Palo Alto, CA"))
        self.assertTrue(is_india_location("BLR office", ["blr"]))


class CtcTest(unittest.TestCase):
    def test_ranges(self) -> None:
        cases = {
            "CTC 12-18 LPA": (12.0, 18.0),
            "₹12L – ₹18L": (12.0, 18.0),
            "salary 20 to 30 lakhs": (20.0, 30.0),
            "Rs. 8.5 - 10 lacs": (8.5, 10.0),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(parse_ctc_lpa(text), expected)

    def test_single_and_crore(self) -> None:
        self.assertEqual(parse_ctc_lpa("up to 15 LPA"), (15.0, 15.0))
        self.assertEqual(parse_ctc_lpa("1.2 Cr package"), (120.0, 120.0))

    def test_absent(self) -> None:
        self.assertEqual(parse_ctc_lpa("competitive pay"), (None, None))


class NoticeAndExperienceTest(unittest.TestCase):
    def test_notice(self) -> None:
        self.assertEqual(parse_notice_period_days("Immediate joiners preferred"), 0)
        self.assertEqual(parse_notice_period_days("Notice period max 30 days"), 30)
        self.assertEqual(parse_notice_period_days("60 days notice is fine"), 60)
        self.assertEqual(parse_notice_period_days("notice period up to 2 months"), 60)
        self.assertIsNone(parse_notice_period_days("great culture"))

    def test_experience(self) -> None:
        self.assertEqual(parse_experience_band("3-5 yrs"), (3.0, 5.0))
        self.assertEqual(parse_experience_band("3 to 5 years"), (3.0, 5.0))
        self.assertEqual(parse_experience_band("5+ years"), (5.0, None))
        self.assertEqual(parse_experience_band("fresh grads"), (None, None))

    def test_work_mode(self) -> None:
        self.assertIs(infer_work_mode("Bengaluru (Hybrid)"), WorkMode.HYBRID)
        self.assertIs(infer_work_mode("Remote - India"), WorkMode.REMOTE_INDIA)
        self.assertIs(infer_work_mode("Remote"), WorkMode.REMOTE_GLOBAL)
        self.assertIs(infer_work_mode("Pune"), WorkMode.UNKNOWN)


class SkillsTest(unittest.TestCase):
    def test_canonical(self) -> None:
        self.assertEqual(canonical_skill("K8s"), "kubernetes")
        self.assertEqual(canonical_skill(" Postgres "), "postgresql")

    def test_extract(self) -> None:
        found = extract_skills("Strong Python, k8s and Postgres; React.js a plus")
        self.assertEqual(set(found), {"python", "kubernetes", "postgresql", "react"})

    def test_ambiguous_words_not_extracted(self) -> None:
        self.assertEqual(extract_skills("go to market with ml-driven ts"), [])

    def test_no_partial_word_match(self) -> None:
        self.assertNotIn("java", extract_skills("Strong JavaScript"))


class ModelsTest(unittest.TestCase):
    def test_ctc_range_validated(self) -> None:
        with self.assertRaises(ValidationError):
            make_job(ctc_min_lpa=30, ctc_max_lpa=10)

    def test_job_id_and_hash(self) -> None:
        a = make_job(source="greenhouse", external_id="x1")
        b = make_job(source="adzuna", external_id="y2")
        self.assertEqual(a.job_id, "greenhouse:x1")
        self.assertEqual(a.content_hash, b.content_hash)  # same JD on two boards

    def test_profile_helpers(self) -> None:
        p = make_profile()
        self.assertIn("kubernetes", p.all_skills())
        self.assertEqual(p.employers(), {"payquick", "billdesk labs"})
        redacted = p.redacted()
        self.assertIsNone(redacted.email)
        self.assertIsNone(redacted.phone)
        self.assertEqual(redacted.full_name, p.full_name)

    def test_profile_is_immutable(self) -> None:
        with self.assertRaises(ValidationError):
            make_profile().full_name = "x"  # type: ignore[misc]

    def test_widened_query(self) -> None:
        q = make_query(work_modes=[WorkMode.HYBRID], ctc_min_lpa=20, keywords=["a", "b", "c"], limit=10)
        w = q.widened()
        self.assertEqual(w.work_modes, [])
        self.assertIsNone(w.ctc_min_lpa)
        self.assertEqual(w.keywords, ["a"])
        self.assertEqual(w.limit, 20)

    def test_edit_decision_needs_content(self) -> None:
        with self.assertRaises(ValidationError):
            ReviewDecision(job_id="j", decision=Decision.EDIT)
        ok = ReviewDecision(job_id="j", decision=Decision.EDIT, edited_summary="new")
        self.assertEqual(ok.edited_summary, "new")

    def test_interview_pack_needs_questions(self) -> None:
        with self.assertRaises(ValidationError):
            InterviewPack(job_id="j", questions=[])


class MatchingTest(unittest.TestCase):
    def test_happy_match(self) -> None:
        self.assertTrue(matches_query(make_job(), make_query(exp_years=4)))

    def test_title_mismatch(self) -> None:
        self.assertFalse(matches_query(make_job(title="Data Analyst"), make_query()))

    def test_location_mismatch_unless_remote(self) -> None:
        pune = make_job(location="Pune", locations=["Pune"])
        self.assertFalse(matches_query(pune, make_query()))
        remote = make_job(location="Remote", locations=[], work_mode=WorkMode.REMOTE_INDIA)
        self.assertTrue(matches_query(remote, make_query()))

    def test_experience_band_with_slack(self) -> None:
        self.assertTrue(matches_query(make_job(exp_min=5, exp_max=8), make_query(exp_years=4)))
        self.assertFalse(matches_query(make_job(exp_min=6, exp_max=9), make_query(exp_years=4)))

    def test_ctc_only_filters_when_known(self) -> None:
        self.assertFalse(matches_query(make_job(ctc_max_lpa=20), make_query(ctc_min_lpa=25)))
        no_ctc = make_job(ctc_min_lpa=None, ctc_max_lpa=None)
        self.assertTrue(matches_query(no_ctc, make_query(ctc_min_lpa=25)))


if __name__ == "__main__":
    unittest.main()
