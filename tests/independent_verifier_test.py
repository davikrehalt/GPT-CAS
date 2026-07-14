"""Black-box tests for the isolated raw J,g verifier."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "tools" / "laughable_jg_verify.py"
FIXTURE = ROOT / "tests" / "certificates" / "gf5_x5_faithful.json"


def qq_coefficient(numerator, denominator=1):
    return {
        "numerator": str(numerator),
        "denominator": str(denominator),
    }


def term(coefficient, *exponents):
    return {"coefficient": coefficient, "exponents": list(exponents)}


def qq_document(variables, generators, g, order="grevlex"):
    return {
        "schema": "laughable-jg-v1",
        "field": {"kind": "QQ"},
        "variables": list(variables),
        "order": order,
        "ideal_generators": generators,
        "g": g,
    }


class IndependentVerifierTests(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        with FIXTURE.open(encoding="utf-8") as source:
            cls.positive = json.load(source)

    def invoke_path(self, path, *arguments):
        completed = subprocess.run(
            [sys.executable, str(VERIFIER), *arguments, str(path)],
            cwd=str(ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.stderr, "")
        self.assertTrue(completed.stdout.endswith("\n"))
        self.assertEqual(len(completed.stdout.splitlines()), 1)
        return completed.returncode, json.loads(completed.stdout)

    def invoke_document(self, document, *arguments):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8", delete=False
        ) as destination:
            json.dump(document, destination, separators=(",", ":"))
            path = Path(destination.name)
        try:
            return self.invoke_path(path, *arguments)
        finally:
            path.unlink()

    def test_hand_authored_faithful_fixture(self):
        code, report = self.invoke_path(FIXTURE)
        self.assertEqual(code, 0)
        self.assertEqual(
            report,
            {
                "cycle_valid": True,
                "derivatives_in_J": True,
                "faithful": True,
                "finite_quotient": True,
                "g_in_J": True,
                "length_P_mod_J2": 10,
                "length_Q": 5,
                "multiplication_rank": 5,
                "reason": "verified",
                "status": "verified",
                "supported_at_origin": True,
                "verified": True,
            },
        )

    def test_verifier_source_is_isolated(self):
        source = VERIFIER.read_text(encoding="utf-8")
        self.assertNotIn("import laughableengine", source)
        self.assertNotIn("from laughableengine", source)
        self.assertNotIn("import sage", source.lower())
        self.assertNotIn("subprocess", source)

    def test_tampered_witness_is_recomputed_and_rejected(self):
        tampered = copy.deepcopy(self.positive)
        tampered["g"][0]["exponents"] = [4]
        code, report = self.invoke_document(tampered)
        self.assertEqual(code, 2)
        self.assertFalse(report["g_in_J"])
        self.assertFalse(report["cycle_valid"])
        self.assertFalse(report["verified"])
        self.assertEqual(report["length_Q"], 5)
        self.assertEqual(report["length_P_mod_J2"], 10)

    def test_wrong_characteristic_exposes_derivative_failure(self):
        wrong_characteristic = copy.deepcopy(self.positive)
        wrong_characteristic["field"]["modulus"] = "7"
        code, report = self.invoke_document(wrong_characteristic)
        self.assertEqual(code, 2)
        self.assertTrue(report["g_in_J"])
        self.assertFalse(report["derivatives_in_J"])
        self.assertFalse(report["cycle_valid"])
        self.assertEqual(report["multiplication_rank"], 5)
        self.assertFalse(report["faithful"])

    def test_colon_equality_does_not_hide_derivative_failure_over_qq(self):
        document = qq_document(
            ["x"],
            [[term(qq_coefficient(1), 2)]],
            [term(qq_coefficient(1), 2)],
            order="lex",
        )
        code, report = self.invoke_document(document)
        self.assertEqual(code, 2)
        self.assertTrue(report["g_in_J"])
        self.assertFalse(report["derivatives_in_J"])
        self.assertEqual(report["multiplication_rank"], 2)
        self.assertFalse(report["faithful"])

    def test_cycle_with_singular_multiplication_is_not_faithful(self):
        singular = copy.deepcopy(self.positive)
        singular["g"][0]["exponents"] = [6]
        code, report = self.invoke_document(singular)
        self.assertEqual(code, 2)
        self.assertTrue(report["cycle_valid"])
        self.assertEqual(report["multiplication_rank"], 4)
        self.assertEqual(report["length_Q"], 5)
        self.assertFalse(report["faithful"])
        self.assertIn("not injective", report["reason"])

    def test_origin_support_is_checked_from_the_raw_ideal(self):
        # f=x^2-x and g=f^2.  The cycle identities hold, but P/(f) is not
        # supported at the origin and multiplication by g modulo f^2 is zero.
        document = qq_document(
            ["x"],
            [[
                term(qq_coefficient(1), 2),
                term(qq_coefficient(-1), 1),
            ]],
            [
                term(qq_coefficient(1), 4),
                term(qq_coefficient(-2), 3),
                term(qq_coefficient(1), 2),
            ],
        )
        code, report = self.invoke_document(document)
        self.assertEqual(code, 2)
        self.assertTrue(report["finite_quotient"])
        self.assertTrue(report["cycle_valid"])
        self.assertFalse(report["supported_at_origin"])
        self.assertEqual(report["multiplication_rank"], 0)
        self.assertFalse(report["faithful"])

    def test_positive_dimensional_ideal_is_conclusively_rejected(self):
        document = qq_document(
            ["x", "y"],
            [[term(qq_coefficient(1), 1, 0)]],
            [term(qq_coefficient(1), 2, 0)],
        )
        code, report = self.invoke_document(document)
        self.assertEqual(code, 2)
        self.assertFalse(report["finite_quotient"])
        self.assertTrue(report["cycle_valid"])
        self.assertIsNone(report["length_Q"])
        self.assertIsNone(report["multiplication_rank"])
        self.assertFalse(report["faithful"])

    def test_derived_discovery_claim_is_forbidden_by_schema(self):
        malformed = copy.deepcopy(self.positive)
        malformed["faithful"] = True
        code, report = self.invoke_document(malformed)
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "malformed")
        self.assertFalse(report["verified"])
        self.assertIn("extra=faithful", report["reason"])

    def test_duplicate_json_key_is_malformed(self):
        text = FIXTURE.read_text(encoding="utf-8")
        text = text.replace(
            '"schema": "laughable-jg-v1",',
            '"schema": "laughable-jg-v1",\n  "schema": "laughable-jg-v1",',
            1,
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8", delete=False
        ) as destination:
            destination.write(text)
            path = Path(destination.name)
        try:
            code, report = self.invoke_path(path)
        finally:
            path.unlink()
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "malformed")
        self.assertIn("duplicate JSON object key", report["reason"])

    def test_noncanonical_coefficient_is_malformed(self):
        malformed = copy.deepcopy(self.positive)
        malformed["g"][0]["coefficient"] = "5"
        code, report = self.invoke_document(malformed)
        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "malformed")
        self.assertIn("canonical range", report["reason"])

    def test_resource_exhaustion_is_inconclusive(self):
        code, report = self.invoke_path(
            FIXTURE, "--max-standard-monomials", "4"
        )
        self.assertEqual(code, 3)
        self.assertEqual(report["status"], "resource_limit")
        self.assertFalse(report["verified"])
        self.assertFalse(report["faithful"])
        self.assertIn("standard monomial limit", report["reason"])


if __name__ == "__main__":
    unittest.main()
