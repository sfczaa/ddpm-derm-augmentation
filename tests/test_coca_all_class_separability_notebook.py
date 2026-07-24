"""Static safety checks for the all-class separability Colab diagnostic.

The implementation-stage notebook must keep the REPLACE_AFTER_PUSH pinning
placeholder (so Run all fails loud until an independent reviewer pins the pushed
commit), stay unexecuted and compilable, and never reach test data, synthetic
images, formal training, or checkpoints.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "colab_coca_v4_all_class_separability_diagnostic.ipynb"


class AllClassSeparabilityNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell.get("source", []))
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        )
        cls.all_source = "\n".join(
            "".join(cell.get("source", [])) for cell in cls.notebook["cells"]
        )

    def test_placeholder_pinning_and_unexecuted_compilable(self):
        first = "".join(self.notebook["cells"][0]["source"])
        self.assertIn('EXPECTED_GIT_COMMIT = "REPLACE_AFTER_PUSH"', first)
        self.assertIn('EXPECTED_GIT_COMMIT != "REPLACE_AFTER_PUSH"', first)
        self.assertIn("len(EXPECTED_GIT_COMMIT) == 40", first)
        # the reviewer pins the real commit later; no 40-hex SHA may be present yet
        self.assertNotRegex(json.dumps(self.notebook), r'EXPECTED_GIT_COMMIT = "[0-9a-f]{40}"')
        for index, cell in enumerate(self.notebook["cells"]):
            self.assertFalse(cell.get("outputs"))
            if cell["cell_type"] == "code":
                self.assertIsNone(cell.get("execution_count"))
                compile("".join(cell["source"]), f"diagnostic:cell-{index}", "exec")

    def test_scope_is_descriptive_all_class_representation_only(self):
        for required in (
            'DIAGNOSTIC_VERSION = "v1_all_class_separability"',
            "EXPECTED_TRAIN_ROWS = 6995",
            "EXPECTED_VALIDATION_ROWS = 1510",
            '"df": 85',
            '"df": 14',
            "ddpm_derm.coca_all_class_separability_diagnostic",
            'record["interpretation_scope"] == "descriptive_representation_diagnostic_not_model_or_candidate_selection"',
            "ALL-CLASS COCA SEPARABILITY DIAGNOSTIC COMPLETED",
            "condition_selected=false",
        ):
            self.assertIn(required, self.all_source)

    def test_never_touches_test_synthetic_formal_or_checkpoints(self):
        # No test-split access, no formal training, no checkpoints, and no
        # synthetic-ingestion path. The historical group name "synthetic_df" is
        # allowed only where the notebook verifies the *prior* embedding record.
        for banned in (
            'load_split("test")',
            "test.csv",
            "train_classifier",
            "CANDIDATE_MANIFEST",
            "EXPECTED_CANDIDATE_SHA256",
            "load_generated_manifest",
            "build_classifier_mixture_frame",
            "--generated-manifest",
            "--mixture-synthetic-count",
            "--variant",
            "--evaluation-scope",
            "FORMAL_ROOT",
            "best.pt",
            "last.pt",
            "torch.save",
        ):
            self.assertNotIn(banned, self.code)
        # the only synthetic references are the guarded prior-record identifiers
        self.assertIn('"synthetic_df": 500', self.code)  # prior embedding group_counts
        self.assertNotIn("outputs/synthetic_df", self.code)
        # the review asserts no checkpoint files were produced
        self.assertIn('rglob("*.pt")', self.code)
        self.assertIn("must not write model checkpoints", self.code)

    def test_requires_and_guards_prior_evidence_before_any_write(self):
        for required in (
            'V4_FAILURE_RECORD = CLASSIFIER_ROOT / "v4_focal_inverse_frequency" / "latest_validation_failure.json"',
            "MIXTURE_RECORD = CLASSIFIER_ROOT",
            'v4_failure["validation_status"] == "VALIDATION FAILED"',
            'embedding["diagnostic_status"] == "COMPLETED"',
            'mixture["diagnostic_status"] == "COMPLETED"',
            'mixture["formal_training_started"] is False',
            'mixture["test_data_accessed"] is False',
            'mixture.get("condition_selected", False) is False',
            'mixture["interpretation_scope"] == "descriptive_not_candidate_selection"',
            "guard_before = {",
            "guard_after == guard_before",
        ):
            self.assertIn(required, self.code)
        # prior evidence is snapshotted before the diagnostic subprocess runs
        self.assertLess(
            self.code.index("guard_before = {"),
            self.code.index("ddpm_derm.coca_all_class_separability_diagnostic"),
        )
        # and the guard is re-checked after the run
        self.assertLess(
            self.code.index("ddpm_derm.coca_all_class_separability_diagnostic"),
            self.code.index("guard_after == guard_before"),
        )

    def test_completed_latest_record_is_guarded_before_writing(self):
        self.assertIn(
            "assert not LATEST_RECORD.exists()",
            self.code,
        )
        self.assertLess(
            self.code.index("assert not LATEST_RECORD.exists()"),
            self.code.index("ddpm_derm.coca_all_class_separability_diagnostic"),
        )

    def test_train_and_validation_only_copy_with_visible_progress(self):
        for required in (
            "def copy_group(",
            'f"START {group} copy: total={total}"',
            "flush=True",
            "copy failed at ",
            ") from exc",
            'EXPECTED_TRAIN_CLASS_COUNTS)',
            'set(split_frames["train"][field]) & set(split_frames["val"][field])',
            '"synthetic" not in set(frame.get("source"',
        ):
            self.assertIn(required, self.code)

    def test_runtime_local_data_is_isolated_from_drive_output(self):
        for required in (
            'LOCAL_DATA_DIR = Path("/content/ham10000-train-val-only")',
            'str(LOCAL_DATA_DIR).startswith("/content/")',
            "SHARED_RUN_ROOT not in LOCAL_DATA_DIR.parents",
            "coca_run.ensure_tree(SHARED_RUN_ROOT",
        ):
            self.assertIn(required, self.code)

    def test_secure_clone_leaves_no_credentials(self):
        for required in (
            'userdata.get("GH_TOKEN")',
            'subprocess.run(["git", "clone", REPO_URL, str(CODE_DIR)], check=True, env=clone_env)',
            'clone_env["GIT_CONFIG_VALUE_0"] = ""',
            "del token, basic_credential, clone_env",
            'assert "@" not in remote and "x-access-token" not in remote',
            'version("scikit-learn") == "1.8.0"',
        ):
            self.assertIn(required, self.code)
        self.assertNotIn("@github.com", self.code)
        self.assertNotRegex(json.dumps(self.notebook), r"gh[pousr]_[A-Za-z0-9]")
        self.assertNotRegex(json.dumps(self.notebook), r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+")

    def test_three_formal_records_and_completion_marker(self):
        for required in (
            'record_path = OUTPUT_DIR / "all_class_separability_diagnostic.json"',
            'completed_path = OUTPUT_DIR / "_COMPLETED.json"',
            'identity_path = OUTPUT_DIR / "diagnostic_identity.json"',
            'integrity_path = OUTPUT_DIR / "record_integrity.json"',
            'embedding_path = OUTPUT_DIR / "all_class_embeddings.npz"',
            "record_bytes == completed_path.read_bytes() == LATEST_RECORD.read_bytes()",
            'record["fixed_knn_k_values"] == [1, 5, 10]',
            'set(record["analyses"]["cosine_knn"]["by_k"]) == {"k1", "k5", "k10"}',
        ):
            self.assertIn(required, self.code)

    def test_post_run_parses_and_validates_immutable_identity(self):
        # the identity file is really parsed, and the FULL expected identity --
        # including a freshly recomputed model_identity -- is rebuilt and
        # validated field-by-field (not just fixed scalar fields).
        for required in (
            'identity = json.loads(identity_path.read_text(encoding="utf-8"))',
            'record["diagnostic_identity"] == identity',
            'build_model(arch="coca_vit_b32", freeze_backbone=True, coca_pretrained="laion2b_s13b_b90k")',
            'expected_model_identity = compute_model_identity(review_model, "coca_vit_b32", 224)',
            "expected_identity = diag.build_identity(",
            "git_commit=commit",
            'sha256(LOCAL_DATA_DIR / "manifests" / "train.csv")',
            'sha256(LOCAL_DATA_DIR / "manifests" / "val.csv")',
            "shared_root_uuid=shared_root_uuid",
            "model_identity=expected_model_identity",
            'dependency_versions={"open_clip_torch": version("open_clip_torch")',
            "algorithm_identities=dict(diag.ALGORITHM_IDENTITIES)",
            "diag.validate_diagnostic_identity(record, identity, expected_identity)",
            "model_identity == expected_model_identity",
            "record[\"model_identity\"] == expected_model_identity",
            'model_identity["arch"] == "coca_vit_b32"',
            'model_identity["model_name"] == "coca_ViT-B-32"',
            'model_identity["pretrained_tag"] == "laion2b_s13b_b90k"',
            'model_identity["freeze_mode"] == "frozen_image_encoder_linear_head"',
            'model_identity["input_resolution"] == [224, 224]',
            'model_identity["preprocessing_identity"]["eval"] == expected_model_identity["preprocessing_identity"]["eval"]',
        ):
            self.assertIn(required, self.code)
        # the expected identity supplies model_identity, so a consistently-wrong
        # model_identity cannot pass; the validator runs before the decision cell.
        self.assertLess(
            self.code.index("diag.validate_diagnostic_identity"),
            self.code.index("ALL-CLASS COCA SEPARABILITY DIAGNOSTIC COMPLETED"),
        )

    def test_post_run_recomputes_record_integrity_sidecar(self):
        for required in (
            "diag.validate_record_integrity(output_dir=OUTPUT_DIR, latest_record_path=LATEST_RECORD)",
            'integrity["schema"] == "all_class_separability_record_integrity_v1"',
            "integrity[name][\"sha256\"] == sha256(path) == hashlib.sha256(data).hexdigest()",
            'integrity[name]["bytes"] == path.stat().st_size == len(data)',
            'integrity["raw_bytes_equal"] is True',
        ):
            self.assertIn(required, self.code)

    def test_post_run_validates_eight_key_npz_contract(self):
        for required in (
            "set(saved.files) == set(expected_npz_shapes)",
            'saved[name].dtype != object',
            "saved[name].dtype == np.float32",
            'saved[prefix + "_labels"].dtype == np.int64',
            'saved[key].dtype.kind == "U"',
            "np.array_equal(saved[prefix + \"_labels\"], frame[\"label_idx\"].to_numpy(dtype=np.int64))",
            "np.asarray(frame[field].astype(str).tolist(), dtype=np.str_)",
            'sha256(embedding_path) == record["embedding_artifact"]["sha256"]',
            'embedding_path.stat().st_size == record["embedding_artifact"]["size_bytes"]',
            'npz_keys == record["embedding_artifact"]["npz_keys"]',
            "diag._validate_npz_file(embedding_path, n_train=EXPECTED_TRAIN_ROWS, n_val=EXPECTED_VALIDATION_ROWS)",
        ):
            self.assertIn(required, self.code)


if __name__ == "__main__":
    unittest.main()
