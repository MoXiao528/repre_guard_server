from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import evidence_router as module
from evidence_router import EvidenceRouter


LANGUAGES = ("ar", "de", "en", "es", "fr", "pt", "ru", "zh")
DOMAINS = ("academic", "news", "novel", "seo", "webtext", "wiki")
CLASSES = [f"{language}:{domain}" for language in LANGUAGES for domain in DOMAINS]
FILES = (
    "config.json",
    "router_config.json",
    "router_model.safetensors",
    "sentencepiece.bpe.model",
    "tokenizer.json",
    "tokenizer_config.json",
)


def encode(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()


def artifact_sha(directory):
    files = {}
    for name in FILES:
        encoded = (directory / name).read_bytes()
        files[name] = {
            "bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    return hashlib.sha256(encode(files)).hexdigest()


def write_artifact(directory):
    config = {
        "model_type": "xlm-roberta",
        "architectures": ["XLMRobertaHierarchicalRouter"],
        "id2label": {str(i): name for i, name in enumerate(CLASSES)},
        "label2id": {name: i for i, name in enumerate(CLASSES)},
        "vocab_size": 64,
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 16,
        "max_position_embeddings": 514,
        "hidden_dropout_prob": 0.1,
    }
    router_config = {
        "schema_version": 1,
        "architecture": "one_xlmr_encoder_one_language_head_eight_domain_heads",
        "language_heads": 1,
        "domain_heads": 8,
        "domains_per_head": 6,
        "max_length": 512,
        "truncation": "canonical_nfkc_whitespace_then_head_255_tail_255",
        "classes": CLASSES,
    }
    for name in FILES:
        (directory / name).write_bytes(b"synthetic file")
    (directory / "config.json").write_bytes(encode(config))
    (directory / "router_config.json").write_bytes(encode(router_config))
    return config


class XLMRobertaTokenizerFast:
    is_fast = True
    model_max_length = 512
    bos_token_id, pad_token_id, eos_token_id, unk_token_id, mask_token_id = (
        0,
        1,
        2,
        3,
        250001,
    )

    def __init__(self, ids=None):
        self.ids = ids if ids is not None else [3, 4]
        self.calls = []
        self.prepared = None

    def __len__(self):
        return 250002

    def num_special_tokens_to_add(self, *, pair):
        assert pair is False
        return 2

    def __call__(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return {"input_ids": [self.ids.copy()]}

    def prepare_for_model(self, ids, **kwargs):
        assert kwargs == {
            "add_special_tokens": True,
            "truncation": False,
            "padding": False,
            "return_attention_mask": True,
            "return_token_type_ids": False,
        }
        self.prepared = [0, *ids, 2]
        return {"input_ids": self.prepared, "attention_mask": [1] * len(self.prepared)}

    def pad(self, rows, *, pad_to_multiple_of, return_tensors):
        import torch

        assert pad_to_multiple_of == 8 and return_tensors == "pt" and len(rows) == 1
        row = rows[0]
        padding = (-len(row["input_ids"])) % 8
        return {
            "input_ids": torch.tensor([row["input_ids"] + [1] * padding]),
            "attention_mask": torch.tensor([row["attention_mask"] + [0] * padding]),
        }


class FakeModel:
    def __init__(self):
        self.calls = 0
        self.failure = False

    def __call__(self, **kwargs):
        import torch

        assert not torch.is_grad_enabled()
        assert kwargs["input_ids"].device.type == "cpu"
        self.calls += 1
        if self.failure:
            raise RuntimeError("private input and artifact path")
        return {"logits": torch.zeros(1, 48)}


class FakeIdentifier:
    def __init__(self):
        self.calls = []
        self.ranking = [("en", 0.02), ("ja", 0.01)]
        self.empty = [("sr", 0.02), ("en", 0.01)]
        self.failure = False

    def rank(self, text):
        self.calls.append(text)
        if self.failure:
            raise RuntimeError("private input or dependency path")
        return list(self.ranking if text else self.empty)


class EvidenceRouterTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.config = write_artifact(self.directory)
        self.sha = artifact_sha(self.directory)
        self.real_load_identifier = module._load_language_identifier
        self.identifier = FakeIdentifier()
        loader = patch.object(
            module, "_load_language_identifier", return_value=self.identifier
        )
        self.identifier_loader = loader.start()
        self.addCleanup(loader.stop)

    def load(self, **overrides):
        kwargs = dict(enabled=True, model_path=self.directory, artifact_sha256=self.sha)
        kwargs.update(overrides)
        return EvidenceRouter(**kwargs)

    def test_off_and_invalid_configuration_do_not_touch_files_or_import_models(self):
        script = """
import sys
sys.path.insert(0, sys.argv[1])
from evidence_router import EvidenceRouter
for enabled in (False, 'false', '0', '', 'off'):
    router = EvidenceRouter(enabled=enabled, model_path='unused', artifact_sha256='invalid')
    assert router.predict_route(None) == {'status':'off','routerArtifactSha256':None,'route':None,'reason':None}
router = EvidenceRouter(enabled='typo', model_path='unused', artifact_sha256='a'*64)
assert router.reason == 'invalid_evidence_router_config'
router = EvidenceRouter(enabled=True, model_path=sys.argv[2], artifact_sha256=sys.argv[3])
assert router.reason == 'model_unavailable'
assert router.predict_route('text')['route'] is None
assert not {'torch','transformers','numpy','safetensors','config','server','py3langid'} & sys.modules.keys()
"""
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                "-c",
                script,
                str(Path(module.__file__).parent),
                str(self.directory),
                self.sha,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with patch.object(module, "_verify_artifact") as verify:
            for overrides in (
                {"enabled": 1},
                {"artifact_sha256": "A" * 64},
                {"artifact_sha256": None},
                {"model_path": ""},
            ):
                with self.subTest(overrides=overrides):
                    self.assertEqual(
                        self.load(**overrides).reason, "invalid_evidence_router_config"
                    )
            verify.assert_not_called()

    def test_directory_and_file_integrity_fail_before_model_loading(self):
        for fault in (
            "missing",
            "extra",
            "directory",
            "empty",
            "changed",
            "sha",
            "reparse",
        ):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                write_artifact(directory)
                digest = artifact_sha(directory)
                target = directory / "router_model.safetensors"
                if fault == "missing":
                    target.unlink()
                elif fault == "extra":
                    (directory / "unexpected.bin").write_bytes(b"x")
                elif fault == "directory":
                    target.unlink()
                    target.mkdir()
                elif fault == "empty":
                    target.write_bytes(b"")
                elif fault == "changed":
                    target.write_bytes(b"wrong bytes")
                elif fault == "sha":
                    digest = "a" * 64
                original_lstat = Path.lstat

                def lstat(path):
                    if fault == "reparse" and path == directory:
                        return SimpleNamespace(
                            st_file_attributes=module.stat.FILE_ATTRIBUTE_REPARSE_POINT
                        )
                    return original_lstat(path)

                with (
                    patch.object(Path, "lstat", lstat),
                    patch.object(module, "_load_model") as load,
                ):
                    router = self.load(model_path=directory, artifact_sha256=digest)
                    self.assertEqual(router.reason, "model_unavailable")
                    self.assertIsNone(router.artifact_sha256)
                    load.assert_not_called()

    def test_schema_and_class_order_are_validated_even_with_matching_sha(self):
        for fault in (
            "bool",
            "version",
            "classes",
            "extra",
            "duplicate",
            "noncanonical",
            "nonfinite",
            "model_binding",
        ):
            with self.subTest(fault=fault):
                write_artifact(self.directory)
                target = self.directory / "router_config.json"
                value = json.loads(target.read_bytes())
                if fault == "bool":
                    value["schema_version"] = True
                elif fault == "version":
                    value["schema_version"] = 2
                elif fault == "classes":
                    value["classes"].reverse()
                elif fault == "extra":
                    value["unrecognized"] = True
                target.write_bytes(encode(value))
                if fault == "duplicate":
                    target.write_bytes(
                        target.read_bytes().replace(
                            b'"schema_version": 1,',
                            b'"schema_version": 1, "schema_version": 1,',
                        )
                    )
                elif fault == "noncanonical":
                    target.write_bytes(json.dumps(value).encode())
                elif fault == "nonfinite":
                    target.write_bytes(
                        target.read_bytes().replace(
                            b'"schema_version": 1,', b'"schema_version": NaN,'
                        )
                    )
                elif fault == "model_binding":
                    value = self.config.copy()
                    value["label2id"] = {**value["label2id"], "ar:academic": False}
                    (self.directory / "config.json").write_bytes(encode(value))
                with patch.object(module, "_load_model") as load:
                    router = self.load(artifact_sha256=artifact_sha(self.directory))
                    self.assertEqual(router.reason, "model_unavailable")
                    load.assert_not_called()

    def test_loading_is_once_and_results_do_not_alias_inputs_or_instance(self):
        tokenizer, model = XLMRobertaTokenizerFast(), FakeModel()
        with (
            patch.object(
                module, "_verify_artifact", wraps=module._verify_artifact
            ) as verify,
            patch.object(
                module, "_load_model", return_value=(tokenizer, model)
            ) as load,
        ):
            router = self.load()
            self.assertEqual(router.status, "ready")
            first = router.predict_route("  Ａ\r\nB\u00a0C  ")
            second = router.predict_route("next")
            self.assertEqual(verify.call_count, 1)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(self.identifier_loader.call_count, 1)
            self.assertEqual(self.identifier.calls, ["", "A B C", "next"])
        self.assertEqual(model.calls, 2)
        self.assertEqual(tokenizer.calls[0][0], ["A B C"])
        self.assertEqual(
            tokenizer.calls[0][1],
            {
                "add_special_tokens": False,
                "truncation": False,
                "padding": False,
                "return_attention_mask": False,
                "return_token_type_ids": False,
            },
        )
        self.assertEqual(first["status"], "predicted")
        self.assertEqual(first["routerArtifactSha256"], self.sha)
        self.assertEqual(first["route"]["language"], "ar")
        self.assertEqual(first["route"]["domain"], "academic")
        self.assertAlmostEqual(first["route"]["confidence"]["language"], 1 / 8)
        self.assertAlmostEqual(first["route"]["confidence"]["domain"], 1 / 6)
        first["route"]["confidence"]["language"] = -1
        self.assertGreater(second["route"]["confidence"]["language"], 0)
        self.assertEqual(
            set(second), {"status", "routerArtifactSha256", "route", "reason"}
        )
        json.dumps(second, allow_nan=False)

    def test_load_failure_is_cached_and_changed_directory_is_rejected(self):
        for fault in ("loader", "changed"):
            with self.subTest(fault=fault):
                write_artifact(self.directory)

                def load(*args):
                    if fault == "loader":
                        raise RuntimeError("private directory or dependency detail")
                    (self.directory / "tokenizer.json").write_bytes(
                        b"changed after verification"
                    )
                    return XLMRobertaTokenizerFast(), FakeModel()

                with patch.object(module, "_load_model", side_effect=load) as loader:
                    router = self.load()
                    for _ in range(2):
                        result = router.predict_route("private input")
                        self.assertEqual(
                            result,
                            {
                                "status": "failed",
                                "routerArtifactSha256": None,
                                "route": None,
                                "reason": "model_unavailable",
                            },
                        )
                    self.assertEqual(loader.call_count, 1)
                self.assertIsNone(router._model)
                self.assertIsNone(router._tokenizer)

    def test_invalid_text_skips_tokenizer_and_model_and_inference_failure_recovers(
        self,
    ):
        tokenizer, model = XLMRobertaTokenizerFast(), FakeModel()
        with patch.object(module, "_load_model", return_value=(tokenizer, model)):
            router = self.load()
        for text in (None, True, 1, [], "", " \r\n\u00a0", "x" * 20001):
            with self.subTest(text_type=type(text).__name__):
                self.assertEqual(router.predict_route(text)["reason"], "invalid_text")
        self.assertEqual(tokenizer.calls, [])
        self.assertEqual(model.calls, 0)
        model.failure = True
        result = router.predict_route("private input")
        self.assertEqual(result["reason"], "model_failure")
        self.assertIsNone(result["route"])
        self.assertEqual(router.status, "ready")
        model.failure = False
        result = router.predict_route("word " * 4000)
        self.assertEqual(result["status"], "predicted")
        self.assertEqual(tokenizer.calls[-1][0], [" ".join(["word"] * 4000)])

    def test_language_decisions_skip_xlmr_and_preserve_supported_routes(self):
        tokenizer, model = XLMRobertaTokenizerFast(), FakeModel()
        with patch.object(module, "_load_model", return_value=(tokenizer, model)):
            router = self.load()
        for language in (
            "it",
            "nl",
            "uk",
            "fa",
            "ja",
            "ko",
            "hi",
            "tr",
            "wuu",
            "yue",
            "ary",
            "arz",
        ):
            self.identifier.ranking = [(language, 0.8), ("en", 0.2)]
            with self.subTest(language=language):
                result = router.predict_route("original text")
                self.assertEqual(result["status"], "unsupported")
                self.assertEqual(result["reason"], "unsupported_language")
                self.assertEqual(result["routerArtifactSha256"], self.sha)
                self.assertIsNone(result["route"])
        for ranking in (self.identifier.empty, [("und", 0.1)], [("zxx", 0.9)]):
            self.identifier.ranking = ranking
            result = router.predict_route("no")
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["reason"], "language_undetermined")
        self.assertEqual(model.calls, 0)
        self.assertEqual(tokenizer.calls, [])
        for language in LANGUAGES:
            self.identifier.ranking = [(language, 0.02), ("ja", 0.01)]
            result = router.predict_route("  A\nＢ  ")
            self.assertEqual(result["status"], "predicted")
            self.assertEqual(result["route"]["language"], "ar")
            self.assertAlmostEqual(result["route"]["confidence"]["language"], 1 / 8)
        self.assertEqual(model.calls, 8)
        self.assertEqual(self.identifier.calls[-8:], ["A B"] * 8)

    def test_language_errors_are_private_and_request_failure_recovers(self):
        with (
            patch.object(
                module,
                "_load_language_identifier",
                side_effect=ImportError("private path"),
            ) as load,
            patch.object(module, "_load_model") as model_load,
        ):
            router = self.load()
            self.assertEqual(router.reason, "model_unavailable")
            self.assertIsNone(router.artifact_sha256)
            router.predict_route("first")
            router.predict_route("second")
            load.assert_called_once()
            model_load.assert_not_called()
        tokenizer, model = XLMRobertaTokenizerFast(), FakeModel()
        with patch.object(module, "_load_model", return_value=(tokenizer, model)):
            router = self.load()
        self.identifier.failure = True
        result = router.predict_route("secret")
        self.assertEqual(result["reason"], "model_failure")
        self.assertIsNone(result["route"])
        self.assertNotIn("secret", json.dumps(result))
        self.identifier.failure = False
        for ranking in ([], [("en", float("nan"))], [("en", float("inf"))]):
            self.identifier.ranking = ranking
            self.assertEqual(router.predict_route("next")["reason"], "model_failure")
        self.assertEqual(model.calls, 0)
        self.identifier.ranking = [("en", 0.9)]
        self.assertEqual(router.predict_route("recovered")["status"], "predicted")

    def test_real_py3langid_rank_and_no_information_contract(self):
        if importlib.util.find_spec("py3langid") is None:
            self.skipTest("Use existing isolated py3langid package on test path")
        identifier = self.real_load_identifier()
        self.assertGreater(len(identifier.labels), 8)
        self.assertIsNone(identifier.min_confidence)
        tokenizer, model = XLMRobertaTokenizerFast(), FakeModel()
        with (
            patch.object(module, "_load_language_identifier", return_value=identifier),
            patch.object(module, "_load_model", return_value=(tokenizer, model)),
        ):
            router = self.load()
        cases = [
            (
                "The library will open every Sunday so students can read and study together.",
                "predicted",
            ),
            (
                "La biblioteca sarà aperta anche la domenica e gli studenti potranno studiare insieme.",
                "unsupported",
            ),
            (
                "La bibliothèque ouvrira le dimanche pour permettre aux étudiants de travailler ensemble.",
                "predicted",
            ),
            (
                "Die Bibliothek ist auch am Sonntag geöffnet, damit die Studierenden gemeinsam lernen können.",
                "predicted",
            ),
            (
                "La biblioteca abrirá los domingos para que los estudiantes puedan estudiar juntos.",
                "predicted",
            ),
            (
                "A biblioteca estará aberta aos domingos para que os estudantes possam estudar juntos.",
                "predicted",
            ),
            (
                "Библиотека будет открыта по воскресеньям, чтобы студенты могли заниматься вместе.",
                "predicted",
            ),
            (
                "ستفتح المكتبة أبوابها يوم الأحد حتى يتمكن الطلاب من القراءة والدراسة معًا.",
                "predicted",
            ),
            (
                "市图书馆下个月开始在星期日开放，学生可以预约自习室，一起阅读和学习。",
                "predicted",
            ),
        ]
        for text, expected in cases:
            with self.subTest(text=text[:20]):
                self.assertEqual(identifier.rank(text)[0], identifier.classify(text))
                self.assertEqual(router.predict_route(text)["status"], expected)
        for text in (
            "no",
            "!? ... --- ### *** @@@ !!!",
            "12345 67890 2026 09 07 42 3.14159",
        ):
            with self.subTest(text=text):
                result = router.predict_route(text)
                self.assertEqual(result["reason"], "language_undetermined")
                self.assertIsNone(result["route"])
        self.assertEqual(model.calls, 8)

    def test_head_tail_boundary_and_padding_use_literal_expected_tokens(self):
        for count in (2, 510, 511, 2000):
            with self.subTest(count=count):
                tokenizer = XLMRobertaTokenizerFast(list(range(10, 10 + count)))
                batch = module._tokenize_head_tail(tokenizer, "already canonical text")
                expected = (
                    list(range(10, 10 + count))
                    if count <= 510
                    else list(range(10, 265))
                    + list(range(10 + count - 255, 10 + count))
                )
                self.assertEqual(tokenizer.prepared, [0, *expected, 2])
                self.assertLessEqual(batch["input_ids"].shape[1], 512)
                self.assertEqual(batch["input_ids"].shape[1] % 8, 0)
                self.assertEqual(int(batch["attention_mask"].sum()), len(expected) + 2)

    def test_language_first_not_global_argmax_and_nonfinite_logits_fail(self):
        import torch

        probabilities = [
            0.24,
            *([0.0] * 5),
            *([0.05] * 6),
            *([0.04] * 6),
            *([0.03] * 6),
            0.04,
            *([0.0] * 23),
        ]
        # Zero probabilities are represented by finite underflowing logits.
        logits = torch.tensor(
            [[math.log(p) if p else -1000 for p in probabilities]], dtype=torch.float64
        )
        result = module._route_from_logits(logits)
        self.assertEqual(result["language"], "de")
        self.assertEqual(result["domain"], "academic")
        self.assertAlmostEqual(result["confidence"]["language"], 0.3)
        self.assertAlmostEqual(result["confidence"]["domain"], 1 / 6)
        self.assertEqual(int(logits.argmax()), 0)
        for invalid in (
            torch.zeros(48),
            torch.zeros(1, 47),
            torch.ones(1, 48, dtype=torch.int64),
            torch.full((1, 48), float("nan")),
            torch.full((1, 48), float("inf")),
            [[0.0] * 48],
        ):
            with (
                self.subTest(kind=type(invalid).__name__),
                self.assertRaises(ValueError),
            ):
                module._route_from_logits(invalid)

    def test_tiny_real_state_load_preserves_keys_outputs_cpu_eval_and_rng(self):
        import torch
        from safetensors.torch import save_file
        from transformers import AutoTokenizer, XLMRobertaConfig, XLMRobertaModel

        # Build the state independently of the consumer's model factory.
        encoder = XLMRobertaModel(XLMRobertaConfig.from_dict(self.config))
        state = {
            f"encoder.{name}": value for name, value in encoder.state_dict().items()
        }
        state["language_classifier.weight"] = torch.zeros(8, 8)
        state["language_classifier.bias"] = torch.tensor(
            [math.log(i) for i in range(1, 9)]
        )
        for index in range(8):
            state[f"domain_classifiers.{index}.weight"] = torch.zeros(6, 8)
            state[f"domain_classifiers.{index}.bias"] = torch.tensor(
                [math.log(i) for i in range(6, 0, -1)]
            )
        save_file(state, self.directory / "router_model.safetensors")
        tokenizer = XLMRobertaTokenizerFast()
        rng = torch.get_rng_state().clone()
        with (
            patch.object(
                AutoTokenizer, "from_pretrained", return_value=tokenizer
            ) as load,
            patch.object(
                torch.cuda,
                "_lazy_init",
                side_effect=AssertionError("must not initialize CUDA"),
            ),
        ):
            router = self.load(artifact_sha256=artifact_sha(self.directory))
            self.assertEqual(router.status, "ready", router.reason)
            load.assert_called_once_with(
                self.directory,
                local_files_only=True,
                trust_remote_code=False,
                use_fast=True,
            )
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertFalse(router._model.training)
            self.assertEqual(set(router._model.state_dict()), set(state))
            self.assertTrue(
                all(
                    p.device.type == "cpu"
                    and p.dtype == torch.float32
                    and not p.requires_grad
                    for p in router._model.parameters()
                )
            )
            result = router.predict_route("test input")
        self.assertEqual(result["route"]["language"], "zh")
        self.assertEqual(result["route"]["domain"], "academic")
        self.assertAlmostEqual(
            result["route"]["confidence"]["language"], 8 / 36, places=7
        )
        self.assertAlmostEqual(
            result["route"]["confidence"]["domain"], 6 / 21, places=7
        )
        with torch.inference_mode():
            joint = router._model(**module._tokenize_head_tail(tokenizer, "test"))[
                "logits"
            ].exp()
        expected = torch.tensor(
            [
                [
                    language * domain / (36 * 21)
                    for language in range(1, 9)
                    for domain in range(6, 0, -1)
                ]
            ]
        )
        torch.testing.assert_close(joint, expected)
        for fault in ("missing", "extra", "shape"):
            altered = state.copy()
            if fault == "missing":
                altered.pop("domain_classifiers.7.bias")
            elif fault == "extra":
                altered["unexpected.weight"] = torch.zeros(1)
            else:
                altered["language_classifier.bias"] = torch.zeros(7)
            save_file(altered, self.directory / "router_model.safetensors")
            with (
                self.subTest(fault=fault),
                patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer),
            ):
                self.assertEqual(
                    self.load(artifact_sha256=artifact_sha(self.directory)).reason,
                    "model_unavailable",
                )

    def test_tokenizer_binding_failure_precedes_encoder_construction(self):
        from transformers import AutoTokenizer, XLMRobertaModel

        for key, value in (
            ("is_fast", False),
            ("model_max_length", 1024),
            ("mask_token_id", 9),
        ):
            tokenizer = XLMRobertaTokenizerFast()
            setattr(tokenizer, key, value)
            with (
                self.subTest(key=key),
                patch.object(AutoTokenizer, "from_pretrained", return_value=tokenizer),
                patch.object(XLMRobertaModel, "__init__") as construct,
            ):
                self.assertEqual(self.load().reason, "model_unavailable")
                construct.assert_not_called()

    def test_optional_environment_config_is_raw_and_does_not_load_router(self):
        environment = os.environ.copy()
        environment.update(
            REPRE_GUARD_EVIDENCE_ENABLED="typo",
            REPRE_GUARD_EVIDENCE_MODEL_PATH="missing",
            REPRE_GUARD_EVIDENCE_ARTIFACT_SHA256="bad",
            REPRE_GUARD_EVIDENCE_TIMEOUT_SECONDS="nan",
        )
        script = """
import sys
from config import settings
assert settings.evidence_enabled == 'typo'
assert settings.evidence_model_path == 'missing'
assert settings.evidence_artifact_sha256 == 'bad'
assert settings.evidence_timeout_seconds == 'nan'
assert 'evidence_router' not in sys.modules and 'torch' not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            env=environment,
            cwd=Path(module.__file__).parent,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
