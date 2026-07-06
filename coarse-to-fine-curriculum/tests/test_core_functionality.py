from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest

from ctf.hierarchy import compute_hierarchy
from ctf.models import build_model
import numpy as np
import pandas as pd
from scripts.analyze_pareto import add_baseline_thresholds, add_pareto_columns, normalized_auc
from scripts.analyze_teacher_hierarchy_suite import condition_label, summarize_suite
from scripts.plan_figure11_resnet18 import build_command
from scripts.plan_figure11_resnet18 import parse_args as parse_plan_args
from scripts.plan_teacher_hierarchy_suite import build_plan
from scripts.plan_teacher_hierarchy_suite import parse_args as parse_teacher_plan_args
import torch
from train_coarse_to_fine import (
    build_curriculum_schedule,
    classification_metrics_from_confusion,
    clusters_to_membership,
    estimate_roughness_metrics,
    hierarchy_distance_matrix_from_levels,
    marginalized_loss,
    parse_weight_list,
    random_permutation_hierarchy,
    seed_everything,
    teacher_embedding_distance,
)


class ModelScalingTests(unittest.TestCase):
    def test_cnn_width_multiplier_changes_parameter_count_and_spec(self) -> None:
        small = build_model("cnn", (3, 32, 32), 10, cnn_width_multiplier=0.5)
        base = build_model("cnn", (3, 32, 32), 10, cnn_width_multiplier=1.0)
        large = build_model("cnn", (3, 32, 32), 10, cnn_width_multiplier=2.0)

        n_small = sum(p.numel() for p in small.parameters())
        n_base = sum(p.numel() for p in base.parameters())
        n_large = sum(p.numel() for p in large.parameters())

        self.assertLess(n_small, n_base)
        self.assertLess(n_base, n_large)
        self.assertEqual(small.spec.name, "cnn_w0.5")
        self.assertEqual(large(torch.randn(2, 3, 32, 32)).shape, (2, 10))

    def test_cifar_resnet_depth_and_width_are_configurable(self) -> None:
        resnet8 = build_model("cifar_resnet8", (3, 32, 32), 100)
        resnet20 = build_model("cifar_resnet20", (3, 32, 32), 100)
        resnet20_w2 = build_model("cifar_resnet20", (3, 32, 32), 100, cifar_resnet_width_multiplier=2.0)

        n8 = sum(p.numel() for p in resnet8.parameters())
        n20 = sum(p.numel() for p in resnet20.parameters())
        n20_w2 = sum(p.numel() for p in resnet20_w2.parameters())

        self.assertLess(n8, n20)
        self.assertLess(n20, n20_w2)
        self.assertEqual(resnet20.spec.name, "cifar_resnet20_w1")
        self.assertEqual(resnet20_w2(torch.randn(2, 3, 32, 32)).shape, (2, 100))

    def test_invalid_cifar_resnet_depth_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_model("cifar_resnet10", (3, 32, 32), 10)


class LossAndMetricTests(unittest.TestCase):
    def test_clusters_to_membership_expands_cluster_mask_to_all_rows(self) -> None:
        membership = clusters_to_membership([[0, 2], [1]], 3, torch.device("cpu"))
        expected = torch.tensor(
            [
                [True, False, True],
                [False, True, False],
                [True, False, True],
            ]
        )
        self.assertTrue(torch.equal(membership.cpu(), expected))

    def test_marginalized_loss_matches_manual_logsumexp(self) -> None:
        logits = torch.tensor([[2.0, 1.0, -1.0], [0.0, 2.0, 3.0]])
        targets = torch.tensor([0, 1])
        membership = torch.tensor(
            [
                [True, True, False],
                [False, True, True],
                [False, True, True],
            ]
        )
        got = marginalized_loss(logits, targets, membership)
        log_probs = torch.log_softmax(logits, dim=1)
        expected = -torch.stack(
            [
                torch.logsumexp(log_probs[0, [0, 1]], dim=0),
                torch.logsumexp(log_probs[1, [1, 2]], dim=0),
            ]
        ).mean()
        self.assertTrue(torch.allclose(got, expected))

    def test_classification_metrics_from_confusion(self) -> None:
        confusion = torch.tensor([[2, 0], [1, 1]])
        metrics = classification_metrics_from_confusion(confusion)
        self.assertAlmostEqual(metrics["recall_macro"], 0.75)
        self.assertAlmostEqual(metrics["balanced_acc"], 0.75)
        self.assertIn("precision_macro", metrics)
        self.assertIn("f1_macro", metrics)

    def test_hierarchy_distance_matrix_from_levels(self) -> None:
        levels = [
            [[0, 1, 2], [3]],
            [[0, 1], [2], [3]],
        ]
        dist = hierarchy_distance_matrix_from_levels(levels, 4)
        self.assertEqual(dist.shape, (4, 4))
        self.assertEqual(dist[0, 0], 0.0)
        self.assertLess(dist[0, 1], dist[0, 2])
        self.assertGreaterEqual(dist.min(), 0.0)
        self.assertLessEqual(dist.max(), 1.0)

    def test_random_permutation_hierarchy_preserves_shape_and_seed(self) -> None:
        levels = [
            [[0, 1, 2], [3]],
            [[0, 1], [2], [3]],
        ]
        first, first_perm = random_permutation_hierarchy(levels, num_classes=4, seed=11)
        second, second_perm = random_permutation_hierarchy(levels, num_classes=4, seed=11)
        other, _ = random_permutation_hierarchy(levels, num_classes=4, seed=12)
        self.assertEqual(first, second)
        self.assertEqual(first_perm, second_perm)
        self.assertNotEqual(first, other)
        self.assertEqual(
            [[len(cluster) for cluster in level] for level in first],
            [[3, 1], [2, 1, 1]],
        )
        for level in first:
            self.assertEqual(sorted(item for cluster in level for item in cluster), [0, 1, 2, 3])

    def test_parse_weight_list_pads_and_truncates(self) -> None:
        self.assertEqual(parse_weight_list("1,2", 4), [1.0, 2.0, 2.0, 2.0])
        self.assertEqual(parse_weight_list("1,2,3", 2), [1.0, 2.0])
        with self.assertRaises(ValueError):
            parse_weight_list("1,0", 2)

    def test_teacher_embedding_distance_uses_class_prototypes(self) -> None:
        class DummyTeacher(torch.nn.Module):
            def forward_features(self, x: torch.Tensor) -> torch.Tensor:
                return x

        features = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        )
        labels = torch.tensor([0, 0, 1, 1])
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features, labels), batch_size=2)
        dist = teacher_embedding_distance(DummyTeacher(), loader, num_classes=2, device=torch.device("cpu"))
        self.assertEqual(dist.shape, (2, 2))
        self.assertAlmostEqual(float(dist[0, 0]), 0.0)
        self.assertAlmostEqual(float(dist[1, 1]), 0.0)
        self.assertGreater(float(dist[0, 1]), 0.8)

    def test_build_model_supports_pretrained_resnet_backbone(self) -> None:
        model = build_model("resnet18", (3, 32, 32), 10, pretrained_backbone=True)
        output = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(output.shape, (2, 10))


class RoughnessProbeTests(unittest.TestCase):
    def test_roughness_probe_returns_expected_keys(self) -> None:
        torch.manual_seed(0)
        model = build_model("cnn", (3, 32, 32), 4, cnn_width_multiplier=0.25)
        probe_batches = [(torch.randn(2, 3, 32, 32), torch.tensor([0, 1]))]
        args = argparse.Namespace(
            multi_weighting="static",
            multi_static_weights="1,1,1,1",
            sharpness_rho=0.001,
            hessian_iters=1,
            hessian_samples=1,
        )
        metrics = estimate_roughness_metrics(
            model=model,
            probe_batches=probe_batches,
            membership=None,
            multiloss_memberships=None,
            args=args,
            adaptive_log_weights=None,
        )
        expected = {
            "rough_grad_norm_mean",
            "rough_grad_norm_skew",
            "rough_gradient_noise_scale",
            "rough_critical_sharpness",
            "rough_relative_critical_sharpness",
            "rough_hessian_top_eigenvalue",
            "rough_hessian_frobenius",
            "rough_hessian_trace",
        }
        self.assertTrue(expected.issubset(metrics.keys()))
        for key in expected:
            self.assertTrue(np.isfinite(metrics[key]), key)


class ReproducibilityTests(unittest.TestCase):
    def test_seed_everything_enables_deterministic_torch(self) -> None:
        info = seed_everything(123, deterministic=True)
        self.assertTrue(info["deterministic"])
        self.assertTrue(torch.are_deterministic_algorithms_enabled())
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertTrue(torch.backends.cudnn.deterministic)

    def test_hierarchy_tie_breaking_is_seeded(self) -> None:
        dist = np.ones((5, 5), dtype=np.float32) - np.eye(5, dtype=np.float32)
        first = compute_hierarchy(dist, seed=7)
        second = compute_hierarchy(dist, seed=7)
        other = compute_hierarchy(dist, seed=8)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)


class PlannerTests(unittest.TestCase):
    def test_plan_command_includes_dataset_model_widths_and_reference_dir(self) -> None:
        args = parse_plan_args(
            [
                "--dataset",
                "fashion-mnist",
                "--model",
                "cnn",
                "--cnn-width-multiplier",
                "0.5",
                "--output-dir",
                "/runs",
            ]
        )
        command, row = build_command(args, "curriculum", 10)
        self.assertIn("--dataset fashion-mnist", command)
        self.assertIn("--cnn-width-multiplier 0.5", command)
        self.assertIn("fig11-cnn-w0.5-fashion-mnist-seed42-curr10", command)
        self.assertIn("/runs/fig11-cnn-w0.5-fashion-mnist-seed42-baseline", command)
        self.assertEqual(row["dataset"], "fashion-mnist")

    def test_plan_command_includes_roughness_flags(self) -> None:
        args = parse_plan_args(["--roughness-probes", "--roughness-epochs", "1,2"])
        command, _ = build_command(args, "baseline", None)
        self.assertIn("--roughness-probes", command)
        self.assertIn("--roughness-epochs 1,2", command)

    def test_adaptive_schedule_filters_levels_by_cluster_count(self) -> None:
        levels = [
            [[0, 1, 2, 3]],
            [[0, 1], [2, 3]],
            [[0], [1], [2], [3]],
        ]
        schedule = build_curriculum_schedule(
            num_classes=4,
            hierarchy_levels=levels,
            curriculum_epochs=10,
            total_epochs=30,
            min_clusters=2,
            policy="adaptive_plateau",
            stage_max_epochs=7,
        )
        self.assertEqual([item["name"] for item in schedule], ["level_1_2clusters", "fine_tune"])
        self.assertEqual(schedule[0]["epochs"], 7)
        self.assertTrue(schedule[0]["adaptive"])

    def test_hard_to_easy_schedule_reverses_intermediate_levels(self) -> None:
        levels = [
            [[0, 1, 2, 3]],
            [[0, 1], [2, 3]],
            [[0], [1], [2], [3]],
        ]
        schedule = build_curriculum_schedule(
            num_classes=4,
            hierarchy_levels=levels,
            curriculum_epochs=12,
            total_epochs=20,
            policy="fixed",
            curriculum_order="hard_to_easy",
        )
        self.assertEqual(schedule[0]["name"], "level_1_2clusters")
        self.assertEqual(schedule[-1]["name"], "fine_tune")


class TeacherHierarchySuiteTests(unittest.TestCase):
    def test_teacher_suite_condition_label_detects_all_conditions(self) -> None:
        self.assertEqual(condition_label("teacher-cifar100-cnn-w0.5-seed42-baseline", {"mode": "baseline"}), "baseline")
        self.assertEqual(condition_label("teacher-cifar100-cnn-w0.5-seed42-self-curr20", {"mode": "curriculum", "distance_source": "classifier_weights"}), "self")
        self.assertEqual(condition_label("teacher-cifar100-cnn-w0.5-seed42-teacher-curr20", {"mode": "curriculum", "distance_source": "teacher_embeddings"}), "teacher")
        self.assertEqual(condition_label("teacher-cifar100-cnn-w0.5-seed42-teacher-anti-curr20", {"mode": "curriculum", "distance_source": "teacher_embeddings", "curriculum_order": "hard_to_easy"}), "teacher_anti")
        self.assertEqual(condition_label("teacher-cifar100-cnn-w0.5-seed42-random1001-curr20", {"mode": "curriculum", "distance_source": "random_permutation"}), "random")

    def test_teacher_suite_summary_aggregates_random_and_teacher_rows(self) -> None:
        runs = pd.DataFrame(
            [
                {"dataset": "cifar100", "model_label": "cnn_w0.5", "seed": 42, "condition": "baseline", "curriculum_epochs": np.nan, "best_test_acc": 0.35, "final_test_acc": 0.32, "auc_test_acc": 0.30},
                {"dataset": "cifar100", "model_label": "cnn_w0.5", "seed": 42, "condition": "self", "curriculum_epochs": 20, "best_test_acc": 0.375, "final_test_acc": 0.33, "auc_test_acc": 0.28},
                {"dataset": "cifar100", "model_label": "cnn_w0.5", "seed": 42, "condition": "teacher", "curriculum_epochs": 20, "best_test_acc": 0.392, "final_test_acc": 0.35, "auc_test_acc": 0.31},
                {"dataset": "cifar100", "model_label": "cnn_w0.5", "seed": 42, "condition": "teacher_anti", "curriculum_epochs": 20, "best_test_acc": 0.36, "final_test_acc": 0.325, "auc_test_acc": 0.27},
                {"dataset": "cifar100", "model_label": "cnn_w0.5", "seed": 42, "condition": "random", "curriculum_epochs": 20, "best_test_acc": 0.356, "final_test_acc": 0.321, "auc_test_acc": 0.26},
                {"dataset": "cifar100", "model_label": "cnn_w0.5", "seed": 42, "condition": "random", "curriculum_epochs": 20, "best_test_acc": 0.362, "final_test_acc": 0.324, "auc_test_acc": 0.265},
            ]
        )
        curriculum, paired, aggregate = summarize_suite(runs)
        self.assertEqual(len(curriculum), 5)
        self.assertEqual(len(paired), 1)
        pair = paired.iloc[0]
        self.assertAlmostEqual(pair["teacher_best_gain"], 0.042)
        self.assertAlmostEqual(pair["random_mean_best_gain"], 0.009)
        self.assertAlmostEqual(pair["teacher_minus_random_mean_best_gain"], 0.033)
        methods = set(aggregate["method"])
        self.assertIn("Teacher hierarchy", methods)
        self.assertIn("Random hierarchy mean", methods)

    def test_teacher_plan_builds_expected_run_ids(self) -> None:
        args = parse_teacher_plan_args([])
        script_lines, manifest_rows = build_plan(args)
        script = "\n".join(script_lines)
        self.assertIn("EXPERIMENT=teacher_hierarchy_suite", script)
        self.assertIn("TEACHER_HIERARCHY_SPECS=cifar100:cnn:0.5:1.0:20:100", script)
        run_ids = {row["run_id"] for row in manifest_rows}
        self.assertIn("teacher-cifar100-cnn-w0.5-seed42-baseline", run_ids)
        self.assertIn("teacher-cifar100-cnn-w0.5-seed42-teacher-curr20", run_ids)
        self.assertIn("teacher-cifar100-cnn-w0.5-seed44-random1003-curr20", run_ids)


class ParetoAnalysisTests(unittest.TestCase):
    def test_auc_and_pareto_columns(self) -> None:
        history = pd.DataFrame({"epoch": [1, 2, 3], "test_acc": [0.1, 0.2, 0.4]})
        self.assertAlmostEqual(normalized_auc(history, "test_acc"), 0.225)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rows = []
            for label, accs in {
                "baseline": [0.1, 0.2, 0.3],
                "curr10": [0.2, 0.3, 0.35],
                "curr20": [0.05, 0.1, 0.15],
            }.items():
                run_dir = root / label
                run_dir.mkdir()
                pd.DataFrame({"epoch": [1, 2, 3], "test_acc": accs, "val_acc": accs}).to_csv(
                    run_dir / "history.csv",
                    index=False,
                )
                mode = "baseline" if label == "baseline" else "curriculum"
                rows.append(
                    {
                        "run_dir": str(run_dir),
                        "dataset": "toy",
                        "model": "cnn",
                        "seed": 1,
                        "mode": mode,
                        "label": label,
                        "best_test_acc": max(accs),
                        "auc_test_acc": normalized_auc(
                            pd.DataFrame({"epoch": [1, 2, 3], "test_acc": accs}),
                            "test_acc",
                        ),
                        "mean_rough_hessian_top_eigenvalue": 1.0 if label != "curr20" else 5.0,
                        "mean_rough_gradient_noise_scale": 1.0 if label != "curr20" else 5.0,
                    }
                )
            summary = pd.DataFrame(rows)
            summary = add_baseline_thresholds(summary)
            summary = add_pareto_columns(summary)
            curr10 = summary[summary["label"] == "curr10"].iloc[0]
            curr20 = summary[summary["label"] == "curr20"].iloc[0]
            self.assertAlmostEqual(curr10["gain_best_test_acc"], 0.05)
            self.assertTrue(bool(curr10["pareto_accuracy_speed"]))
            self.assertFalse(bool(curr20["pareto_accuracy_speed"]))


if __name__ == "__main__":
    unittest.main()
