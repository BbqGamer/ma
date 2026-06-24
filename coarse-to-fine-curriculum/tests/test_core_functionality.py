from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ctf.models import build_model
from scripts.analyze_pareto import add_baseline_thresholds, add_pareto_columns, normalized_auc
from scripts.plan_figure11_resnet18 import parse_args as parse_plan_args
from scripts.plan_figure11_resnet18 import build_command
from train_coarse_to_fine import (
    classification_metrics_from_confusion,
    estimate_roughness_metrics,
    hierarchy_distance_matrix_from_levels,
    marginalized_loss,
    parse_weight_list,
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

    def test_parse_weight_list_pads_and_truncates(self) -> None:
        self.assertEqual(parse_weight_list("1,2", 4), [1.0, 2.0, 2.0, 2.0])
        self.assertEqual(parse_weight_list("1,2,3", 2), [1.0, 2.0])
        with self.assertRaises(ValueError):
            parse_weight_list("1,0", 2)


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
