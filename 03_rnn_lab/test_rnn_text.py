"""文本 CPU 回归；人工字符串仅为单元测试夹具，不属于正式评论数据。"""

from dataclasses import replace
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from rnn_lab.text_data import (PAD, UNK, TextDataBundle, build_vocab, clean_text, encode_text,
    load_text_data, load_text_preprocessing, make_text_dataloaders, save_text_preprocessing,
    text_data_diagnostics)
from rnn_lab.text_engine import (TextExperimentConfig, evaluate_text_final, export_text_experiments_csv,
    predict_text, run_text_experiment, save_text_run, text_experiment_records)
from rnn_lab.text_model import TextClassifier, masked_mean, valid_last
from rnn_lab.text_observation import observe_text_recurrence
from rnn_lab.text_plots import (plot_text_comparison, plot_text_confusion, plot_text_history,
                                plot_text_states, text_examples)


def fixture_data():
    # 测试用的可学习规则与真实评论完全分开；只证明梯度/标签/接口有用。
    def split(prefix):
        return [{"id": f"{prefix}-{index}", "text": char * length, "label": label}
                for index, (label, char, length) in enumerate(
                    [(label, char, length) for length in (1, 2, 3, 4) for label, char in ((0, "差"), (1, "好"))])]
    return TextDataBundle({name: split(name) for name in ("train", "val", "test")}, version="unit-fixture-only")


class TextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_clean_encoding_and_training_only_vocab(self):
        vocab = build_vocab(["不好！", "好好", "房间差"])
        self.assertEqual(vocab[PAD], 0)
        self.assertEqual(vocab[UNK], 1)
        self.assertEqual(vocab, build_vocab(reversed(["不好！", "好好", "房间差"])))
        self.assertEqual(clean_text("  不好！\n但还行  "), "不好! 但还行")
        unseen = encode_text("𠮷陌", vocab)
        self.assertEqual(unseen["token_ids"], [1, 1])
        self.assertNotIn("𠮷", vocab)
        empty = encode_text(" \n\t", vocab)
        self.assertTrue(empty["empty"])
        self.assertEqual((empty["token_ids"], empty["length"], empty["original_length"]), ([1], 1, 0))
        long = encode_text("好" * 11, vocab, 3)
        self.assertTrue(long["truncated"])
        self.assertEqual(long["length"], 3)
        with self.assertRaises(ValueError):
            encode_text("好", vocab, 0)
        data = fixture_data()
        data.splits["val"][0]["text"] = "陌生字"
        loaders = make_text_dataloaders(data, vocab, max_length=3, batch_size=8)
        batch = next(iter(loaders["val"]))
        self.assertEqual(batch["token_ids"].dtype, torch.int64)
        self.assertEqual(batch["lengths"].dtype, torch.int64)
        self.assertEqual(batch["labels"].dtype, torch.int64)
        self.assertEqual(tuple(batch["token_ids"].shape), (8, 3))
        self.assertEqual(batch["lengths"][1].item(), 1)
        self.assertTrue(torch.equal(batch["token_ids"][1, 1:], torch.zeros(2, dtype=torch.int64)))
        self.assertGreater(text_data_diagnostics(data, vocab, 3)["val"]["unknown_count"], 0)
        with TemporaryDirectory() as tmp:
            path = save_text_preprocessing(vocab, 3, data.class_names, Path(tmp) / "preprocessing.json")
            restored = load_text_preprocessing(path)
            self.assertEqual(restored["vocab"], vocab)
            self.assertEqual(encode_text("不好", restored["vocab"], restored["max_length"]), encode_text("不好", vocab, 3))

    def test_effective_last_and_masked_mean(self):
        values = torch.tensor([[[1., 2.], [3., 4.], [900., 900.]], [[5., 6.], [800., 800.], [700., 700.]]])
        lengths = torch.tensor([2, 1])
        torch.testing.assert_close(valid_last(values, lengths), torch.tensor([[3., 4.], [5., 6.]]))
        torch.testing.assert_close(masked_mean(values, lengths), torch.tensor([[2., 3.], [5., 6.]]))
        self.assertFalse(torch.allclose(valid_last(values, lengths), values[:, -1]))
        for bad in (torch.tensor([0, 1]), torch.tensor([4, 1]), torch.tensor([2., 1.])):
            with self.assertRaises(ValueError):
                valid_last(values, bad)

    def test_padding_and_batch_invariance_all_cells_readouts(self):
        for cell in ("rnn", "lstm", "gru", "mean_embedding"):
            for readout in ("last", "mean"):
                for layers in (1, 2):
                    torch.manual_seed(7)
                    model = TextClassifier(11, 3, embedding_dim=5, hidden_size=7,
                                           cell_type=cell, num_layers=layers, dropout=.25, readout=readout).eval()
                    solo = torch.tensor([[2, 3]], dtype=torch.int64)
                    padded = torch.tensor([[2, 3, 0, 0, 0]], dtype=torch.int64)
                    together = torch.tensor([[2, 3, 0, 0, 0], [4, 5, 6, 7, 8]], dtype=torch.int64)
                    with torch.no_grad():
                        prediction = model(solo, torch.tensor([2]))
                        torch.testing.assert_close(prediction, model(padded, torch.tensor([2])), rtol=1e-5, atol=1e-6)
                        torch.testing.assert_close(prediction, model(together, torch.tensor([2, 5]))[:1], rtol=1e-5, atol=1e-6)
                    self.assertEqual(tuple(prediction.shape), (1, 3))
                    self.assertTrue(bool(torch.isfinite(prediction).all()))

    def test_learning_export_predict_observe_and_cache(self):
        data = fixture_data()
        results = []
        for cell in ("rnn", "lstm", "gru", "mean_embedding"):
            config = TextExperimentConfig(name=f"fixture {cell}", cell_type=cell, embedding_dim=6,
                hidden_size=8, max_length=5, epochs=18, batch_size=8, learning_rate=.03)
            result = run_text_experiment(config, data)
            self.assertIsNone(result["metrics"])
            self.assertLess(result["history"]["train_loss"][-1], result["history"]["train_loss"][0])
            self.assertEqual(set(result["val_metrics"]["predictions"].tolist()), {0, 1})
            self.assertEqual(torch.get_num_threads(), 1)
            prediction = predict_text(result, "好" * 8)
            self.assertTrue(prediction["truncated"])
            self.assertEqual(prediction["length"], 5)
            self.assertAlmostEqual(sum(prediction["probabilities"].values()), 1, places=6)
            if cell != "mean_embedding":
                model = result["model"]
                model.train()
                model.classifier.eval()
                for index, parameter in enumerate(model.parameters()):
                    parameter.grad = torch.full_like(parameter, .25) if index % 2 else None
                before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
                gradients = [None if parameter.grad is None else parameter.grad.clone() for parameter in model.parameters()]
                modes = [module.training for module in model.modules()]
                hooks = len(model.recurrent._forward_hooks)
                trace = observe_text_recurrence(result, "好差好")
                self.assertEqual(trace["characters"], ["好", "差", "好"])
                self.assertEqual(tuple(trace["hidden_states"].shape), (3, 8))
                self.assertLess(trace["replay_max_error"], 1e-5)
                self.assertEqual(trace["cell_states"] is not None, cell == "lstm")
                self.assertEqual(hooks, len(model.recurrent._forward_hooks))
                self.assertEqual(modes, [module.training for module in model.modules()])
                for name, tensor in model.state_dict().items():
                    torch.testing.assert_close(before[name], tensor, rtol=0, atol=0)
                for gradient, parameter in zip(gradients, model.parameters()):
                    if gradient is None:
                        self.assertIsNone(parameter.grad)
                    else:
                        torch.testing.assert_close(gradient, parameter.grad, rtol=0, atol=0)
                plt.close(plot_text_states(trace))
            results.append(result)
        self.assertEqual(len(text_examples(results[0], correct=True)), 5)
        plt.close(plot_text_comparison(results))
        plt.close(plot_text_history(results[0]))
        plt.close(plot_text_confusion(results[0]))
        with self.assertRaises(ValueError):
            plot_text_confusion(results[0], split="test")
        with TemporaryDirectory() as tmp:
            csv_path = export_text_experiments_csv(results, Path(tmp) / "compare.csv")
            with csv_path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["test_accuracy"], "")
            self.assertEqual(rows[0]["test_evaluated"], "False")
            required = {"task", "name", "data_version", "seed", "cell_type", "num_layers", "hidden_size", "embedding_dim",
                        "vocab_size", "max_length", "readout", "n_classes", "train_size", "val_size", "test_size",
                        "parameter_count", "epochs", "batch_size", "learning_rate", "val_loss", "val_accuracy",
                        "elapsed_seconds", "test_evaluated", "test_loss", "test_accuracy"}
            self.assertTrue(required.issubset(rows[0]))
            run1 = save_text_run(results[0], tmp)
            run2 = save_text_run(results[0], tmp)
            self.assertNotEqual(run1, run2)
            self.assertTrue((run1 / "model.pt").is_file())
            self.assertEqual(json.loads((run1 / "metrics.json").read_text())["test"], None)
        tested = results[0]
        first = evaluate_text_final(tested)
        self.assertIs(first, evaluate_text_final(tested))
        plt.close(plot_text_confusion(tested, split="test"))
        with torch.no_grad():
            next(tested["model"].parameters()).add_(.001)
        with self.assertRaises(RuntimeError):
            evaluate_text_final(tested)
        with self.assertRaises(RuntimeError):
            text_experiment_records([tested])
        with self.assertRaises(RuntimeError):
            plot_text_confusion(tested, split="test")

    def test_offline_file_validation_and_actual_bundle_if_available(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(json.dumps({"version": "unit-fixture", "class_names": ["负面", "正面"]}), encoding="utf-8")
            for split, text in (("train", "训练文本"), ("val", "验证文本"), ("test", "测试文本")):
                with (root / f"{split}.tsv").open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["id", "text", "label"], delimiter="\t")
                    writer.writeheader()
                    writer.writerow({"id": split, "text": text, "label": 0})
            loaded = load_text_data(root)
            self.assertEqual(len(loaded.splits["val"]), 1)
            (root / "test.tsv").write_text("id\ttext\tlabel\ntest\t训练文本\t1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_text_data(root)
        actual_dir = Path(__file__).parent / "06_text_classification" / "data"
        if (actual_dir / "train.tsv").is_file():
            actual = load_text_data(actual_dir)
            vocab = build_vocab(row["text"] for row in actual.splits["train"])
            self.assertNotIn(PAD, clean_text(actual.splits["train"][0]["text"]))
            self.assertGreater(len(vocab), 2)
            for split in ("train", "val", "test"):
                self.assertTrue(actual.splits[split])
                self.assertEqual({row["label"] for row in actual.splits[split]}, {0, 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
