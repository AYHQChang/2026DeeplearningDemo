"""检查课堂图中的实际数值与观察边界；人工字符串只作为测试夹具。"""
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from rnn_lab.text_data import TextDataBundle, build_vocab
from rnn_lab.text_engine import TextExperimentConfig, predict_text
from rnn_lab.text_model import TextClassifier
from rnn_lab.teaching_visuals import (plot_embedding_vectors, plot_prefix_predictions,
                                      plot_validation_groups)


class TeachingVisualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.vocab = build_vocab(["好差不"])

    def tearDown(self):
        plt.close("all")

    def test_prefix_curves_match_inference_and_leave_model_untouched(self):
        for readout in ("last", "mean"):
            model = TextClassifier(len(self.vocab), 2, embedding_dim=4, hidden_size=5,
                                   cell_type="lstm", num_layers=2, dropout=.2, readout=readout)
            model.train()
            model.embedding.eval()  # 混合模式也必须原样恢复。
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
            modes = [module.training for module in model.modules()]
            params = [parameter.detach().clone() for parameter in model.parameters()]
            grads = [parameter.grad.clone() for parameter in model.parameters()]
            result = {"model": model, "vocab": self.vocab, "metrics": None,
                      "data": TextDataBundle({}),
                      "config": TextExperimentConfig(cell_type="lstm", readout=readout,
                                                     max_length=3, device="cpu")}
            with patch("rnn_lab.text_engine.evaluate_text_final", side_effect=AssertionError("不能触发测试")):
                fig = plot_prefix_predictions(result, "好不好差")  # 第四字截断。
            lines = fig.axes[0].lines
            self.assertEqual(len(lines), 2)
            expected = [predict_text(result, prefix)["probabilities"] for prefix in ["好", "好不", "好不好"]]
            for line, name in zip(lines, result["data"].class_names):
                np.testing.assert_allclose(line.get_ydata(), [value[name] for value in expected], atol=1e-7)
            self.assertEqual(modes, [module.training for module in model.modules()])
            for parameter, before, gradient in zip(model.parameters(), params, grads):
                torch.testing.assert_close(parameter, before, rtol=0, atol=0)
                torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
            self.assertTrue(all(not module._forward_hooks for module in model.modules()))
            self.assertIsNone(result["metrics"])

    def test_error_groups_follow_ids_and_report_correct_denominators(self):
        rows = [{"id": "a", "text": "好", "label": 1},
                {"id": "b", "text": "差", "label": 0},
                {"id": "c", "text": "不好好", "label": 0},
                {"id": "d", "text": "陌", "label": 1}]
        result = {"data": TextDataBundle({"val": rows}), "vocab": self.vocab,
                  "config": TextExperimentConfig(max_length=2),
                  "val_metrics": {"ids": ["d", "a", "c", "b"],
                                  "predictions": torch.tensor([0, 1, 0, 1])}}
        fig = plot_validation_groups(result)
        np.testing.assert_allclose([bar.get_height() for bar in fig.axes[0].patches], [2/3, 0, 1/3, 1])
        labels = [text.get_text() for text in fig.axes[0].texts]
        self.assertTrue(labels[0].startswith("2/3"))
        self.assertTrue(labels[3].startswith("1/1"))
        # 空组不能伪装成 0% 成绩。
        result["vocab"] = build_vocab(["好差不陌"])
        fig = plot_validation_groups(result)
        self.assertIn("无样本", [text.get_text() for text in fig.axes[0].texts])

    def test_embedding_heatmap_is_actual_lookup_with_repeated_character(self):
        embedding = torch.nn.Embedding(len(self.vocab), 4, padding_idx=0)
        with torch.no_grad():
            embedding.weight.copy_(torch.arange(len(self.vocab)*4).reshape(-1, 4))
        fig = plot_embedding_vectors(embedding, self.vocab, "好不好")
        shown = np.asarray(fig.axes[0].images[0].get_array())
        expected = embedding.weight.detach().numpy()[[self.vocab["好"], self.vocab["不"], self.vocab["好"]]]
        np.testing.assert_array_equal(shown, expected)
        np.testing.assert_array_equal(shown[0], shown[2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
