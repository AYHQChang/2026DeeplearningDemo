"""三级学生入口的功能边界；不包含LSTM/GRU分类器的作业答案。"""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import unittest
from unittest.mock import patch

import torch
from torch import nn

from rnn_lab.text_course import make_tuned_config, run_course_experiment, check_student_model
from rnn_lab.text_engine import TextExperimentConfig, export_text_experiments_csv
from rnn_lab.text_model import TextClassifier
from test_rnn_text import fixture_data


class BrokenProbe(nn.Module):
    """故意有问题的接口夹具，不是可训练分类器。"""
    def __init__(self, cell_type="lstm", mode="width"):
        super().__init__()
        self.cell_type, self.readout, self.mode = cell_type, "last", mode
        self.embedding = nn.Embedding(8, 3, padding_idx=0)
        self.recurrent = (nn.LSTM if cell_type == "lstm" else nn.GRU)(3, 4, batch_first=True)
        self.classifier = nn.Linear(4, 2)

    def forward(self, token_ids, lengths):
        if self.mode == "todo":
            raise NotImplementedError("学生前向尚未补齐")
        shape = (len(token_ids), 3 if self.mode == "shape" else 2)
        value = float("nan") if self.mode == "nan" else float(token_ids.shape[1])
        return torch.full(shape, value, device=token_ids.device)


class CourseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_level1_and_one_parameter_records(self):
        config = TextExperimentConfig(name="一级基线", cell_type="rnn", epochs=1,
                                      embedding_dim=4, hidden_size=5, batch_size=8, max_length=3)
        tuned = make_tuned_config(config, "learning_rate", .0015)
        changed = [key for key in config.__dataclass_fields__ if key != "name" and getattr(config,key)!=getattr(tuned,key)]
        self.assertEqual(changed, ["learning_rate"])
        runs = [run_course_experiment(c, fixture_data(), level=1) for c in (config,tuned)]
        for result in runs:
            self.assertIsInstance(result["model"].recurrent, nn.RNN)
            self.assertIsNone(result["metrics"])
            self.assertEqual(result["course_level"],1)
        with TemporaryDirectory() as tmp:
            path = export_text_experiments_csv(runs,Path(tmp)/"runs.csv")
            with path.open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows),2)
            self.assertTrue(all(row["test_accuracy"]=="" and row["implementation"]=="level1_provided_rnn" for row in rows))
        for key,value in [("cell_type","gru"),("epochs",1),("batch_size",2.5),("hidden_size",-1)]:
            with self.assertRaises(ValueError):make_tuned_config(config,key,value)

    def test_missing_models_cannot_fall_back_to_teacher(self):
        data = fixture_data()
        with patch("rnn_lab.text_course.run_text_experiment",side_effect=AssertionError("不得回退")) as train:
            for level,cell in [(2,"lstm"),(3,"gru")]:
                config = TextExperimentConfig(cell_type=cell)
                with self.assertRaises(NotImplementedError):run_course_experiment(config,data,level=level)
                reference = TextClassifier(8,2,cell_type=cell)
                with self.assertRaises(TypeError):run_course_experiment(config,data,level=level,model=reference)
            with self.assertRaises(ValueError):run_course_experiment(TextExperimentConfig(cell_type="gru"),data,level=1)
            train.assert_not_called()

    def test_bad_logits_padding_and_todo_are_explicit(self):
        for mode in ("shape","nan","width"):
            model = BrokenProbe(mode=mode)
            model.train();model.embedding.eval()
            modes = [module.training for module in model.modules()]
            for parameter in model.parameters():parameter.grad=torch.ones_like(parameter)
            with self.assertRaises(ValueError):check_student_model(model,2,"lstm")
            self.assertEqual(modes,[module.training for module in model.modules()])
            self.assertTrue(all(torch.equal(p.grad,torch.ones_like(p)) for p in model.parameters()))
        with self.assertRaises(NotImplementedError):check_student_model(BrokenProbe(mode="todo"),2,"lstm")
        with self.assertRaises(TypeError):check_student_model(BrokenProbe("gru"),2,"lstm")


if __name__=="__main__":
    unittest.main(verbosity=2)
