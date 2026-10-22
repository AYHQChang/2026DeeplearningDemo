"""三级改进的行为回归；使用教师参考作为夹具，不提供学生分类器答案。"""
from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch
import unittest
import torch
from torch import nn
from rnn_lab.text_course import check_student_model, make_tuned_config, check_design_pair, course_score
from rnn_lab.text_engine import TextExperimentConfig, run_text_experiment, evaluate_text_model, evaluate_text_final, text_experiment_records
from rnn_lab.text_model import TextClassifier
from test_rnn_text import fixture_data


def reference_probe(cell="gru", readout="last"):
    reference = TextClassifier(8, 2, 3, 4, cell_type=cell, readout=readout)
    probe = nn.Module()
    # 复用既有教师模型的层和前向方法，仅用于检查器回归。
    for name, module in reference.named_children():
        probe.add_module(name, module)
    probe.cell_type, probe.readout = cell, readout
    probe.forward = reference.forward
    return probe


class UpgradeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_semantics_and_gradients_preserve_state(self):
        for cell in ("gru", "lstm"):
            for readout in ("last", "mean"):
                model = reference_probe(cell, readout)
                model.train(); model.embedding.eval()
                for p in model.parameters():
                    p.grad = torch.ones_like(p)
                state, modes = deepcopy(model.state_dict()), [m.training for m in model.modules()]
                rng = torch.random.get_rng_state().clone()
                result = check_student_model(model, 2, cell)
                self.assertTrue(result["readout_semantics"] and result["gradient_connected"])
                self.assertEqual(modes, [m.training for m in model.modules()])
                self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, state[name])
                self.assertTrue(all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters()))
                self.assertEqual(len(model.recurrent._forward_hooks), 0)
                self.assertEqual(len(model.classifier._forward_pre_hooks), 0)

    def test_wrong_first_state_is_rejected(self):
        model = reference_probe()
        # 捕获实际第一个状态并在分类头前替代，使PAD不变性仍成立。
        captured = []
        a = model.recurrent.register_forward_hook(lambda m, a, o: captured.append(o[0][:, 0]))
        b = model.classifier.register_forward_pre_hook(lambda m, a: (captured[-1],))
        try:
            with self.assertRaisesRegex(ValueError, "读出语义"):
                check_student_model(model, 2, "gru")
        finally:
            a.remove(); b.remove()

    def test_detached_graph_and_softmax_rejected(self):
        model = reference_probe()
        h = model.recurrent.register_forward_hook(lambda m, a, o: (o[0].detach(), o[1]))
        try:
            with self.assertRaisesRegex(ValueError, "反向计算"):
                check_student_model(model, 2, "gru")
        finally:
            h.remove()
        model = reference_probe();model.probabilities = nn.Softmax(dim=1)
        with self.assertRaises(ValueError): check_student_model(model, 2, "gru")

    def test_best_checkpoint_is_restored_and_ties_are_deterministic(self):
        config = TextExperimentConfig(cell_type="rnn", embedding_dim=3, hidden_size=4,
            epochs=4, batch_size=8, selection="best_val_accuracy")
        observed = []
        def metrics(model, loader):
            result = evaluate_text_model(model, loader)
            i = len(observed)
            result.update(accuracy=[.5, .8, .8, .6][i], loss=[.8, .4, .4, .3][i])
            observed.append(deepcopy(model.state_dict()))
            return result
        with patch("rnn_lab.text_engine.evaluate_text_model", side_effect=metrics):
            run = run_text_experiment(config, fixture_data())
        self.assertEqual(run["selected_epoch"], 2)
        for name, value in run["model"].state_dict().items():
            torch.testing.assert_close(value, observed[1][name])
        self.assertEqual(run["val_metrics"]["accuracy"], .8)
        self.assertEqual(len(run["history"]["val_accuracy"]), 4)
        self.assertEqual(run["optimizer_steps"], 4)
        self.assertIsNone(run["metrics"])

    def test_optimizers_records_and_score(self):
        config = TextExperimentConfig(cell_type="rnn", embedding_dim=3, hidden_size=4,
            epochs=1, batch_size=8, selection="best_val_accuracy")
        for optimizer in ("adam", "sgd", "adamw"):
            run = run_text_experiment(replace(config, optimizer=optimizer), fixture_data())
            run["course_level"] = 2
            self.assertIsNone(text_experiment_records([run])[0]["provisional_score"])
            with self.assertRaises(ValueError): course_score(run)
            final = evaluate_text_final(run)
            for level, factor in ((1,.8),(2,.9),(3,1.0)):
                run["course_level"] = level
                self.assertAlmostEqual(course_score(run)["provisional_score"], 100*final["accuracy"]*factor)
                row=text_experiment_records([run])[0]
                self.assertEqual(row["optimizer"],optimizer)
                self.assertAlmostEqual(row["provisional_score"],100*final["accuracy"]*factor)
            with torch.no_grad(): next(run["model"].parameters()).add_(.1)
            with self.assertRaises(RuntimeError): course_score(run)

    def test_controlled_design_and_tuning(self):
        c=TextExperimentConfig(cell_type="gru")
        self.assertEqual(check_design_pair(c,replace(c,readout="mean")),"readout")
        for variant in (c,replace(c,readout="mean",learning_rate=.001),replace(c,cell_type="lstm")):
            with self.assertRaises(ValueError):check_design_pair(c,variant)
        c=replace(c,cell_type="rnn")
        self.assertEqual(make_tuned_config(c,"optimizer","sgd").optimizer,"sgd")
        self.assertIsNone(make_tuned_config(c,"gradient_clip",None).gradient_clip)
        for parameter,value in (("learning_rate",float('nan')),("dropout",.2),("epochs",2.5)):
            with self.assertRaises(ValueError): make_tuned_config(c,parameter,value)


if __name__ == '__main__':
    unittest.main(verbosity=2)
