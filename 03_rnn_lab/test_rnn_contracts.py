"""应用任务共用契约回归；不包含任何手写循环/作业转换答案。"""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import nn

from rnn_lab import (ExperimentConfig, evaluate_final, export_experiments_csv,
                     plot_comparison, print_result_table, run_experiment)
from rnn_lab.manual_rnn import StudentRNN, copy_rnn_weights, verify_manual_rnn


def expect(error, callback):
    try:
        callback()
    except error:
        return
    raise AssertionError(f'应抛出 {error.__name__}')


def test_student_boundary():
    for f, h in ((2, 3), (8, 5)):
        core = StudentRNN(f, h)
        reference = nn.RNN(f, h, batch_first=True)
        copy_rnn_weights(reference, core)
        copied = nn.RNN(f, h, batch_first=True)
        copy_rnn_weights(core, copied)
        for name, p in core.named_parameters():
            torch.testing.assert_close(p, dict(reference.named_parameters())[name])
            torch.testing.assert_close(p, dict(copied.named_parameters())[name])
        expect(NotImplementedError, lambda: core(torch.randn(2, 4, f)))
        expect(NotImplementedError, lambda: core.step(torch.randn(2, f), torch.randn(2, h)))
        expect(NotImplementedError, lambda: verify_manual_rnn(core))
        expect(ValueError, lambda: verify_manual_rnn(reference))
    expect(ValueError, lambda: copy_rnn_weights(nn.RNN(2, 3), StudentRNN(4, 3)))
    expect(ValueError, lambda: StudentRNN(0, 3))


def test_final_cache_and_records():
    torch.set_num_threads(1)
    config = ExperimentConfig(epochs=1, sequence_length=5, train_size=24, val_size=12,
                              test_size=12, hidden_size=4, batch_size=12)
    result = run_experiment(config, evaluate_test=False)
    result.update(task='image_classification', data_version='contract_fixture',
                  data_split='original_train/val/test', representation='patches', patch_size=2, readout='last')
    with TemporaryDirectory() as tmp:
        path = export_experiments_csv([result], Path(tmp) / 'records.csv')
        row = next(csv.DictReader(path.open(encoding='utf-8-sig')))
        assert row['test_evaluated'] == 'False' and row['test_accuracy'] == ''
        assert row['task'] == 'image_classification' and row['patch_size'] == '2'
        assert row['data_version'] == 'contract_fixture'
        first = evaluate_final(result)
        assert evaluate_final(result) is first
        with torch.no_grad():
            next(result['model'].parameters()).add_(0.01)
        expect(RuntimeError, lambda: evaluate_final(result))
        expect(RuntimeError, lambda: export_experiments_csv([result], path))
        expect(RuntimeError, lambda: plot_comparison([result], split='test'))
        expect(RuntimeError, lambda: print_result_table([result]))
    other = run_experiment(config, evaluate_test=False)
    evaluate_final(other)
    other['model'].readout = 'mean'
    expect(RuntimeError, lambda: evaluate_final(other))


if __name__ == '__main__':
    test_student_boundary()
    test_final_cache_and_records()
    print('RNN shared contracts passed: TODO boundaries, weight copy, final-cache fingerprint, CSV metadata.')
