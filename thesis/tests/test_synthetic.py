import numpy as np
import pytest
import torch

from data.synthetic import (SyntheticTask, TaskConfig, binned_features, calibrate_signal_scale,
                            run_baselines, time_grid, unigram_features)


@pytest.fixture(scope="module")
def task():
    return SyntheticTask(TaskConfig(n_classes=4, vocab_size=8, concentration=1.0, seed=3))


def test_probs_are_distributions_and_f_in_unit_interval(task):
    P = task.probs(50)
    assert P.shape == (4, 50, 8)
    torch.testing.assert_close(P.sum(-1), torch.ones(4, 50, dtype=torch.float64))
    f = task.f(np.linspace(0, 1, 1001))
    assert f.min() >= 0 and f.max() <= 1


def test_same_classes_for_any_length(task):
    """Classes are defined on t in [0,1]: P at t = i/K must coincide across K."""
    P64, P128 = task.probs(64), task.probs(128)
    torch.testing.assert_close(P64, P128[:, ::2])


def test_empirical_frequencies_match_generator(task):
    K, n = 16, 40000
    g = torch.Generator().manual_seed(0)
    labels = torch.full((n,), 2, dtype=torch.long)
    x, _ = task.sample(n, K, generator=g, labels=labels)
    emp = torch.zeros(K, 8, dtype=torch.float64)
    emp.scatter_add_(1, x.T, torch.ones(K, n, dtype=torch.float64))
    emp /= n
    P = task.probs(K)[2]
    se = (P * (1 - P) / n).sqrt()
    assert ((emp - P).abs() <= 5 * se + 1e-4).all()


def test_bayes_posterior_is_normalised(task):
    x, _ = task.sample(32, 20, generator=torch.Generator().manual_seed(0))
    lp = task.bayes_log_posterior(x)
    torch.testing.assert_close(lp.exp().sum(-1), torch.ones(32, dtype=torch.float64))


def test_position_only_shares_Q_and_unigram_is_chance():
    t = SyntheticTask(TaskConfig(n_classes=6, vocab_size=16, concentration=0.5, position_only=True, seed=1))
    assert np.allclose(t.Q1, t.Q1[0]) and np.allclose(t.Q2, t.Q2[0])
    # match_mean makes the expected unigram distribution (almost) identical across classes
    fbar = t.f(time_grid(4096)).mean(1)
    assert np.abs(fbar - 0.5).max() < 1e-2
    res = run_baselines(t, K=64, n_train=6000, n_test=3000)
    chance = 1 / 6
    assert abs(res["unigram"] - chance) < 0.05, res
    assert res["binned"] > chance + 0.2, res


def test_bayes_is_never_beaten_by_baselines():
    for po in (False, True):
        t = SyntheticTask(TaskConfig(n_classes=8, vocab_size=16, concentration=3.0, position_only=po, seed=0))
        res = run_baselines(t, K=32, n_train=8000, n_test=4000)
        se = np.sqrt(res["bayes"] * (1 - res["bayes"]) / 4000)
        for name in ("unigram", "binned"):
            assert res[name] <= res["bayes"] + 3 * se, (po, res)


def test_signal_scale_calibration_hits_target():
    cfg = TaskConfig(n_classes=8, vocab_size=16, concentration=3.0, seed=0)
    s = calibrate_signal_scale(cfg, K=256, target_acc=0.9, n=4000, tol=0.01)
    assert 0 < s < 1
    acc = SyntheticTask(TaskConfig(**{**cfg.__dict__, "signal_scale": s})).bayes_accuracy(256, 4000, seed=4321)
    assert abs(acc - 0.9) < 0.02


def test_feature_shapes():
    x = torch.randint(0, 8, (5, 30))
    assert unigram_features(x, 8).shape == (5, 8)
    np.testing.assert_allclose(unigram_features(x, 8).sum(1), 1.0)
    b = binned_features(x, 8, n_bins=4)
    assert b.shape == (5, 32)
    np.testing.assert_allclose(b.reshape(5, 4, 8).sum(-1), 1.0)
