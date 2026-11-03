import pytest
import torch

from model.classifier import ModelConfig, SequenceClassifier


@pytest.mark.parametrize("attention", ["discrete", "continuous"])
@pytest.mark.parametrize("pooling", ["mean", "cls"])
def test_forward_shapes_any_length(attention, pooling):
    torch.manual_seed(0)
    m = SequenceClassifier(ModelConfig(vocab_size=16, n_classes=5, n_embd=32, n_head=4,
                                       attention=attention, pooling=pooling, n_basis=8))
    for K in (16, 64, 200):
        assert m(torch.randint(0, 16, (3, K))).shape == (3, 5)


@pytest.mark.parametrize("attention", ["discrete", "continuous"])
def test_bidirectional(attention):
    """Changing the last token must change the first position's representation (no causal mask)."""
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=16, n_embd=32, n_head=4, attention=attention, n_basis=8, init_var=0.5)
    m = SequenceClassifier(cfg).double()
    x = torch.randint(0, 16, (1, 32))
    x2 = x.clone(); x2[0, -1] = (x[0, -1] + 1) % 16
    emb = lambda x: m.tok_emb(x) + m.pos(torch.arange(32, dtype=torch.float64) / 32)[None]
    a, b = m.blocks[0].attn(emb(x)), m.blocks[0].attn(emb(x2))
    assert (a[0, 0] - b[0, 0]).abs().max() > 1e-8


def test_both_models_share_embedding_scheme():
    d = ModelConfig(attention="discrete"); c = ModelConfig(attention="continuous")
    md, mc = SequenceClassifier(d), SequenceClassifier(c)
    assert type(md.pos) is type(mc.pos) and md.pos.freqs.equal(mc.pos.freqs)


def test_continuous_basis_shared_across_layers():
    m = SequenceClassifier(ModelConfig(attention="continuous", n_layer=3))
    assert len({id(b.attn.basis) for b in m.blocks}) == 1


def test_recorded_densities():
    m = SequenceClassifier(ModelConfig(attention="continuous", n_layer=2, n_head=4))
    m.set_record(True)
    m(torch.randint(0, 32, (2, 40)))
    dens = m.recorded_densities()
    assert len(dens) == 2 and dens[0][0].shape == (2, 4, 40)
    assert ((dens[0][0] > 0) & (dens[0][0] < 1)).all() and (dens[0][1] > 0).all()
