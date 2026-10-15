import pytest
import torch
from astrograph.evaluation.reference_backbones import make_reference_model

@pytest.mark.parametrize('family',['residual_tcn','inception_time','timesnet_ref','moderntcn_ref','patchtst_ref'])
def test_reference_three_heads_forward_backward(family):
    torch.set_num_threads(1);torch.manual_seed(4)
    m=make_reference_model(family,7,5,14,t=20)
    for x in (torch.randn(4,20,7),torch.zeros(4,20,7)):
        y=m(x);assert [v.shape for v in y]==[(4,2),(4,5),(4,14)]
        assert all(torch.isfinite(v).all() for v in y)
        sum(v.square().mean() for v in y).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())

def test_modern_separates_variable_and_feature_mixing():
    m=make_reference_model('moderntcn_ref',7,5,14)
    assert m.encoder.blocks[0].features[0].groups==7
    assert m.encoder.blocks[0].variables[0].groups==16

def test_patch_encoder_weights_shared_across_channels():
    m=make_reference_model('patchtst_ref',7,5,14)
    assert m.encoder.patch.in_features==4 and m.encoder.width==7*10*32

def test_unknown_model_not_silently_substituted():
    with pytest.raises(ValueError):make_reference_model('Mantis',7,5,14)
