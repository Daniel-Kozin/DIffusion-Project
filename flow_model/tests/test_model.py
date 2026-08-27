import torch

from flow_model.velocity_model import FlowMatchingUNet3D


def test_forward_shape():
    model = FlowMatchingUNet3D(num_classes=4, base_ch=4, embed_channels=8)
    x = torch.randn(2, 4, 26, 128, 128)
    t = torch.rand(2)
    out = model(x, t)
    assert out.shape == x.shape


def test_grad_checkpoint_matches_no_checkpoint():
    # dropout=0 so the only source of difference between the two forward passes
    # can be the checkpointing mechanism itself, not independent dropout draws
    torch.manual_seed(0)
    model_ckpt = FlowMatchingUNet3D(num_classes=4, base_ch=4, embed_channels=8,
                                     dropout=0.0, use_grad_checkpoint=True)
    torch.manual_seed(0)
    model_no_ckpt = FlowMatchingUNet3D(num_classes=4, base_ch=4, embed_channels=8,
                                        dropout=0.0, use_grad_checkpoint=False)
    model_no_ckpt.load_state_dict(model_ckpt.state_dict())

    model_ckpt.train()
    model_no_ckpt.train()

    x = torch.randn(1, 4, 26, 128, 128)
    t = torch.rand(1)

    out_ckpt = model_ckpt(x, t)
    out_no_ckpt = model_no_ckpt(x, t)

    assert torch.allclose(out_ckpt, out_no_ckpt, atol=1e-5)
