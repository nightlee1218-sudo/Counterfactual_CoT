import torch


def test_torch_checkpoint_reload_preserves_model_and_optimizer_state(tmp_path):
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(4, 3)
    target = torch.randn(4, 2)

    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    optimizer.step()
    expected = model(x).detach().clone()

    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)

    reloaded = torch.nn.Linear(3, 2)
    reloaded_optimizer = torch.optim.AdamW(reloaded.parameters(), lr=1e-3)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    reloaded.load_state_dict(payload["model"])
    reloaded_optimizer.load_state_dict(payload["optimizer"])

    actual = reloaded(x).detach()
    assert torch.equal(actual, expected)
    assert reloaded_optimizer.state_dict()["state"]
