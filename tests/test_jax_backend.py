import numpy as np
import pytest

jax = pytest.importorskip("jax")

from tensorless.backends.jax_backend import JaxTinyTransformer


@pytest.mark.parametrize("task", ["text-generation", "text-classification"])
def test_jax_transformer_forward_backward_and_state_round_trip(task):
    classes = 3 if task == "text-classification" else 0
    model = JaxTinyTransformer(
        vocab_size=12,
        d_model=4,
        layers=1,
        heads=2,
        ff_mult=2,
        dropout=0.0,
        max_seq_len=6,
        task=task,
        n_classes=classes,
        pad_id=0,
    )
    input_ids = np.array([[1, 2, 3, 0], [2, 3, 4, 5]], dtype=np.int64)
    if task == "text-generation":
        targets = np.array([[2, 3, 4, 0], [3, 4, 5, 6]], dtype=np.int64)
        logits = model.forward(input_ids)
        loss = model.loss_and_backward(input_ids, targets)
    else:
        targets = np.array([1, 2], dtype=np.int64)
        mask = (input_ids != 0).astype(np.float32)
        logits = model.forward(input_ids, mask)
        loss = model.loss_and_backward(input_ids, targets, mask)

    assert logits.shape == ((2, 4, 12) if task == "text-generation" else (2, 3))
    assert np.isfinite(loss)
    assert any(np.any(parameter.grad != 0) for parameter in model.parameters())

    state = model.state_dict()
    restored = JaxTinyTransformer(12, 4, 1, 2, 2, 0.0, 6, task=task, n_classes=classes)
    restored.load_state_dict(state)
    np.testing.assert_allclose(restored.forward(input_ids, None if task == "text-generation" else mask), logits)