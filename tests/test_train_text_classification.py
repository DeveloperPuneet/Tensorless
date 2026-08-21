import tensorless as tl
from tensorless.data.loader import load_dataset


def test_train_text_classification(text_classification_dir, workdir):
    model = tl.train(
        text_classification_dir,
        out="sentiment.tl",
        epochs=10,
        d_model=32,
        layers=2,
        heads=2,
        batch_size=16,
        max_seq_len=64,
    )
    assert model.task == "text-classification"
    assert set(model.meta["classes"]) == {"positive", "negative"}


def test_text_classification_accuracy_on_train_set(text_classification_dir, workdir):
    model = tl.train(
        text_classification_dir,
        out="sentiment.tl",
        epochs=15,
        d_model=32,
        layers=2,
        heads=2,
        batch_size=16,
        max_seq_len=64,
    )
    ds = load_dataset(text_classification_dir)
    correct = sum(1 for t, l in zip(ds.texts, ds.labels) if model.predict(t) == l)
    accuracy = correct / len(ds.texts)
    assert accuracy > 0.9
