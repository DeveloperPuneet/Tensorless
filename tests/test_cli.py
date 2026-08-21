import os

from tensorless.cli.main import main


def test_cli_train(text_corpus, workdir):
    rc = main(["train", text_corpus, "--out", "model.tl", "--epochs", "1", "--d-model", "16", "--layers", "1", "--heads", "2", "--batch-size", "16", "--quiet"])
    assert rc == 0
    assert os.path.isfile("model.tl")


def test_cli_inspect(text_corpus, workdir, capsys):
    rc = main(["inspect", text_corpus])
    assert rc == 0
    captured = capsys.readouterr()
    assert "text-generation" in captured.out


def test_cli_info(text_corpus, workdir, capsys):
    main(["train", text_corpus, "--out", "model.tl", "--epochs", "1", "--d-model", "16", "--layers", "1", "--heads", "2", "--batch-size", "16", "--quiet"])
    rc = main(["info", "model.tl"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "text-generation" in captured.out


def test_cli_run_with_prompt(text_corpus, workdir, capsys):
    main(["train", text_corpus, "--out", "model.tl", "--epochs", "1", "--d-model", "16", "--layers", "1", "--heads", "2", "--batch-size", "16", "--quiet"])
    rc = main(["run", "model.tl", "--prompt", "the quick"])
    assert rc == 0
    captured = capsys.readouterr()
    assert len(captured.out.strip()) > 0


def test_cli_error_returns_nonzero(workdir):
    rc = main(["inspect", "does_not_exist_path"])
    assert rc == 1
