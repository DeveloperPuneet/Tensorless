from tensorless.data.fingerprint import fingerprint_path


def test_fingerprint_stable(text_corpus):
    fp1 = fingerprint_path(text_corpus)
    fp2 = fingerprint_path(text_corpus)
    assert fp1 == fp2


def test_fingerprint_changes_on_content_change(workdir):
    path = workdir / "a.txt"
    path.write_text("hello world")
    fp1 = fingerprint_path(str(path))
    path.write_text("hello world!!!")
    fp2 = fingerprint_path(str(path))
    assert fp1 != fp2


def test_fingerprint_changes_on_new_file(workdir):
    d = workdir / "data"
    d.mkdir()
    (d / "a.txt").write_text("hello")
    fp1 = fingerprint_path(str(d))
    (d / "b.txt").write_text("world")
    fp2 = fingerprint_path(str(d))
    assert fp1 != fp2


def test_fingerprint_unaffected_by_mtime(workdir):
    import os
    import time

    path = workdir / "a.txt"
    path.write_text("hello world")
    fp1 = fingerprint_path(str(path))
    # Touch the file (content unchanged) -- fingerprint must be stable.
    time.sleep(0.01)
    os.utime(path, None)
    fp2 = fingerprint_path(str(path))
    assert fp1 == fp2
