import platform
import pytest


def test_init_and_close(ff_lib):
    embeddings = ff_lib.ff_embeddings()
    ff_lib.initialize_ff_embeddings(embeddings, "tests/data/test.fifu", mmap=False)
    ff_lib.close_ff_embeddings(embeddings)


def test_init_and_close_mmap(ff_lib):
    embeddings = ff_lib.ff_embeddings()
    ff_lib.initialize_ff_embeddings(embeddings, "tests/data/test.fifu", mmap=True)
    ff_lib.close_ff_embeddings(embeddings)
