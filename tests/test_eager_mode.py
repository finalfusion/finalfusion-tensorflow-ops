import numpy as np
import pytest
import tensorflow as tf

tf.enable_eager_execution()


def test_init_and_close(ff_lib):
    embeddings = ff_lib.ff_embeddings()
    ff_lib.initialize_ff_embeddings(embeddings, "tests/data/test.fifu", mmap=False)
    ff_lib.close_ff_embeddings(embeddings)


def test_init_and_close_mmap(ff_lib):
    embeddings = ff_lib.ff_embeddings()
    ff_lib.initialize_ff_embeddings(embeddings, "tests/data/test.fifu", mmap=True)
    ff_lib.close_ff_embeddings(embeddings)


def test_eager_lookup(ff_lib):
    embeddings = ff_lib.ff_embeddings()
    ff_lib.initialize_ff_embeddings(embeddings, "tests/data/test.fifu", mmap=False)

    ber = ff_lib.ff_lookup(embeddings, "Berlin", mask_empty_string=False, mask_failed_lookup=False)
    ber_list = ff_lib.ff_lookup(embeddings, ["Berlin"], mask_empty_string=False, mask_failed_lookup=False)
    ber_tensor = ff_lib.ff_lookup(embeddings, [["Berlin"]], mask_empty_string=False, mask_failed_lookup=False)

    assert ber.shape == (100,)
    assert ber_list.shape == (1, 100)
    assert ber_tensor.shape == (1, 1, 100)

    ff_lib.close_ff_embeddings(embeddings)


def test_eager_lookup_masked(ff_lib):
    embeddings = ff_lib.ff_embeddings()
    ff_lib.initialize_ff_embeddings(embeddings, "tests/data/test.fifu", False)
    tuebingen_masked = ff_lib.ff_lookup(embeddings, "Tübingen", mask_empty_string=False, mask_failed_lookup=True,
                                        embedding_len=100)
    empty_masked = ff_lib.ff_lookup(embeddings, "", mask_empty_string=True, mask_failed_lookup=False, embedding_len=100)
    empty_masked_through_fail = ff_lib.ff_lookup(embeddings, "", mask_empty_string=False, mask_failed_lookup=True,
                                                 embedding_len=100)
    assert np.allclose(tuebingen_masked, 0.)
    assert np.allclose(empty_masked, 0.)
    assert np.allclose(empty_masked_through_fail, 0.)
    ff_lib.close_ff_embeddings(embeddings)


def test_eager_errors(ff_lib):
    embeddings = ff_lib.ff_embeddings()
    with pytest.raises(tf.errors.UnknownError):
        ff_lib.initialize_ff_embeddings(embeddings, "foo.fifu", False)

    ff_lib.initialize_ff_embeddings(embeddings, "tests/data/test.fifu", False)

    with pytest.raises(tf.errors.AlreadyExistsError):
        ff_lib.initialize_ff_embeddings(embeddings, "tests/data/test.fifu", False)

    with pytest.raises(tf.errors.InvalidArgumentError):
        ff_lib.ff_lookup(embeddings, "Tübingen", mask_empty_string=False, mask_failed_lookup=False, embedding_len=100)

    # shape mismatch, 10 vs. actual 100
    with pytest.raises(tf.errors.InvalidArgumentError):
        ff_lib.ff_lookup(embeddings, "Berlin", mask_empty_string=False, mask_failed_lookup=False, embedding_len=10)

    with pytest.raises(tf.errors.InvalidArgumentError):
        ff_lib.ff_lookup(embeddings, "", mask_empty_string=False, mask_failed_lookup=False, embedding_len=100)

    ff_lib.close_ff_embeddings(embeddings)
    with pytest.raises(tf.errors.NotFoundError):
        ff_lib.close_ff_embeddings(embeddings)
