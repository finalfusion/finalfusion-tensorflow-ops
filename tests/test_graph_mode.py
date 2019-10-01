import numpy as np
import pytest
import tensorflow as tf
import finalfusion_tensorflow as ff_tf


def test_graph_lookup():
    embeddings = ff_tf.ff_embeddings()
    init = ff_tf.initialize_ff_embeddings(embeddings, "testdata/test.fifu", False)

    ber = ff_tf.ff_lookup(embeddings, "Berlin", mask_empty_string=False, mask_failed_lookup=False, embedding_len=100)
    assert ber.shape == (100,)

    ber_list = ff_tf.ff_lookup(embeddings, ["Berlin"], mask_empty_string=False, mask_failed_lookup=False,
                               embedding_len=100)
    assert ber_list.shape == (1, 100)

    ber_tensor = ff_tf.ff_lookup(embeddings, [["Berlin"]], mask_empty_string=False, mask_failed_lookup=False,
                                 embedding_len=100)
    assert ber_tensor.shape == (1, 1, 100)

    ber_no_shape = ff_tf.ff_lookup(embeddings, "Berlin", mask_empty_string=False, mask_failed_lookup=False)
    assert ber_no_shape.shape.rank == 1
    assert ber_no_shape.shape[0].value is None

    ber_list_no_shape = ff_tf.ff_lookup(embeddings, ["Berlin"], mask_empty_string=False, mask_failed_lookup=False)
    assert ber_list_no_shape.shape.rank == 2
    assert ber_list_no_shape.shape[0].value == tf.Dimension(1)
    assert ber_list_no_shape.shape[1].value is None

    with tf.Session() as sess:
        sess.run([init])
        res = sess.run([ber, ber_list, ber_tensor])
        assert res[0].shape == (100,)
        assert res[1].shape == (1, 100)
        assert res[2].shape == (1, 1, 100)
        sess.run([ff_tf.close_ff_embeddings(embeddings)])


def test_graph_lookup_masked():
    embeddings = ff_tf.ff_embeddings()
    init = ff_tf.initialize_ff_embeddings(embeddings, "testdata/test.fifu", True)
    tuebingen_masked = ff_tf.ff_lookup(embeddings, "Tübingen", mask_empty_string=False, mask_failed_lookup=True,
                                       embedding_len=100)
    empty_masked = ff_tf.ff_lookup(embeddings, "", mask_empty_string=True, mask_failed_lookup=False, embedding_len=100)
    empty_masked_through_fail = ff_tf.ff_lookup(embeddings, "", mask_empty_string=False, mask_failed_lookup=True,
                                                embedding_len=100)
    with tf.Session() as sess:
        sess.run([init])
        res = sess.run([tuebingen_masked, empty_masked, empty_masked_through_fail])
        assert np.allclose(res, 0.)


def test_graph_errors():
    embeddings = ff_tf.ff_embeddings()
    tuebingen_unmasked = ff_tf.ff_lookup(embeddings, "Tübingen", mask_empty_string=False, mask_failed_lookup=False,
                                         embedding_len=100)
    ber_bad_shape = ff_tf.ff_lookup(embeddings, "Berlin", mask_empty_string=False, mask_failed_lookup=False,
                                    embedding_len=10)
    assert ber_bad_shape.shape == (10,)
    empty_unmasked = ff_tf.ff_lookup(embeddings, "", mask_empty_string=False, mask_failed_lookup=False,
                                     embedding_len=100)

    with tf.Session() as sess:
        with pytest.raises(tf.errors.UnknownError):
            sess.run([ff_tf.initialize_ff_embeddings(embeddings, "foo.fifu", False)])

        sess.run([ff_tf.initialize_ff_embeddings(embeddings, "testdata/test.fifu", False)])

        with pytest.raises(tf.errors.AlreadyExistsError):
            sess.run([ff_tf.initialize_ff_embeddings(embeddings, "testdata/test.fifu", False)])
        with pytest.raises(tf.errors.InvalidArgumentError):
            sess.run([tuebingen_unmasked])
        with pytest.raises(tf.errors.InvalidArgumentError):
            sess.run([empty_unmasked])
        with pytest.raises(tf.errors.InvalidArgumentError):
            sess.run([ber_bad_shape])
        sess.run([ff_tf.close_ff_embeddings(embeddings)])
        with pytest.raises(tf.errors.NotFoundError):
            sess.run([ff_tf.close_ff_embeddings(embeddings)])
