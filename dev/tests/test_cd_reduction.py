"""The cached CD reduction operator algebra (cd_reduction)."""
def test_the_cached_operator_algebra_is_read_only_and_unchanged():
    """The J-only operators are cached (they were rebuilt for J = 0..6 on
    every call, 356k builds in one CD config run); a caller must not be
    able to corrupt the cache, and the levels must equal the uncached
    construction bit for bit."""
    import numpy as np
    import pytest
    from backend.spectral import cd_reduction as R
    ops = R._reduced_operators("A", 3)
    with pytest.raises(ValueError):
        ops[0][0, 0] = 1.0
    p2, pz2, pd = R.angular_momentum_operators(2)
    p2[0, 0] = -1.0                       # the public function's arrays are the caller's own
    np.testing.assert_array_equal(R.angular_momentum_operators(2)[0], 6.0 * np.eye(5))
    for cached, fresh in zip(R._squares(2), R._axis_squares(*R.angular_momentum_operators(2))):
        np.testing.assert_array_equal(cached, fresh)
    abc = (835840.3, 435351.7, 278138.7)
    tau = np.array([[-0.1, 0.02, 0.0], [0.02, -0.03, 0.01], [0.0, 0.01, -0.02]])
    ref = []
    for j in range(7):
        sq = R._axis_squares(*R.angular_momentum_operators(j))
        h = abc[0] * sq[0] + abc[1] * sq[1] + abc[2] * sq[2]
        for al in range(3):
            for be in range(3):
                prod = sq[al] @ sq[be]
                h += 0.25 * tau[al, be] * 0.5 * (prod + prod.T)
        ref.append(np.sort(np.linalg.eigvalsh(0.5 * (h + h.T))))
    np.testing.assert_array_equal(R.levels_from_tau(abc, tau), np.concatenate(ref))
