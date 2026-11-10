"""Tests for backend/spectral/hessian_cache.py.

The cache exists so an interrupted benchmark does not throw away an hour of
MP2 gradients. Its failure mode is not slowness, it is returning a Hessian
computed for something else -- a different level of theory most of all, since
an HF Hessian handed back for an MP2 request looks entirely plausible and is
simply a different force field. So most of these tests are about what must
MISS rather than what must hit.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.spectral.hessian_cache import HessianCache, cache_key

_ELEMS = ["C", "O", "H", "H"]


def _coords(shift=0.0):
    return np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0],
                     [-0.5, 0.9, 0.0], [-0.5, -0.9, shift]])


def _hessian(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(12, 12))
    return a + a.T


def _cache(tmp_path, method="mp2", basis="cc-pvtz", elements=None, **kw):
    return HessianCache(tmp_path, method=method, basis=basis,
                        elements=_ELEMS if elements is None else elements, **kw)


class TestRoundTrip:
    def test_stores_and_returns_the_same_matrix(self, tmp_path):
        c = _cache(tmp_path)
        h = _hessian()
        c.store(_coords(), h)
        np.testing.assert_allclose(c.load(_coords()), h, rtol=0.0, atol=0.0)

    def test_a_cold_cache_misses(self, tmp_path):
        assert _cache(tmp_path).load(_coords()) is None

    def test_wrap_computes_once_then_serves_from_disk(self, tmp_path):
        calls = []

        def compute(coords):
            calls.append(np.asarray(coords).copy())
            return _hessian()

        fn = _cache(tmp_path).wrap(compute)
        first = fn(_coords())
        second = fn(_coords())
        assert len(calls) == 1, "second call should have hit the cache"
        np.testing.assert_allclose(first, second, atol=0.0)

    def test_a_second_cache_object_sees_the_first_one_s_writes(self, tmp_path):
        # The point of the whole module: a NEW process finds the old work.
        h = _hessian(3)
        _cache(tmp_path).store(_coords(), h)
        np.testing.assert_allclose(_cache(tmp_path).load(_coords()), h, atol=0.0)

    def test_counters_report_hits_and_misses(self, tmp_path):
        c = _cache(tmp_path)
        fn = c.wrap(lambda _: _hessian())
        fn(_coords())
        fn(_coords())
        assert c.misses == 1 and c.hits == 1
        assert c.summary()["level"] == "mp2/cc-pvtz"


class TestWhatMustMiss:
    """Each of these would return a confidently wrong Hessian if keyed wrong."""

    def test_a_different_method_misses(self, tmp_path):
        _cache(tmp_path, method="hf").store(_coords(), _hessian())
        assert _cache(tmp_path, method="mp2").load(_coords()) is None

    def test_a_different_basis_misses(self, tmp_path):
        _cache(tmp_path, basis="6-31g").store(_coords(), _hessian())
        assert _cache(tmp_path, basis="cc-pvtz").load(_coords()) is None

    def test_different_elements_miss(self, tmp_path):
        # Same atom count, same geometry, different molecule.
        _cache(tmp_path, elements=["C", "O", "H", "H"]).store(_coords(), _hessian())
        assert _cache(tmp_path, elements=["C", "S", "H", "H"]).load(_coords()) is None

    def test_a_different_geometry_misses(self, tmp_path):
        c = _cache(tmp_path)
        c.store(_coords(0.0), _hessian())
        assert c.load(_coords(0.05)) is None

    def test_a_geometry_difference_below_the_rounding_hits(self, tmp_path):
        # Identity is defined at 1e-9 Ang, matching the in-memory cache the
        # benchmark already used, so the two agree on what the same point is.
        c = _cache(tmp_path)
        h = _hessian()
        c.store(_coords(0.0), h)
        np.testing.assert_allclose(c.load(_coords(1e-12)), h, atol=0.0)

    def test_case_and_whitespace_in_the_level_do_not_split_the_cache(self, tmp_path):
        h = _hessian()
        _cache(tmp_path, method="MP2", basis=" CC-pVTZ ").store(_coords(), h)
        np.testing.assert_allclose(
            _cache(tmp_path, method="mp2", basis="cc-pvtz").load(_coords()),
            h, atol=0.0)


class TestCorruptionIsAMiss:
    """A bad file costs one Hessian; trusting it could cost a conclusion."""

    def test_a_truncated_file_is_rejected(self, tmp_path):
        c = _cache(tmp_path)
        c.store(_coords(), _hessian())
        path = c.path_for(_coords())
        path.write_bytes(path.read_bytes()[:40])
        assert c.load(_coords()) is None
        assert c.rejected == 1

    def test_an_empty_file_is_rejected(self, tmp_path):
        c = _cache(tmp_path)
        c.store(_coords(), _hessian())
        c.path_for(_coords()).write_bytes(b"")
        assert c.load(_coords()) is None

    def test_a_file_whose_contents_disagree_with_its_name_is_rejected(self, tmp_path):
        # Simulates a hash collision, or a cache directory copied between
        # machines: the filename says one thing, the payload another. The
        # payload is checked, so this must miss.
        c = _cache(tmp_path)
        c.store(_coords(0.0), _hessian())
        good = c.path_for(_coords(0.0))
        other = _cache(tmp_path)
        other.store(_coords(0.3), _hessian(9))
        good.write_bytes(other.path_for(_coords(0.3)).read_bytes())
        assert c.load(_coords(0.0)) is None
        assert c.rejected >= 1

    def test_a_wrong_shaped_hessian_is_rejected(self, tmp_path):
        c = _cache(tmp_path)
        c.store(_coords(), np.eye(9))          # 9x9 for a 4-atom molecule
        assert c.load(_coords()) is None

    def test_a_miss_after_rejection_recomputes(self, tmp_path):
        c = _cache(tmp_path)
        c.store(_coords(), _hessian())
        c.path_for(_coords()).write_bytes(b"junk")
        calls = []
        fn = c.wrap(lambda x: (calls.append(1), _hessian(5))[1])
        got = fn(_coords())
        assert len(calls) == 1
        np.testing.assert_allclose(got, _hessian(5), atol=0.0)
        # And the bad file was replaced, so the next call hits.
        np.testing.assert_allclose(fn(_coords()), _hessian(5), atol=0.0)
        assert len(calls) == 1


class TestDisabled:
    def test_disabled_never_reads_or_writes(self, tmp_path):
        c = _cache(tmp_path, enabled=False)
        c.store(_coords(), _hessian())
        assert c.load(_coords()) is None
        assert not any(tmp_path.rglob("*.npz"))

    def test_disabled_still_computes(self, tmp_path):
        calls = []
        fn = _cache(tmp_path, enabled=False).wrap(
            lambda x: (calls.append(1), _hessian())[1])
        fn(_coords())
        fn(_coords())
        assert len(calls) == 2


class TestAtomicWrites:
    def test_no_temporary_files_survive_a_store(self, tmp_path):
        # The process this protects against dies at an arbitrary moment, so a
        # reader must never find a half-written file.
        c = _cache(tmp_path)
        c.store(_coords(), _hessian())
        leftovers = [p for p in tmp_path.rglob("*") if p.is_file()
                     and ".tmp" in p.name]
        assert leftovers == []

    def test_restoring_over_an_existing_entry_works(self, tmp_path):
        c = _cache(tmp_path)
        c.store(_coords(), _hessian(1))
        c.store(_coords(), _hessian(2))
        np.testing.assert_allclose(c.load(_coords()), _hessian(2), atol=0.0)


class TestCacheKey:
    def test_the_key_is_stable_across_calls(self):
        a = cache_key("mp2", "cc-pvtz", _ELEMS, _coords())
        b = cache_key("mp2", "cc-pvtz", _ELEMS, _coords())
        assert a == b

    @pytest.mark.parametrize("kw", [
        {"method": "hf"}, {"basis": "6-31g"},
        {"elements": ["C", "N", "H", "H"]},
    ])
    def test_the_key_changes_with_every_keyed_input(self, kw):
        base = dict(method="mp2", basis="cc-pvtz", elements=_ELEMS,
                    coords_ang=_coords())
        assert cache_key(**base) != cache_key(**{**base, **kw})

    def test_the_key_changes_with_the_geometry(self):
        assert (cache_key("mp2", "cc-pvtz", _ELEMS, _coords(0.0))
                != cache_key("mp2", "cc-pvtz", _ELEMS, _coords(0.1)))

    def test_masses_are_deliberately_not_in_the_key(self):
        # A Hessian is mass-independent, so one cached matrix serves every
        # isotopologue. That is where most of the saving comes from, and it is
        # a property worth pinning rather than leaving implicit.
        import inspect
        assert "mass" not in inspect.signature(cache_key).parameters
