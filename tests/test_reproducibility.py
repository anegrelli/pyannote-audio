"""Tests for the TF32 reproducibility helpers.

The tests need no fixture: each one saves and restores the torch state it
touches, so the file also runs outside pytest by calling every ``test_*``
function. CUDA branches use ``torch.device("cuda")`` objects only, since the
helpers look at ``device.type`` and never touch a GPU.
"""

import types
import warnings

import torch

from pyannote.audio.utils import reproducibility
from pyannote.audio.utils.reproducibility import (
    ReproducibilityError,
    ReproducibilityWarning,
)

CUDA = torch.device("cuda")
CPU = torch.device("cpu")
PER_OPERATION = hasattr(torch.backends.cuda.matmul, "fp32_precision")


def _operations():
    backends = torch.backends
    return {
        "matmul": backends.cuda.matmul,
        "conv": backends.cudnn.conv,
        "rnn": backends.cudnn.rnn,
    }


def _legacy_reading():
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def _save_state():
    state = {"legacy": _legacy_reading()}
    if PER_OPERATION:
        state["cudnn"] = torch.backends.cudnn.fp32_precision
        state["operations"] = {
            name: op.fp32_precision for name, op in _operations().items()
        }
    return state


def _restore_state(state):
    # Legacy setters first and per-operation values last, so both
    # representations end consistent and the legacy getters, which raise once
    # the two APIs disagree, read the saved values again for whatever runs next.
    legacy = state["legacy"]
    torch.backends.cuda.matmul.allow_tf32 = legacy["matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = legacy["cudnn_allow_tf32"]
    torch.set_float32_matmul_precision(legacy["float32_matmul_precision"])
    if PER_OPERATION:
        torch.backends.cudnn.fp32_precision = state["cudnn"]
        for name, op in _operations().items():
            op.fp32_precision = state["operations"][name]
    assert _legacy_reading() == legacy, "legacy precision getters not restored"


def _enable_tf32():
    if PER_OPERATION:
        for op in _operations().values():
            op.fp32_precision = "tf32"
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _tf32_reading():
    if PER_OPERATION:
        return {name: op.fp32_precision for name, op in _operations().items()}
    return {
        "matmul": torch.backends.cuda.matmul.allow_tf32,
        "cudnn": torch.backends.cudnn.allow_tf32,
    }


def _reset_owner():
    reproducibility._precision_owner_declared = False


def _reproducibility_warnings(caught):
    return [w for w in caught if issubclass(w.category, ReproducibilityWarning)]


class _Untouchable:
    """Stands in for torch and fails on any attribute access."""

    def __getattr__(self, name):
        raise AssertionError(f"torch.{name} was accessed")


def _legacy_torch(matmul, cudnn):
    """Builds a torch stand-in without the per-operation precision API."""
    return types.SimpleNamespace(
        backends=types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                matmul=types.SimpleNamespace(allow_tf32=matmul)
            ),
            cudnn=types.SimpleNamespace(allow_tf32=cudnn),
        )
    )


def test_precision_owner_declaration_is_idempotent():
    _reset_owner()
    try:
        assert reproducibility.precision_owner_declared() is False
        reproducibility.declare_precision_owner()
        reproducibility.declare_precision_owner()
        assert reproducibility.precision_owner_declared() is True
    finally:
        _reset_owner()


def test_declared_owner_keeps_tf32_and_stays_silent():
    state = _save_state()
    _reset_owner()
    try:
        _enable_tf32()
        before = _tf32_reading()
        reproducibility.declare_precision_owner()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reproducibility.fix_reproducibility(CUDA)
            reproducibility.warn_reproducibility(CUDA)
            reproducibility.raise_reproducibility(CUDA)
        assert _reproducibility_warnings(caught) == []
        assert _tf32_reading() == before
    finally:
        _reset_owner()
        _restore_state(state)


def test_declared_owner_never_reads_torch_state():
    real_torch = reproducibility.torch
    _reset_owner()
    try:
        reproducibility.declare_precision_owner()
        reproducibility.torch = _Untouchable()
        reproducibility.fix_reproducibility(CUDA)
        reproducibility.warn_reproducibility(CUDA)
        reproducibility.raise_reproducibility(CUDA)
    finally:
        reproducibility.torch = real_torch
        _reset_owner()


def test_fix_disables_tf32_through_both_apis():
    state = _save_state()
    _reset_owner()
    try:
        _enable_tf32()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reproducibility.fix_reproducibility(CUDA)
        emitted = _reproducibility_warnings(caught)
        assert len(emitted) == 1
        assert str(emitted[0].message).startswith(
            "TensorFloat-32 (TF32) has been disabled"
        )
        if PER_OPERATION:
            assert _tf32_reading() == {
                "matmul": "ieee",
                "conv": "ieee",
                "rnn": "ieee",
            }
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    finally:
        _reset_owner()
        _restore_state(state)


def test_tf32_on_any_single_operation_is_detected():
    if not PER_OPERATION:
        return
    state = _save_state()
    _reset_owner()
    try:
        for name, op in _operations().items():
            for other in _operations().values():
                other.fp32_precision = "ieee"
            op.fp32_precision = "tf32"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                reproducibility.warn_reproducibility(CUDA)
            assert len(_reproducibility_warnings(caught)) == 1, name
            try:
                reproducibility.raise_reproducibility(CUDA)
            except ReproducibilityError:
                pass
            else:
                raise AssertionError(f"no ReproducibilityError for {name}")
    finally:
        _reset_owner()
        _restore_state(state)


def test_tf32_inherited_from_the_cudnn_parent_is_disabled():
    if not PER_OPERATION:
        return
    state = _save_state()
    _reset_owner()
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "none"
        torch.backends.cudnn.rnn.fp32_precision = "none"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reproducibility.fix_reproducibility(CUDA)
        assert len(_reproducibility_warnings(caught)) == 1
        assert _tf32_reading() == {"matmul": "ieee", "conv": "ieee", "rnn": "ieee"}
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    finally:
        _reset_owner()
        _restore_state(state)


def test_ieee_everywhere_is_left_alone():
    if not PER_OPERATION:
        return
    state = _save_state()
    _reset_owner()
    try:
        for op in _operations().values():
            op.fp32_precision = "ieee"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reproducibility.fix_reproducibility(CUDA)
            reproducibility.warn_reproducibility(CUDA)
            reproducibility.raise_reproducibility(CUDA)
        assert _reproducibility_warnings(caught) == []
    finally:
        _reset_owner()
        _restore_state(state)


def test_cpu_device_is_left_alone():
    state = _save_state()
    _reset_owner()
    try:
        _enable_tf32()
        before = _tf32_reading()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reproducibility.fix_reproducibility(CPU)
            reproducibility.warn_reproducibility(CPU)
            reproducibility.raise_reproducibility(CPU)
        assert _reproducibility_warnings(caught) == []
        assert _tf32_reading() == before
    finally:
        _reset_owner()
        _restore_state(state)


def test_torch_without_per_operation_api_uses_legacy_flags():
    real_torch = reproducibility.torch
    _reset_owner()
    try:
        legacy = _legacy_torch(matmul=False, cudnn=True)
        reproducibility.torch = legacy
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reproducibility.warn_reproducibility(CUDA)
            reproducibility.fix_reproducibility(CUDA)
        assert len(_reproducibility_warnings(caught)) == 2
        assert legacy.backends.cuda.matmul.allow_tf32 is False
        assert legacy.backends.cudnn.allow_tf32 is False
        assert not hasattr(legacy.backends.cuda.matmul, "fp32_precision")
        reproducibility.raise_reproducibility(CUDA)

        legacy.backends.cuda.matmul.allow_tf32 = True
        try:
            reproducibility.raise_reproducibility(CUDA)
        except ReproducibilityError:
            pass
        else:
            raise AssertionError("no ReproducibilityError on legacy torch")
    finally:
        reproducibility.torch = real_torch
        _reset_owner()
