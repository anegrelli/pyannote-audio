# MIT License
#
# Copyright (c) 2023- CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Context: https://github.com/pyannote/pyannote-audio/issues/1370

import warnings

import torch


class ReproducibilityError(Exception):
    ...


class ReproducibilityWarning(UserWarning):
    ...


_precision_owner_declared = False


def declare_precision_owner() -> None:
    """Hand the TF32 policy over to the calling application.

    Once declared, the helpers below neither read nor write torch precision
    state and never warn, so an application that sets its own policy is not
    overridden by every inference call. The declaration lasts for the process.
    """
    global _precision_owner_declared
    _precision_owner_declared = True


def precision_owner_declared() -> bool:
    return _precision_owner_declared


def _has_per_operation_api() -> bool:
    return hasattr(torch.backends.cuda.matmul, "fp32_precision")


def _tf32_enabled() -> bool:
    # On torch 2.9+ the legacy getters raise once the legacy and per-operation
    # APIs have been mixed, so read the per-operation values whenever they exist.
    if _has_per_operation_api():
        return "tf32" in (
            torch.backends.cuda.matmul.fp32_precision,
            torch.backends.cudnn.conv.fp32_precision,
            torch.backends.cudnn.rnn.fp32_precision,
        )
    return torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32


def _disable_tf32() -> None:
    # The legacy setters write both representations but can leave a
    # per-operation value at "none", which inherits from cuda/all and then
    # generic/all and may still resolve to "tf32"; the explicit "ieee" writes
    # are what turn TF32 off for each operation.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if _has_per_operation_api():
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cudnn.rnn.fp32_precision = "ieee"


def raise_reproducibility(device: torch.device):
    if _precision_owner_declared:
        return
    if (device.type == "cuda") and _tf32_enabled():
        raise ReproducibilityError(
            "Please disable TensorFloat-32 (TF32) by calling\n"
            "   >>> import torch\n"
            "   >>> torch.backends.cuda.matmul.allow_tf32 = False\n"
            "   >>> torch.backends.cudnn.allow_tf32 = False\n"
            "or you might face reproducibility issues and obtain lower accuracy.\n"
            "See https://github.com/pyannote/pyannote-audio/issues/1370 for more details."
        )


def warn_reproducibility(device: torch.device):
    if _precision_owner_declared:
        return
    if (device.type == "cuda") and _tf32_enabled():
        warnings.warn(
            ReproducibilityWarning(
                "Please disable TensorFloat-32 (TF32) by calling\n"
                "   >>> import torch\n"
                "   >>> torch.backends.cuda.matmul.allow_tf32 = False\n"
                "   >>> torch.backends.cudnn.allow_tf32 = False\n"
                "or you might face reproducibility issues and obtain lower accuracy.\n"
                "See https://github.com/pyannote/pyannote-audio/issues/1370 for more details."
            )
        )


def fix_reproducibility(device: torch.device):
    if _precision_owner_declared:
        return
    if (device.type == "cuda") and _tf32_enabled():
        _disable_tf32()
        warnings.warn(
            ReproducibilityWarning(
                "TensorFloat-32 (TF32) has been disabled as it might lead to reproducibility issues and lower accuracy.\n"
                "It can be re-enabled by calling\n"
                "   >>> import torch\n"
                "   >>> torch.backends.cuda.matmul.allow_tf32 = True\n"
                "   >>> torch.backends.cudnn.allow_tf32 = True\n"
                "See https://github.com/pyannote/pyannote-audio/issues/1370 for more details.\n"
            )
        )
