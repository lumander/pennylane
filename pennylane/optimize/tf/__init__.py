# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
This module contains optimizers for the standard :mod:`QNode` class, which uses the TensorFlow interface.
"""

# Python optimizers that are available in PennyLane
# listed in alphabetical order to avoid circular imports
from .qngtf import QNGOptimizerTF
from .rotosolvetf import RotosolveOptimizerTF


# Optimizers to display in the docs
__all__ = [
    "QNGOptimizerTF",
    "RotosolveOptimizerTF"
]
