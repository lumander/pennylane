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
This module contains optimizers for the standard :mod:`QNode` class, for all the available interfaces.
"""

# Python optimizers that are available in PennyLane
# listed in alphabetical order to avoid circular imports
from .autograd.adagrad import AdagradOptimizer
from .autograd.adam import AdamOptimizer
from .autograd.gradient_descent import GradientDescentOptimizer
from .autograd.momentum import MomentumOptimizer
from .autograd.nesterov_momentum import NesterovMomentumOptimizer
from .autograd.rms_prop import RMSPropOptimizer
from .autograd.qng import QNGOptimizer
from .autograd.rotosolve import RotosolveOptimizer
from .autograd.rotoselect import RotoselectOptimizer
from .tf.qngtf import QNGOptimizerTF
from .tf.rotosolvetf import RotosolveOptimizerTF
from .torch.qngtorch import QNGOptimizerTorch
from .torch.rotosolvetorch import RotosolveOptimizerTorch
