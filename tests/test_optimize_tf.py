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
"""
Unit tests for the :mod:`pennylane` optimizers.
"""
# pylint: disable=redefined-outer-name
import numpy as onp
import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.utils import _flatten
from pennylane.optimize import (RotosolveOptimizerTF)

x_vals = np.linspace(-10, 10, 16, endpoint=False)

# functions and their gradients
univariate_funcs = [np.sin,
                    lambda x: np.exp(x / 10.),
                    lambda x: x ** 2]
multivariate_funcs = [lambda x: np.sin(x[0]) + np.cos(x[1]),
                      lambda x: np.exp(x[0] / 3) * np.tanh(x[1]),
                      lambda x: np.sum([x_ ** 2 for x_ in x])]

@pytest.fixture(scope="function")
def bunch():
    class A:
        rotosolvetf_opt = RotosolveOptimizerTF()
    return A()


class TestOptimizer:
    """Basic optimizer tests.
    """

    @staticmethod
    def rotosolvetf_step(f, x):
        """Helper function to test the RotosolveTF optimizers"""
        # make sure that x is an array
        if np.ndim(x) == 0:
            x = np.array([x])

        # helper function for x[d] = theta
        def insert(xf, d, theta):
            xf[d] = theta
            return xf

        for d, _ in enumerate(x):
            H_0 = float(f(insert(x, d, 0)))
            H_p = float(f(insert(x, d, np.pi / 2)))
            H_m = float(f(insert(x, d, -np.pi / 2)))
            a = onp.arctan2(2 * H_0 - H_p - H_m, H_p - H_m)

            x[d] = -np.pi / 2 - a

            if x[d] <= -np.pi:
                x[d] += 2 * np.pi
        return x

    @pytest.mark.parametrize('x_start', x_vals)
    def test_rotosolvetf_optimizer_univar(self, x_start, bunch, tol):
        """Tests that rotosolvetf optimizer takes one and two steps correctly
        for uni-variate functions."""

        for f in univariate_funcs:
            x_onestep = bunch.rotosolvetf_opt.step(f, x_start)
            x_onestep_target = self.rotosolvetf_step(f, x_start)

            assert np.allclose(x_onestep, x_onestep_target, atol=tol, rtol=0)

            x_twosteps = bunch.rotosolvetf_opt.step(f, x_onestep)
            x_twosteps_target = self.rotosolvetf_step(f, x_onestep_target)

            assert np.allclose(x_twosteps, x_twosteps_target, atol=tol, rtol=0)

    @pytest.mark.parametrize('x_start', [[1.2, 0.2],
                                         [-0.62, -2.1],
                                         [0.05, 0.8],
                                         [[0.3], [0.25]],
                                         [[-0.6], [0.45]],
                                         [[1.3], [-0.9]]])
    def test_rotosolvetf_optimizer_multivar(self, x_start, bunch, tol):
        """Tests that rotosolvetf optimizer takes one and two steps correctly
        for multi-variate functions."""

        for func in multivariate_funcs:
            # alter multivariate_func to accept nested lists of parameters
            f = lambda x: func(np.ravel(x))

            x_onestep = bunch.rotosolvetf_opt.step(f, x_start)
            x_onestep_target = self.rotosolvetf_step(f, x_start)

            assert x_onestep == pytest.approx(x_onestep_target, abs=tol)

            x_twosteps = bunch.rotosolvetf_opt.step(f, x_onestep)
            x_twosteps_target = self.rotosolvetf_step(f, x_onestep_target)

            assert x_twosteps == pytest.approx(x_twosteps_target, abs=tol)

class TestExceptions:
    """Test exceptions are raised for incorrect usage"""

    def test_obj_func_not_a_qnode(self):
        """Test that if the objective function is not a
        QNode, an error is raised."""

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(a):
            qml.RX(a, wires=0)
            return qml.expval(qml.PauliZ(0))

        def cost(a):
            return circuit(a)

        opt = qml.QNGOptimizerTF()
        params = 0.5

        with pytest.raises(
            ValueError, match="The objective function must either be encoded as a single QNode or a VQECost object"
        ):
            opt.step(cost, params)

class TestOptimize:
    """Test basic optimization integration"""

    def test_qubit_rotation(self, tol):
        """Test qubit rotation has the correct QNG value
        every step, the correct parameter updates,
        and correct cost after 200 steps"""
        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            return qml.expval(qml.PauliZ(0))

        def gradient(params):
            """Returns the gradient of the above circuit"""
            da = -np.sin(params[0]) * np.cos(params[1])
            db = -np.cos(params[0]) * np.sin(params[1])
            return np.array([da, db])

        eta = 0.01
        init_params = np.array([0.011, 0.012])
        num_steps = 200

        opt = qml.QNGOptimizerTF(eta)
        theta = init_params

        # optimization for 200 steps total
        for t in range(num_steps):
            theta_new = opt.step(circuit, theta)

            # check metric tensor
            res = opt.metric_tensor
            exp = np.diag([0.25, (np.cos(theta[0]) ** 2)/4])
            assert np.allclose(res, exp, atol=tol, rtol=0)

            # check parameter update
            dtheta = eta * sp.linalg.pinvh(exp) @ gradient(theta)
            assert np.allclose(dtheta, theta - theta_new, atol=tol, rtol=0)

            theta = theta_new

        # check final cost
        assert np.allclose(circuit(theta), -0.9963791, atol=tol, rtol=0)

    def test_single_qubit_vqe(self, tol):
        """Test single-qubit VQE has the correct QNG value
        every step, the correct parameter updates,
        and correct cost after 200 steps"""
        dev = qml.device("default.qubit", wires=1)

        def circuit(params, wires=0):
            qml.RX(params[0], wires=wires)
            qml.RY(params[1], wires=wires)

        coeffs = [1, 1]
        obs_list = [
            qml.PauliX(0),
            qml.PauliZ(0)
        ]

        qnodes = qml.map(circuit, obs_list, dev, measure='expval')
        cost_fn = qml.dot(coeffs, qnodes)

        def gradient(params):
            """Returns the gradient"""
            da = -np.sin(params[0]) * (np.cos(params[1]) + np.sin(params[1]))
            db = np.cos(params[0]) * (np.cos(params[1]) - np.sin(params[1]))
            return np.array([da, db])

        eta = 0.01
        init_params = np.array([0.011, 0.012])
        num_steps = 200

        opt = qml.QNGOptimizerTF(eta)
        theta = init_params

        # optimization for 200 steps total
        for t in range(num_steps):
            theta_new = opt.step(cost_fn, theta,
                                 metric_tensor_fn=qnodes.qnodes[0].metric_tensor)

            # check metric tensor
            res = opt.metric_tensor
            exp = np.diag([0.25, (np.cos(theta[0]) ** 2)/4])
            assert np.allclose(res, exp, atol=tol, rtol=0)

            # check parameter update
            dtheta = eta * sp.linalg.pinvh(exp) @ gradient(theta)
            assert np.allclose(dtheta, theta - theta_new, atol=tol, rtol=0)

            theta = theta_new

        # check final cost
        assert np.allclose(cost_fn(theta), -1.41421356, atol=tol, rtol=0)

    def test_single_qubit_vqe_using_vqecost(self, tol):
        """Test single-qubit VQE using VQECost 
        has the correct QNG value every step, the correct parameter updates,
        and correct cost after 200 steps"""
        dev = qml.device("default.qubit", wires=1)

        def circuit(params, wires=0):
            qml.RX(params[0], wires=wires)
            qml.RY(params[1], wires=wires)

        coeffs = [1, 1]
        obs_list = [
            qml.PauliX(0),
            qml.PauliZ(0)
        ]

        h = qml.Hamiltonian(coeffs=coeffs, observables=obs_list)

        cost_fn = qml.VQECost(ansatz=circuit, hamiltonian=h, device=dev)

        def gradient(params):
            """Returns the gradient"""
            da = -np.sin(params[0]) * (np.cos(params[1]) + np.sin(params[1]))
            db = np.cos(params[0]) * (np.cos(params[1]) - np.sin(params[1]))
            return np.array([da, db])

        eta = 0.01
        init_params = np.array([0.011, 0.012])
        num_steps = 200

        opt = qml.QNGOptimizerTF(eta)
        theta = init_params

        # optimization for 200 steps total
        for t in range(num_steps):
            theta_new = opt.step(cost_fn, theta)

            # check metric tensor
            res = opt.metric_tensor
            exp = np.diag([0.25, (np.cos(theta[0]) ** 2)/4])
            assert np.allclose(res, exp, atol=tol, rtol=0)

            # check parameter update
            dtheta = eta * sp.linalg.pinvh(exp) @ gradient(theta)
            assert np.allclose(dtheta, theta - theta_new, atol=tol, rtol=0)

            theta = theta_new

        # check final cost
        assert np.allclose(cost_fn(theta), -1.41421356, atol=tol, rtol=0)
