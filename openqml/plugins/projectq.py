# Copyright 2018 Xanadu Quantum Technologies Inc.
r"""
ProjectQ plugin
========================

**Module name:** :mod:`openqml.plugins.projectq`

.. currentmodule:: openqml.plugins.projectq

This plugin provides the interface between OpenQML and ProjecQ.
It enables OpenQML to optimize quantum circuits simulable with ProjectQ.

ProjecQ supports several different backends. Of those the following are useful in the current context:

- projectq.backends.Simulator([gate_fusion, ...])	Simulator is a compiler engine which simulates a quantum computer using C++-based kernels.
- projectq.backends.ClassicalSimulator()	        A simple introspective simulator that only permits classical operations.
- projectq.backends.IBMBackend([use_hardware, ...])	The IBM Backend class, which stores the circuit, transforms it to JSON QASM, and sends the circuit through the IBM API.

See PluginAPI._capabilities['backend'] for a list of backend options.

Functions
---------

.. autosummary::
   init_plugin

Classes
-------

.. autosummary::
   Gate
   Observable
   PluginAPI

----
"""

import logging as log
import warnings

import numpy as np

import openqml.plugin
from openqml.circuit import (GateSpec, Command, ParRef, Circuit)

import projectq as pq

# import strawberryfields as sf
# import strawberryfields.ops as sfo
# import strawberryfields.engine as sfe


# tolerance for numerical errors
tolerance = 1e-10

#========================================================
#  define the gate set
#========================================================

class Gate(GateSpec):
    """Implements the quantum gates and observables.
    """
    def __init__(self, name, n_sys, n_par, cls=None, par_domain='R'):
        super().__init__(name, n_sys, n_par, grad_method='F', par_domain=par_domain)
        self.cls = cls  #: class: sf.ops.Operation subclass corresponding to the gate

    def execute(self, par, reg, sim):
        """Applies a single gate or measurement on the current system state.

        Args:
          par (Sequence[float]): gate parameters
          reg   (Sequence[int]): subsystems to which the gate is applied
          sim (~openqml.plugin.PluginAPI): simulator instance keeping track of the system state and measurement results
        """
        # construct the Operation instance
        G = self.cls(*par)
        # apply it
        G | reg


class Observable(Gate):
    """Implements hermitian observables.

    We assume that all the observables in the circuit are consequtive, and commute.
    Since we are only interested in the expectation values, there is no need to project the state after the measurement.
    See :ref:`measurements`.
    """
    def execute(self, par, reg, sim):
        """Estimates the expectation value of the observable in the current system state.

        The arguments are the same as for :meth:`Gate.execute`.
        """
        if self.n_sys != 1:
            raise ValueError('This plugin supports only one-qubit observables.')

        #A = self.cls(*par)  # Operation instance
        # run the queued program so that we obtain the state before the measurement
        state = sim.eng.run(**sim.init_kwargs)  # FIXME remove **kwargs here when SF is updated
        n_eval = sim.n_eval

        if self.cls == sfo.MeasureHomodyne:
            ev, var = state.quad_expectation(reg[0], *par)
        elif self.cls == sfo.MeasureFock:
            ev = state.mean_photon(reg[0])  # FIXME should return var too!
            var = 0
        else:
            warnings.warn('No expectation value method defined for {}.'.format(self.cls))
            ev = 0
            var = 0
        log.info('observable: ev: {}, var: {}'.format(ev, var))

        if n_eval != 0:
            # estimate the ev
            # TODO implement sampling in SF
            # use central limit theorem, sample normal distribution once, only ok if n_eval is large (see https://en.wikipedia.org/wiki/Berry%E2%80%93Esseen_theorem)
            ev = np.random.normal(ev, np.sqrt(var / n_eval))

        sim.eng.register[reg[0]].val = ev  # TODO HACK: store the result (there should be a SF method for computing and storing the expectation value!)


# gates (and state preparations)
Vac  = Gate('Vac', 1, 0, sfo.Vacuum)
Coh  = Gate('Coh', 1, 2, sfo.Coherent)
Squ  = Gate('Squ', 1, 2, sfo.Squeezed)
The  = Gate('The', 1, 1, sfo.Thermal)
Fock = Gate('Fock', 1, 1, sfo.Fock, par_domain='N')
D = Gate('D', 1, 2, sfo.Dgate)
S = Gate('S', 1, 2, sfo.Sgate)
X = Gate('X', 1, 1, sfo.Xgate)
Z = Gate('Z', 1, 1, sfo.Zgate)
R = Gate('R', 1, 1, sfo.Rgate)
F = Gate('Fourier', 1, 0, sfo.Fouriergate)
P = Gate('P', 1, 1, sfo.Pgate)
V = Gate('V', 1, 1, sfo.Vgate)
K = Gate('K', 1, 1, sfo.Kgate)
BS = Gate('BS', 2, 2, sfo.BSgate)
S2 = Gate('S2', 2, 2, sfo.S2gate)
CX = Gate('CX', 2, 1, sfo.CXgate)
CZ = Gate('CZ', 2, 1, sfo.CZgate)

# measurements
MFock = Observable('MFock', 1, 0, sfo.MeasureFock)
MHo   = Observable('MHomodyne', 1, 1, sfo.MeasureHomodyne)
#MX    = Observable('MX', 1, 0, sfo.MeasureX)
#MP    = Observable('MP', 1, 0, sfo.MeasureP)
MHe   = Observable('MHeterodyne', 1, 0, sfo.MeasureHeterodyne)


demo = [
    Command(S,  [0], [ParRef(0), 0]),
    Command(BS, [0, 1], [np.pi/4, 0]),
    Command(R,  [0], [np.pi/3]),
    Command(D,  [1], [ParRef(1), np.pi/3]),
    Command(BS, [0, 1], [-np.pi/4, 0]),
    Command(R,  [0], [ParRef(1)]),
    Command(BS, [0, 1], [np.pi/4, 0]),
]

# circuit templates
_circuit_list = [
  Circuit(demo, 'demo'),
  Circuit(demo +[Command(MHo, [0], [0])], 'demo_ev', out=[0]),
]



class PluginAPI(openqml.plugin.PluginAPI):
    """ProjectQ OpenQML plugin API class.

    Keyword Args:
      backend (str): backend name
    """
    plugin_name = 'Strawberry Fields OpenQML plugin'
    plugin_api_version = '0.1.0'
    plugin_version = sf.version()
    author = 'Xanadu Inc.'
    _circuits = {c.name: c for c in _circuit_list}
    _capabilities = {'backend': list("Simulator", "ClassicalSimulator", "IBMBackend")}

    def __init__(self, name='default', **kwargs):
        super().__init__(name, **kwargs)

        # sensible defaults
        kwargs.setdefault('backend', 'fock')

        # backend-specific capabilities
        self.backend = kwargs['backend']
        # gate and observable sets depend on the backend, so they have to be instance properties
        gates = [Vac, Coh, Squ, The, D, S, X, Z, R, F, P, BS, S2, CX, CZ]
        observables = [MHo]
        if self.backend == 'Simulator':
            pass
        elif self.backend == 'ClassicalSimulator':
            pass
        elif self.backend == 'IBMBackend':
            pass
        else:
            raise ValueError("Unknown backend '{}'.".format(self.backend))

        self._gates = {g.name: g for g in gates}
        self._observables = {g.name: g for g in observables}

        self.init_kwargs = kwargs  #: dict: initialization arguments
        self.eng = None  #: strawberryfields.engine.Engine: engine for executing SF programs

    def __str__(self):
        return super().__str__() +'ProjecQ with Backend: ' +self.backend +'\n'

    def reset(self):
        # reset the engine and backend
        if self.eng is not None:
            self.eng = None  # FIXME this is wasteful, now we construct a new Engine and backend after each reset (because the next circuit may have a different num_subsystems)
            #self.eng.reset()

    def measure(self, A, reg, par=[], n_eval=0):
        temp = self.n_eval  # store the original
        self.n_eval = n_eval
        with self.eng:
            A.execute(par, [reg], self)  # compute the expectation value
        self.n_eval = temp  # restore it
        return self.eng.register[reg].val

    def execute_circuit(self, circuit, params=[], *, reset=True, **kwargs):
        super().execute_circuit(circuit, params, reset=reset, **kwargs)
        circuit = self.circuit

        # set the required number of subsystems
        n = circuit.n_sys
        if self.eng is None:
            self.eng = sfe.Engine(num_subsystems=n, hbar=2)  # FIXME add **self.init_kwargs here when SF is updated, remove hbar=2
        elif self.eng.num_subsystems != n:  # FIXME change to init_num_subsystems when SF is updated to next version
            raise ValueError("Trying to execute a {}-mode circuit '{}' on a {}-mode state.".format(n, circuit.name, self.eng.num_subsystems))

        def parmap(p):
            "Mapping function for gate parameters. Replaces ParRefs with the corresponding parameter values."
            if isinstance(p, ParRef):
                return params[p.idx]
            return p

        # input the program
        reg = self.eng.register
        with self.eng:
            for cmd in circuit.seq:
                # prepare the parameters
                par = map(parmap, cmd.par)
                # execute the gate
                cmd.gate.execute(par, cmd.reg, self)

        self.eng.run(**self.init_kwargs)  # FIXME remove **kwargs here when SF is updated

        if circuit.out is not None:
            # return the estimated expectation values for the requested modes
            return np.array([reg[idx].val for idx in circuit.out])



def init_plugin():
    """Initialize the plugin.

    Every plugin must define this function.
    It should perform whatever initializations are necessary, and then return an API class.

    Returns:
      class: plugin API class
    """

    return PluginAPI
