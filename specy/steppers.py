# Copyright (C) 2014 Mattia Penati <mattia.penati@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy
import scipy.optimize
from .interval import Interval
from .system import MechanicalSystem
from .polynomials import PolynomialSpace
from .quadrature import QuadratureRule


class Stepper(object):
    pass


class SpectralVariationalData(object):
    def __init__(self, dimension, num_of_nodes, num_of_modes):
        # by columns, each column containts a modal (or a nodal) value
        self.position = numpy.empty((dimension, num_of_modes))
        self.momentum = numpy.empty((dimension, num_of_nodes))
        self.next_momentum = numpy.empty(dimension)

    @property
    def size(self):
        return self.position.size + \
            self.momentum.size + self.next_momentum.size

    def to_vector(self, result):
        # position
        offset, size = 0, self.position.size
        result[offset:offset+size] = self.position.reshape(size)
        # momentum
        offset, size = offset+size, self.momentum.size
        result[offset:offset+size] = self.momentum.reshape(size)
        # next_momentum
        offset, size = offset+size, self.next_momentum.size
        result[offset:offset+size] = self.next_momentum.reshape(size)

    def from_vector(self, result):
        # position
        offset, size = 0, self.position.size
        self.position.reshape(size)[:] = result[offset:offset+size]
        # momentum
        offset, size = offset+size, self.momentum.size
        self.momentum.reshape(size)[:] = result[offset:offset+size]
        # next_momentum
        offset, size = offset+size, self.next_momentum.size
        self.next_momentum.reshape(size)[:] = result[offset:offset+size]


class SpectralVariationalEvaluator(object):
    def __init__(self, stepper, q_p, time_interval):
        self.stepper = stepper
        self.curr_position, self.curr_momentum = q_p
        self.time_interval = time_interval

    def __call__(self, data, result):
        assert isinstance(data, SpectralVariationalData)
        assert isinstance(result, SpectralVariationalData)

        # polynomial base
        values = self.stepper.values
        derivs = self.stepper.derivs
        num_of_modes = self.stepper.num_of_modes

        # quadrature
        nodes = self.stepper.nodes
        weights = self.stepper.weights
        num_of_nodes = self.stepper.num_of_nodes

        # rescale the time interval
        eta = self.stepper.reference_interval >> self.time_interval
        t = eta(nodes)

        # mechanical system
        system = self.stepper.system

        # get nodal positions and velocities (by colums)
        nodal_position = \
            numpy.dot(data.position, values)
        nodal_velocity = \
            numpy.dot(data.position, derivs) / eta.derivative(nodes)

        # boundary terms
        for k in range(num_of_modes):
            result.position[:, :] = \
                numpy.outer(
                    data.next_momentum, self.stepper.rboundary_values) - \
                numpy.outer(
                    self.curr_momentum, self.stepper.lboundary_values)
        result.next_momentum = \
            numpy.dot(data.position, self.stepper.lboundary_values) - \
            self.curr_position

        # try to evaluate the Lagrangian's derivative
        try:
            for i in range(num_of_nodes):
                q_dotq = (nodal_position[:, i], nodal_velocity[:, i])
                force, momentum = \
                    system.lagrangian_derivative(q_dotq, t[i])
                # first equation
                result.momentum[:, i] = data.momentum[:, i] - momentum
                # second equation
                result.position -= weights[i] * (
                    numpy.outer(data.momentum[:, i], derivs[:, i]) +
                    eta.derivative(nodes[i]) * numpy.outer(force, values[:, i])
                )

            return
        except NotImplementedError:
            pass

        # try to evaluate the Hamiltonian's derivative
        try:
            for i in range(num_of_nodes):
                q_p = (nodal_position[:, i], data.momentum[:, i])
                force, velocity = \
                    system.hamiltonian_derivative(q_p, t[i])
                # first equation
                result.momentum[:, i] = nodal_velocity[:, i] - velocity
                # second equation
                result.position -= weights[i] * (
                    numpy.dot(data.momentum, derivs.T) -
                    eta.derivative(nodes[i]) * numpy.outer(force, values[:, i])
                )

            return
        except NotImplementedError:
            pass

        # no Lagrangian and no Hamiltonian
        raise ValueError('invalid system is given')


class SpectralVariationalStepper(Stepper):
    def __init__(self, polynomial_space, quadrature_rule, system):
        assert isinstance(polynomial_space, PolynomialSpace), \
            'please specify a valid polynomial space'
        assert isinstance(quadrature_rule, QuadratureRule), \
            'please specify a valid quadrature rule'
        assert isinstance(system, MechanicalSystem), \
            'please specify a valid mechanical system'

        # reference interval (of the polynomials space)
        self.reference_interval = polynomial_space.reference_interval
        # nodes and weights
        self.nodes, self.weights = \
            quadrature_rule.rescale(self.reference_interval)
        # polynomial space
        self.degree = polynomial_space.degree
        self.values, self.derivs = polynomial_space.evaluate(self.nodes)
        self.lboundary_values, _ = \
            polynomial_space.evaluate(self.reference_interval.a)
        self.rboundary_values, _ = \
            polynomial_space.evaluate(self.reference_interval.b)
        # mechanical system
        self.system = system

    def step(self, q_p, time_interval):
        # instantiate everything
        evaluator = SpectralVariationalEvaluator(self, q_p, time_interval)

        initial_data = SpectralVariationalData(
            self.system.dimension, self.num_of_nodes, self.num_of_modes)
        residual = SpectralVariationalData(
            self.system.dimension, self.num_of_nodes, self.num_of_modes)

        # initial guess
        q0, p0 = q_p
        initial_data.position[:, 0] = q0
        initial_data.position[:, 1:] = 0
        initial_data.momentum[:, :] = 0
        initial_data.next_momentum[:] = 0

        # newton method
        def f(x):
            initial_data.from_vector(x)
            evaluator(initial_data, residual)
            result = numpy.empty_like(x)
            residual.to_vector(result)
            return result

        x0 = numpy.empty(initial_data.size)
        initial_data.to_vector(x0)
        sol = scipy.optimize.root(f, x0)
        initial_data.from_vector(sol.x)

        # compute the position
        q1 = numpy.dot(initial_data.position, self.rboundary_values)
        p1 = initial_data.next_momentum[:]

        # reshape if needed
        if self.system.dimension == 1:
            q1, p1 = q1[0], p1[0]
            initial_data.position.shape = (self.num_of_modes,)
            initial_data.momentum.shape = (self.num_of_nodes,)

        # return
        return (q1, p1), (initial_data.position, initial_data.momentum)

    @property
    def num_of_nodes(self):
        return len(self.nodes)

    @property
    def num_of_modes(self):
        return self.degree + 1
