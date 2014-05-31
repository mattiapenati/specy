# Copyright (C) 2014 Mattia Penati <mattia.penati@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


class MechanicalSystem(object):
    def __init__(self, dimension):
        self._dimension = dimension

    @property
    def dimension(self):
        return self._dimension

    def lagrangian(self, q_dotq, t):
        raise NotImplementedError()

    def hamiltonian(self, q_p, t):
        raise NotImplementedError()

    def lagrangian_derivative(self, q_dotq, t):
        # TODO default implementation must use the the numerical derivative
        raise NotImplementedError()

    def hamiltonian_derivative(self, q_p, t):
        # TODO default implementation must use the the numerical derivative
        raise NotImplementedError()
