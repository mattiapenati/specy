# Copyright (C) 2014 Mattia Penati <mattia.penati@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy
from .interval import Interval


class PolynomialSpace(object):
    def __init__(self, degree):
        self._degree = degree

    @property
    def degree(self):
        return self._degree

    def eval(self, nodes):
        raise NotImplementedError()


class LegendrePolynomial(PolynomialSpace):
    def __init__(self, degree):
        super(LegendrePolynomial, self).__init__(degree)

        # setting informations
        self.reference_interval = Interval(-1., +1.)

    def evaluate(self, nodes):
        num_of_nodes = numpy.size(nodes)
        values = numpy.empty((self.degree + 1, num_of_nodes))
        derivs = numpy.empty((self.degree + 1, num_of_nodes))

        # initialization
        values[0, :] = 1
        values[1, :] = nodes

        derivs[0, :] = 0
        derivs[1, :] = 1

        # three terms recurrence
        a = lambda i: (2. * i + 1.) / (i + 1.)
        b = lambda i: i / (i + 1.)

        for i in range(1, self.degree):
            values[i+1, :] = \
                a(i) * nodes * values[i, :] - b(i) * values[i-1, :]

            derivs[i+1, :] = \
                a(i) * (values[i, :] + nodes * derivs[i, :]) - \
                b(i) * derivs[i-1, :]

        # reshape to vectors if needed
        if values.shape[1] == 1:
            values.shape = (values.shape[0],)
            derivs.shape = (derivs.shape[0],)

        return values, derivs
