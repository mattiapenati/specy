# Copyright (C) 2014 Mattia Penati <mattia.penati@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy


class QuadratureRule(object):
    def rescale(self, interval):
        a, b = interval
        a0, b0 = self.reference_interval

        alpha = (b-a)/(b0-a0)
        beta = a - alpha * a0

        nodes = alpha * self.nodes + beta
        weights = alpha * self.weights

        return nodes, weights


class GaussLegendreQuadrature(QuadratureRule):
    def __init__(self, num_of_nodes):
        super(GaussLegendreQuadrature, self).__init__()

        # setting informations
        self.reference_interval = (-1, +1)

        # nodes and weights
        self.nodes = numpy.empty(num_of_nodes)
        self.weights = numpy.empty(num_of_nodes)

        # computing nodes and weights
        n_vec = numpy.arange(1, num_of_nodes, dtype=numpy.float)
        beta = n_vec / numpy.sqrt(4 * n_vec ** 2 - 1)

        J = numpy.diag(beta, -1) + numpy.diag(beta, +1)
        F = J[::2, :][:, 1::2]
        H = numpy.dot(F, F.T)

        values, vectors = numpy.linalg.eigh(H)
        if num_of_nodes % 2 == 1:  # to avoid sqrt of small negative numbers
            values[0] = 0
        half_nodes = numpy.sqrt(values)
        half_weights = vectors[0, :] ** 2

        if num_of_nodes % 2 == 1:
            half_weights[0] *= 2

            self.weights[:num_of_nodes/2] = half_weights[-1:0:-1]
            self.weights[num_of_nodes/2:] = half_weights[:]

            self.nodes[:num_of_nodes/2] = -half_nodes[-1:0:-1]
            self.nodes[num_of_nodes/2:] = half_nodes[:]
        else:
            self.weights[:num_of_nodes/2] = half_weights[::-1]
            self.weights[num_of_nodes/2:] = half_weights[:]

            self.nodes[:num_of_nodes/2] = -half_nodes[::-1]
            self.nodes[num_of_nodes/2:] = half_nodes[:]
