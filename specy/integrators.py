# Copyright (C) 2014 Mattia Penati <mattia.penati@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from .interval import Interval
from .accumulator import Accumulator
import numpy


class Integrator(object):
    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        raise NotImplementedError()


class ConstantTimeStepping(Integrator):
    def __init__(self, stepper, x0, t0, dt):
        self.stepper = stepper
        self._x0 = x0
        self._t0 = Accumulator(t0)
        self._dt = dt

    def __next__(self):
        # update time step
        t0 = self._t0.value
        self._t0 += self._dt
        t1 = self._t0.value

        # compute the next step
        time_interval = Interval(t0, t1)
        xnext, data = self.stepper.step(self._x0, time_interval)

        # update status
        self._x0 = xnext

        return xnext, t1, data


class AdaptiveTimeStepping(Integrator):
    def __init__(
            self, steppers, x0, t0, dt=None,
            absolute_tolerance=1.e-4,
            relative_tolerance=1.e-3):
        self.stepper, self.estimator = steppers
        self._x0 = x0
        self._t0 = Accumulator(t0)
        if dt is None:
            # TODO initial guess
            pass
        else:
            self._dt = dt

        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance

    def __next__(self):
        # update time step
        t0 = self._t0.value
        self._t0 += self._dt
        t1 = self._t0.value

        # compute the next step
        time_interval = Interval(t0, t1)
        xnext, data = self.stepper.step(self._x0, time_interval)
        xestimated, data_estimated = \
            self.estimator.step(self._x0, time_interval)

        # estimate the error
        sc0 = self.absolute_tolerance + \
            self.relative_tolerance * numpy.maximum(
                numpy.absolute(self._x0[0]),
                numpy.absolute(xnext[0]))
        sc1 = self.absolute_tolerance + \
            self.relative_tolerance * numpy.maximum(
                numpy.absolute(self._x0[1]),
                numpy.absolute(xnext[1]))
        error0 = numpy.linalg.norm((xnext[0] - xestimated[0]) / sc0)
        error1 = numpy.linalg.norm((xnext[1] - xestimated[1]) / sc0)

        error0 /= numpy.sqrt(xnext[0].size)
        error1 /= numpy.sqrt(xnext[1].size)

        error = max(error0, error1)

        # update timestep
        error += numpy.spacing(1)
        self._dt *= (self.absolute_tolerance / error) ** \
            (1. / (2. * self.stepper.degree))

        # update status
        self._x0 = xnext

        return xnext, t1, data
