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

    def error_estimator(
        self, x, x_next, x_estimated, absolute_tolerance, relative_tolerance):
        sc0 = self.absolute_tolerance + \
            self.relative_tolerance * numpy.maximum(
                numpy.absolute(x[0]), numpy.absolute(x_next[0]))
        sc1 = self.absolute_tolerance + \
            self.relative_tolerance * numpy.maximum(
                numpy.absolute(x[1]), numpy.absolute(x_next[1]))
        error0 = numpy.linalg.norm((x_next[0] - x_estimated[0]) / sc0)
        error1 = numpy.linalg.norm((x_next[1] - x_estimated[1]) / sc1)

        error0 /= numpy.sqrt(x_next[0].size)
        error1 /= numpy.sqrt(x_next[1].size)

        return max(error0, error1)


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
        self._data0 = None
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
        xnext, data = self.stepper.step(
            self._x0, time_interval, guess=self._data0)
        self._data0 = data

        # compute the estimeted solution
        # TODO guess = self.estimator.interpolate(data, time_interval)
        xestimated, data_estimated = \
            self.estimator.step(self._x0, time_interval)

        # estimate the error
        error = self.error_estimator(
            self._x0, xnext, xestimated,
            self.absolute_tolerance, self.relative_tolerance)

        # update timestep
        error += numpy.spacing(1)
        self._dt *= (self.absolute_tolerance / error) ** \
            (1. / self.stepper.order)

        # update status
        self._x0 = xnext

        return xnext, t1, data


class OrderAdaptiveTimeStepping(Integrator):
    def __init__(
            self, steppers, x0, t0, dt=None,
            absolute_tolerance=1.e-4,
            relative_tolerance=1.e-3):
        self.steppers = steppers
        self._x0 = x0
        self._t0 = Accumulator(t0)
        self._data0 = None
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

        time_interval = Interval(t0, t1)
        for i in range(len(self.steppers) - 1):
            # compute the next step
            xnext, data = self.steppers[i].step(
                self._x0, time_interval, guess=self._data0)

            # compute the estimeted solution
            # TODO guess
            xestimated, data_estimated = \
                self.steppers[i+1].step(self._x0, time_interval)

            # estimate the error
            error = self.error_estimator(
                self._x0, xnext, xestimated,
                self.absolute_tolerance, self.relative_tolerance)

            # update timestep or order
            error += numpy.spacing(1)
            correction = (self.absolute_tolerance / error) ** \
                (1. / (2. * self.steppers[i].degree))

            if correction > 1:
                # update status
                self._x0 = xnext
                self._data0 = data
                self._dt *= correction
                print self.steppers[i].num_of_modes - 1
                return self._x0, t1, self._data0

        # update status
        self._x0 = xnext
        self._data0 = data
        self._dt *= correction
        return self._x0, t1, self._data0

