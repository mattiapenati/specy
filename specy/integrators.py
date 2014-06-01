# Copyright (C) 2014 Mattia Penati <mattia.penati@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

from .interval import Interval


class Integrator(object):
    def __init__(self, stepper):
        self.stepper = stepper

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        raise NotImplementedError()


class NStepsIntegrator(Integrator):
    def __init__(self, stepper, x0, t0, dt, steps):
        super(NStepsIntegrator, self).__init__(stepper)
        self._x0 = x0
        self._t0 = t0
        self._dt = dt
        self._steps = steps
        self._step = 0

    def __next__(self):
        if self._step < self._steps:
            # compute the next step
            time_interval = Interval(self._t0, self._t0 + self._dt)
            xnext, data = self.stepper.step(self._x0, time_interval)

            # update status
            self._x0 = xnext
            self._t0 += self._dt
            self._step += 1

            return xnext, data
        else:
            raise StopIteration()
