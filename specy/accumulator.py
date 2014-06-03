# Copyright (C) 2014 Mattia Penati <mattia.penati@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


class Accumulator(object):
    __slots__ = ['_sum', '_recover']

    def __new__(cls, initial_value=0.):
        result = super(Accumulator, cls).__new__(cls)
        result._sum = initial_value
        result._recover = 0.
        return result

    @property
    def value(self):
        return self._sum

    def __iadd__(self, value):
        y = value - self._recover
        t = self._sum + y
        self._recover = (t - self._sum) - y
        self._sum = t
        return self
