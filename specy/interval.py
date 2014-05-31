# Copyright (C) 2014 Mattia Penati <mattia.penati@gmail.com>
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


class LinearTransformation(object):
    '''Represent a linear transformation'''
    __slots__ = ['_alpha', '_beta']

    def __new__(cls, alpha, beta):
        '''Construct the new linear transformation x -> alpha * x + beta'''
        result = super(LinearTransformation, cls).__new__(cls)
        result._alpha, result._beta = alpha, beta
        return result

    def __call__(self, x):
        '''Compute the linear transformation'''
        return self._alpha * x + self._beta

    @property
    def derivative(self):
        '''Give the derivative of linear transformation'''
        return LinearTransformation(0, self._alpha)


class Interval(object):
    __slots__ = ['_a', '_b']

    def __new__(cls, a, b):
        '''Create the new interval (a,b)'''
        assert a <= b, "please specify a valid interval"
        result = super(Interval, cls).__new__(cls)
        result._a, result._b = a, b
        return result

    @property
    def size(self):
        '''Give the size of interval'''
        return self._b - self._a

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    def __rshift__(self, other):
        '''Create the linear transformation (a,b) >> (c,d)'''
        assert isinstance(other, Interval), 'please specify a valid interval'

        alpha = other.size / self.size
        beta = other._a - alpha * self._a
        return LinearTransformation(alpha, beta)

    def __lshift__(self, other):
        '''Create the linear transformation (a,b) << (c,d)'''
        return other.__rshift__(self)

    def __str__(self):
        return '({0}, {1})'.format(self._a, self._b)
