import numpy
import scipy.special
import matplotlib.pyplot
import matplotlib.animation
import specy

class Pendulum(specy.MechanicalSystem):
    def __init__(self, omega0 = 1.):
        super(Pendulum, self).__init__(1)
        self.omega0 = omega0

    def lagrangian_derivative(self, q_p, t):
        q, p = q_p
        return -self.omega0 ** 2 * numpy.sin(q), p

class PendulumSolution(object):
    def __init__(self, q0, omega0 = 1.):
        self.q0 = q0
        self.omega0 = omega0
        self.sintheta0 = numpy.sin(q0 / 2.)
        self.K = scipy.special.ellipk(self.sintheta0 ** 2)

    def __call__(self, time):
        sn, cn, dn, ph = scipy.special.ellipj(
            self.K - self.omega0 * time, self.sintheta0 ** 2)
        return 2 * numpy.arcsin(self.sintheta0 * sn)

class Plotter(object):
    def __init__(self, system, solution = None):
        self.figure, self.axis = matplotlib.pyplot.subplots()
        self.axis.grid()

        self.dimension = d = system.dimension
        self.lines = [self.axis.plot([], [], 'r-', lw=2)[0] for i in range(d)]

        self.time = []
        self.data = [[] for i in range(d)]

        self.solution = solution
        if solution is not None:
            self.solution_lines = [
                self.axis.plot([], [], 'b-', lw=2)[0] for i in range(d)
            ]
            self.solution_data = [[] for i in range(d)]

    def plot(self, integrator):
        # generate data
        def data_gen():
            while True:
                yield integrator.next()

        # update data
        def run(data):
            # update the data
            q_p, t, interval = data
            q, p = q_p
            print t

            position, momentum, time = interval
            position = numpy.dot(position, integrator.stepper.values)

            if self.dimension == 1:
                for qi, pi, ti in zip(position, momentum, time):
                    self.time.append(ti)
                    self.data[0].append(qi)
                    if self.solution is not None:
                        self.solution_data[0].append(self.solution(ti))
                self.time.append(t)
                self.data[0].append(q)
                if self.solution is not None:
                    self.solution_data[0].append(self.solution(t))
            else:
                raise NotImplementedError()

            xmin, xmax = self.axis.get_xlim()

            if t >= xmax:
                self.axis.set_xlim(xmin, 2*xmax)
                self.axis.figure.canvas.draw()

            for i in range(self.dimension):
                self.lines[i].set_data(self.time, self.data[i])
            if self.solution is not None:
                for i in range(self.dimension):
                    self.solution_lines[i].set_data(
                        self.time, self.solution_data[i])
        ani = matplotlib.animation.FuncAnimation(
            self.figure,
            run,
            data_gen,
            blit=False,
            interval=10,
            repeat=False)
        matplotlib.pyplot.show()

# mechanical system
omega0 = 3.
system = Pendulum(omega0)

# integrators
order = 7
order_raise = 1

stepper_poly_space = specy.LegendrePolynomial(order)
stepper_quadrature = specy.GaussLegendreQuadrature(order + 1)

# initial conditions
q0, p0 = 0.99 * numpy.pi, 0.
dt = 5e-1
solution = PendulumSolution(q0, omega0)

# stepper and integrator
stepper = specy.Stepper(stepper_poly_space, stepper_quadrature, system)
integrator = specy.ConstantTimeStepping(stepper, (q0, p0), 0, dt)

# plot
plotter = Plotter(system, solution)
plotter.axis.set_ylim(-numpy.pi, numpy.pi)
plotter.axis.set_xlim(0, 5)

plotter.plot(integrator)
