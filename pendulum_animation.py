import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax.numpy as jnp
from integration import derivative_function, eulers_method, pendulum_ode
from diffrax import diffeqsolve, ODETerm, SaveAt, ConstantStepSize, PIDController, Solution, Tsit5


def u(t):
    # forcing function
    return 5



if __name__ == "__main__":
    m = 1.0
    b = 0
    L = 1.0
    G = 9.8
    theta_initial = -jnp.pi/2
    omega_initial = 0
    t_stop = 10    # seconds to simulate
    dt = 0.01   # time between each sample
    t = jnp.arange(0, t_stop, dt)
    # states_at_each_time = eulers_method(theta_initial, omega_initial, t, dt, m, L, G, b, u)
    # trying RK4 with diffrax instead
    term = ODETerm(pendulum_ode)
    solver = Tsit5()
    sol = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=t_stop,
        dt0=dt,
        y0=jnp.array([theta_initial, omega_initial]),
        saveat=SaveAt(ts=t),
        args=(m, L, G, b, u),
    )
    theta = sol.ys[:,0]
    omega = sol.ys[:,1]
    times = sol.ts
    # 
    x = L * jnp.sin(theta)
    y = -L * jnp.cos(theta)

    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    trace, = ax.plot([], [], '.-', lw=1, ms=2)
    time_template = 'time = %.1fs, energy = %.1fJ'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    energy_template = 'Energy = %.1fJ'
    energy_text = ax.text(.25, .25, '', horizontalalignment='left',
        verticalalignment='bottom', transform=ax.transAxes)

    '''def animate(i, states, m, g, l):
        print(i)
        thisx = [0, x[i]]
        thisy = [0, y[i]]

        line.set_data(thisx, thisy)
        time_text.set_text(time_template % (i*dt))
        theta = states[i][0]
        thetadot = states[i][1]
        energy = 0.5 * m * l * thetadot**2 + m * g * l * (1 - jnp.cos(theta))
        energy_text.set_text(energy_template % (energy))

        return line, trace, time_text'''
    
    def animate(i, thetas, omegas, x, y, m, g, l):
        theta = thetas[i]
        thetadot = omegas[i]
        thisx = [0, x[i]]
        thisy = [0, y[i]]

        line.set_data(thisx, thisy)
        energy = 0.5 * m * (l * thetadot)**2 + m * g * l * (1 - jnp.cos(theta))
        time_text.set_text(time_template % (i*dt, energy))

        return line, trace, time_text

    ani = animation.FuncAnimation(
        fig, animate, len(theta), fargs=(theta, omega, x, y, m, G, L, ), interval=t_stop, blit=True)

    # plt.plot(times, theta, color='red')
    # ,plt.plot(times, omega, color='blue')

    plt.show()