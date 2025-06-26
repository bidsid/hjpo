import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax.numpy as jnp
from integration import derivative_function, eulers_method, pendulum_ode
from diffrax import diffeqsolve, ODETerm, SaveAt, Heun

def u(t, m, g, l, b, k, umax, theta, thetadot):
    # forcing function
    return jnp.sin(t)

def swingUpU(t, m, g, l, b, k, umax, theta, thetadot):
    E_desired = 2 * m * g * l
    E_current = 0.5 * m * (l * thetadot)**2 + m * g * l * (1 - jnp.cos(theta))
    delta_E = E_current - E_desired
    
    # output = -k * thetadot**2 * delta_E
    output = -k * delta_E * jnp.sign(thetadot + 1e-5)
    output = jnp.clip(output, -umax, umax)
    
    return output

if __name__ == "__main__":
    m = 1.0
    b = 0
    L = 1.0
    G = 9.8
    k = 1
    umax = 20
    useSwingUp = False
    forcingFunc = swingUpU if useSwingUp else u
    theta_initial = jnp.pi/3
    omega_initial = 0
    t_stop = 10    # seconds to simulate
    dt = 0.01   # time between each sample
    t = jnp.arange(0, t_stop, dt)
    
    useDiffrax = True
    sol = theta = omega = times = None
    if not useDiffrax:
        # in house integrator
        # Based on euler's method, which causes energy drift over time
        sol = eulers_method(theta_initial, omega_initial, t, dt, m, L, G, b, k, umax, forcingFunc)
        theta = sol[:,0]
        omega = sol[:,1]
        times = t

    else:
        # integration with diffrax, easy to change model
        term = ODETerm(pendulum_ode)
        solver = Heun()
        sol = diffeqsolve(
            term,
            solver,
            t0=0,
            t1=t_stop,
            dt0=dt,
            y0=jnp.array([theta_initial, omega_initial]),
            saveat=SaveAt(ts=t),
            args=(m, G, L, b, k, umax, forcingFunc),
        )
        theta = sol.ys[:,0]
        omega = sol.ys[:,1]
        times = sol.ts

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