# Code inspired by the guide at https://matplotlib.org/stable/gallery/animation/double_pendulum.html

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax.numpy as jnp
from integration import derivative_function, eulers_method, pendulum_ode
from diffrax import diffeqsolve, ODETerm, SaveAt, Heun

def u(t, m, g, l, b, k, umax, theta, thetadot):
    # forcing function
    return 0

def swingUpU(t, m, g, l, b, k, umax, theta, thetadot):
    E_desired = 2 * m * g * l
    E_current = 0.5 * m * (l * thetadot)**2 + m * g * l * (1 - jnp.cos(theta))
    delta_E = E_current - E_desired
    
    # output = -k * thetadot**2 * delta_E
    output = -k * delta_E * jnp.sign(thetadot + 1e-5)
    output = jnp.clip(output, -umax, umax)
    # def sigmoid(x,mi, mx): return mi + (mx-mi)*(lambda t: (1+200**(-t+0.5))**(-1) )( (x-mi)/(mx-mi) )
    # output = sigmoid(output, -umax, umax)
    # output = umax * jnp.tanh(output)
    
    return output

def generic_torque_calculator_fn(F, t, m, g, l, b, k, umax, theta, thetadot):
    # F is a forcing function that outputs a torque
    return F(t, m, g, l, b, k, umax, theta, thetadot)

def simulateWithDiffraxIntegration(ode, torque_calc, t_stop, dt, theta_initial, omega_initial, m, b, L, G, k, 
                                   umax, policy, loss=None):
    t = jnp.arange(0, t_stop, dt)

    term = ODETerm(ode)
    solver = Heun()
    sol = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=t_stop,
        dt0=dt,
        y0=jnp.array([theta_initial, omega_initial]),
        saveat=SaveAt(ts=t),
        args=(m, G, L, b, k, umax, policy),
    )
    theta = sol.ys[:,0]
    theta = (theta + jnp.pi) % (2 * jnp.pi) - jnp.pi
    omega = sol.ys[:,1]
    times = sol.ts
    energies = 0.5 * m * (L * omega)**2 + m * G * L * (1 - jnp.cos(theta - jnp.pi))
    torques = jnp.array([torque_calc(policy, times[i], m, G, L, b, k, umax, theta=theta[i], omega=omega[i]) for i in range(len(times))])

    x = L * jnp.sin(theta - jnp.pi)
    y = -L * jnp.cos(theta - jnp.pi)

    # fig, axs = plt.figure(figsize=(5, 4))
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f"Pendulum data (starting angle {theta_initial} rads, starting angular vel. {omega_initial} rad/s, umax {umax} Nm)")
    ax = axs[0, 0]
    ax.set_title("Simulation")
    ax.set_ylim(-L, 1.)
    ax.set_xlim(-L, L)
    # ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([], [], 'o-', lw=2)
    trace, = ax.plot([], [], '.-', lw=1, ms=2)
    time_template = 'time = %.1fs, energy = %.1fJ, torque = %.1fNm'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    energy_template = 'Energy = %.1fJ'
    torque_template = '\nTorque = %.1fNm'

    axs[0, 1].set_title("Phase diagram")
    axs[0, 1].plot(theta, omega)
    axs[0, 1].set_xlabel("Angle (radians)")
    axs[0, 1].set_ylabel("Angular velocity (rad/s)")
    axs[1, 0].set_title("Time series")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].set_ylabel("Value (Radians for angular and Nm for Torque/Energy)")
    axs[1, 0].plot(times, theta, color="orange", label="Theta")
    axs[1, 0].plot(times, omega, color="red", label="Omega")
    axs[1, 0].plot(times, energies, color="green", label="Energy")
    axs[1, 0].plot(times, torques, color="blue", label="Torque")
    axs[1, 0].legend()
    axs[1, 1].set_title("Loss vs iteration")
    if loss != None:
        axs[1, 1].set_yscale('log')
        axs[1, 1].plot(loss, color="gray", label="loss")
        # axs[1, 1].plot(jnp.arange(len(loss[0]), len(loss[0]) + len(loss[1]), 1), loss[1], label="Own loss")
        axs[1, 1].legend()
        axs[1, 1].set_xlabel("Epoch number")
        axs[1, 1].set_ylabel("Average Loss amount per epoch")
    
    def animate(i, thetas, omegas, times, energies, torques, x, y):
        theta = thetas[i]
        thetadot = omegas[i]
        thisx = [0, x[i]]
        thisy = [0, y[i]]
        thist = times[i]
        thisenergy = energies[i]
        thistorque = torques[i]

        line.set_data(thisx, thisy)
        # energy = 0.5 * m * (L * thetadot)**2 + m * G * L * (1 - jnp.cos(theta))
        # torque = torque_calc(policy, thist, m, G, L, b, k, umax, theta, thetadot)
        time_text.set_text(time_template % (i*dt, thisenergy, thistorque))

        return line, trace, time_text

    ani = animation.FuncAnimation(
        fig, animate, len(theta), fargs=(theta, omega, times, energies, torques, x, y, ), interval=t_stop, blit=True)

    plt.show()

if __name__ == "__main__":
    m = 1.0
    b = 0
    L = 1.0
    G = 9.8
    k = 1
    umax = m * G * L / 1.5
    useSwingUp = True
    forcingFunc = swingUpU if useSwingUp else u
    theta_initial = 0
    omega_initial = 1
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