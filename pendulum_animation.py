import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax.numpy as jnp

m = 1.0
b = 1.0
L = 1.0
G = 9.8
theta_initial = 40
omega_initial = 0
t_stop = 10    # seconds to simulate
dt = 0.01   # time between each sample
t = jnp.arange(0, t_stop, dt)

def u(t):
    # forcing function
    return 0

# Euler's method to solve the ODE
def derivative_function(t, state, m, l, g, b, u):
    # state is first the angular position then angular velocity
    # Using the technique of turning the 2nd order ODE into 2 first order ODES
    k = m * g / l
    theta1 = state[0]
    theta2 = state[1]
    dtheta1dt = theta2
    # dtheta2dt = (u(t) - b * theta2 - (m * g * l * jnp.sin(theta1)) * theta1) / (m * l**2)
    dtheta2dt = (u(t) - b * theta2 - k * theta1) / m
    # print(dtheta1dt, dtheta2dt)
    return [dtheta1dt, dtheta2dt]

def eulers_method(th_in, om_in, t, dt, m, L, G, b, u):
    state_initial = jnp.radians(jnp.array([th_in, om_in]))
    states_at_each_time = jnp.empty((len(t), 2))
    states_at_each_time = states_at_each_time.at[0].set(state_initial)

    for i in range(1, len(t)):
        current_state = states_at_each_time[i - 1]
        delta = jnp.array(derivative_function(t[i - 1], current_state, m, L, G, b, u))
        updated_state = current_state + delta * dt  # assuming both are arrays
        states_at_each_time = states_at_each_time.at[i].set(updated_state)

    return states_at_each_time

states_at_each_time = eulers_method(theta_initial, omega_initial, t, dt, m, L, G, b, u)
x = L * jnp.sin(states_at_each_time[:,0])
y = -L * jnp.cos(states_at_each_time[:, 0])

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '.-', lw=1, ms=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def animate(i):
    thisx = [0, x[i]]
    thisy = [0, y[i]]

    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    return line, trace, time_text

ani = animation.FuncAnimation(
    fig, animate, len(states_at_each_time), interval=dt*1000, blit=True)

plt.show()