import jax.numpy as jnp

# Euler's method to solve the ODE
def derivative_function(t, state, m, l, g, b, u):
    # state is first the angular position then angular velocity
    # Using the technique of turning the 2nd order ODE into 2 first order ODES
    k = m * g / l
    theta1 = state[0]
    theta2 = state[1]   # omega
    dtheta1dt = theta2
    # dtheta2dt = (u(t) - b * theta2 - (m * g * l * jnp.sin(theta1)) * theta1) / (m * l**2)
    dtheta2dt = (u(t) - b * theta2 - k * jnp.sin(theta1) * theta1) / m
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
        theta = updated_state[0]
        updated_state.at[0].set(theta)
        states_at_each_time = states_at_each_time.at[i].set(updated_state)

    return states_at_each_time

def pendulum_ode(t, y, args):   # for diffrax solvers
    theta, omega = y
    m, l, g, b, u = args
    k = m * g / l
    dtheta = omega
    domega = (u(t) - b * omega - k * jnp.sin(theta)) / m
    return jnp.array([dtheta, domega])