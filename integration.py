import jax.numpy as jnp

# Euler's method to solve the ODE
def derivative_function(t, state, m, l, g, b, k, umax, F):
    # state is first the angular position then angular velocity
    # Using the technique of turning the 2nd order ODE into 2 first order ODES
    spring_constant = m * g / l
    theta1 = state[0]
    theta2 = state[1]   # omega
    dtheta1dt = theta2
    dtheta2dt = (F(t, m, g, l, b, k, umax, theta1, theta2) - b * theta2 - spring_constant * jnp.sin(theta1)) / m
    return [dtheta1dt, dtheta2dt]

def eulers_method(th_in, om_in, t, dt, m, L, G, b, k, umax, F):
    state_initial = jnp.array([th_in, om_in])
    states_at_each_time = jnp.empty((len(t), 2))
    states_at_each_time = states_at_each_time.at[0].set(state_initial)

    for i in range(1, len(t)):
        current_state = states_at_each_time[i - 1]
        delta = jnp.array(derivative_function(t[i - 1], current_state, m, L, G, b, k, umax, F))
        updated_state = current_state + delta * dt  # assuming both are arrays
        theta = updated_state[0]
        updated_state.at[0].set(theta)
        states_at_each_time = states_at_each_time.at[i].set(updated_state)

    return states_at_each_time

def pendulum_ode(t, y, args):   # for diffrax solvers
    theta, omega = y
    m, g, l, b, k, umax, F = args
    spring_constant = m * g / l
    dtheta = omega
    domega = (F(t, m, g, l, b, k, umax, theta, omega) - b * omega - spring_constant * jnp.sin(theta)) / m
    return jnp.array([dtheta, domega])

def pendulum_ode_nn(t, y, args):
    theta, omega = y
    m, g, l, b, k, umax, F = args
    spring_constant = m * g / l
    dtheta = omega
    domega =  F(jnp.array([t, m, g, l, b, k, umax, theta, omega])) - b * omega - spring_constant * jnp.sin(theta) / m
    return jnp.array([dtheta, jnp.squeeze(domega)])