# Structure of the code inspired by https://github.com/sscivier/equinox-nn-example/blob/main/equinox_nn.ipynb

import math
import equinox as eqx
import jax.numpy as jnp
import jax.lax as lax
import jax
import jax.random as jrandom
from jax import vmap, jit
import numpy as np
import optax
from diffrax import diffeqsolve, ODETerm, SaveAt, Heun
from tqdm import tqdm
import jax.tree_util as jtu
from pendulum_animation import simulateWithDiffraxIntegration, swingUpU
import integration

# hyperparameters
LAYERS = [3, 64, 64, 1] # number of neurons in each fully-connected layer
ACTIVATION_FUNC = jnp.tanh
BATCH_SIZE = 64
LEARNING_RATE = 3e-3
STEPS = 300
PRINT_EVERY = 30
SEED = 5678
EPOCHS = 1000

# simulation details
m = 1.0
b = 0.1
L = 1.0
G = 9.8
k = 1
umax = m * G * L / 1.5    # saturated at mgl
theta_initial = -59 * jnp.pi / 180
omega_initial = 0.5
theta_goal = jnp.pi
omega_goal = 0
t_stop = 10    # seconds to simulate
dt = 0.01   # time between each sample
t = jnp.arange(0, t_stop, dt)

key = jax.random.PRNGKey(SEED)

class Policy(eqx.Module):
    layers: list

    def __init__(self, layer_sizes, activation_func, key):
        self.layers = []
        i = 0

        for (feat_in, feat_out) in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            keys = jax.random.split(key, len(LAYERS))
            
            self.layers.append(
                eqx.nn.Linear(feat_in, feat_out, use_bias=True, key=keys[i])
            )  # fully-connected layer
            self.layers.append(
                activation_func
            )
            i = i + 1
                
        self.layers.append(
            eqx.nn.Linear(layer_sizes[-2], layer_sizes[-1], use_bias=True, key=keys[len(LAYERS) - 1])
        )  # final layer

        assert all(layer is not None for layer in self.layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        
        # subtract bias then normalize then call tanh to avoid having too large of an output
        # x = x - jnp.mean(x)
        # x = x / jnp.linalg.norm(x)
        scaled = 0.5 * x
        return umax * jnp.tanh(scaled)
        return x

def dynamics_eqn_func(theta, omega, t, m, g, l, b, k, umax, F):
    spring_constant = m * g / l
    return (F(jnp.array([jnp.sin(theta), jnp.cos(theta), omega])) - b * omega - spring_constant * jnp.sin(theta)) / m

def cost_func(t, theta, omega, currentAction, learnFromAction=None):
    # currently set to learn from the swing up policy
    goal_arr = jnp.array([theta_goal])
    theta_diff = jnp.mod(theta - goal_arr + jnp.pi, 2 * jnp.pi) - jnp.pi
    # theta_diff = jnp.array([theta_goal]) - theta
    omega_diff = jnp.array([omega_goal]) - omega
    action_diff = jnp.array([learnFromAction]) - currentAction

    return  theta_diff**2 + 0.01 * omega_diff**2 + 0.001 * currentAction**2

def terminal_cost_func(theta_final, omega_final, action_final):
    theta_diff = jnp.array([theta_goal]) - theta_final
    omega_diff = jnp.array([omega_goal]) - omega_final

    return 10 * (theta_diff**2 + 0.01 * omega_diff**2 + 0.001 * action_final**2)

def teacher_cost_func(unused_t, unused_theta, unused_omega, currentAction, teacherAction):
    action_diff = jnp.array([teacherAction]) - currentAction
    return action_diff**2

def teacher_terminal_cost_func(unused_theta, unused_omega, unused_action):
    # defined this way for duck typing
    return jnp.array([0])

def augmented_ode(t, y, args):
    theta, omega, J = y
    m, g, l, b, k, umax, F, dynamics_eqn_func, cost_func = args
    dtheta = jnp.array([omega])
    domega = dynamics_eqn_func(theta, omega, t, m, g, l, b, k, umax, F)
    currentAction = domega
    swingUpAction = swingUpU(t, m, g, l, b, k, umax, theta, omega)
    dJ = cost_func(t, theta, omega, currentAction, swingUpAction)

    return jnp.squeeze(jnp.array([dtheta, domega, dJ]), -1)

# guessing this is where the augmented state integration strategy comes in
# to predict the cost of the current policy
def loss_fn(policy, dynamics_eqn_func, cost_func, terminal_cost_func):
    term = ODETerm(augmented_ode)
    solver = Heun()
    sol = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=t_stop,
        dt0=dt,
        y0=jnp.array([theta_initial, omega_initial, 0]),
        saveat=SaveAt(ts=t),
        args=(m, G, L, b, k, umax, policy, dynamics_eqn_func, cost_func)
    )
    
    theta = sol.ys[:,0]
    omega = sol.ys[:,1]
    J = sol.ys[:,2]
    times = sol.ts

    last_elem_index = jnp.size(times) - 1
    time_final = times[last_elem_index]
    theta_final = theta[last_elem_index]
    omega_final = omega[last_elem_index]
    J_final = J[last_elem_index]
    action_final = policy(jnp.array([jnp.sin(theta_final), jnp.cos(theta_final), omega_final]))

    return jnp.squeeze(J_final) + jnp.squeeze(terminal_cost_func(theta_final, omega_final, action_final))

# evaluate function not necessary because in each iteration we just compute the loss
# that the policy gets on a single simulation pretty sure
# and then immediately after update the params with the gradient of the loss

def train(policy, optim, epochs, print_every):
    params = eqx.filter(policy, eqx.is_array)
    # print("params is ", params)
    opt_state = optim.init(params)
    loss = 0

    @eqx.filter_jit
    def make_step(policy, cost_function, terminal_cost_function, opt_state, params):
        # loss, grads = eqx.filter_value_and_grad(loss_fn)(params, dynamics_eqn_func, cost_func, terminal_cost_func)
        grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=False)
        loss, grads = grad_fn(policy, dynamics_eqn_func, cost_function, terminal_cost_function)
        updates, opt_state = optim.update(grads, opt_state)
        # new_params = eqx.apply_updates(params, updates)
        # policy = eqx.combine(new_params, policy)
        policy = eqx.apply_updates(policy, updates)
        return policy, opt_state, params, loss

    with tqdm(
        bar_format="Warm start w/ teacher | Elapsed: {elapsed} | {rate_fmt}",
    ) as t:
        for epoch in range(int(epochs / 2)):
            policy, opt_state, params, loss = make_step(policy, teacher_cost_func, teacher_terminal_cost_func, opt_state, params)
            if (epoch % print_every == 0) or (epoch == epochs - 1):
                print("Most recent loss: ", loss)

    with tqdm(
        bar_format="Self-training to stabilize at top | Elapsed: {elapsed} | {rate_fmt}",
    ) as t:
        for epoch in range(int(epochs / 2)):
            policy, opt_state, params, loss = make_step(policy, cost_func, terminal_cost_func, opt_state, params)
            if (epoch % print_every == 0) or (epoch == epochs - 1):
                print("Most recent loss: ", loss)
    
    return policy


if __name__ == "__main__":
    model = Policy(layer_sizes=LAYERS, activation_func=ACTIVATION_FUNC, key=key)
    print(model)
    print(jtu.tree_structure(model))
    optim = optax.adam(LEARNING_RATE)
    finished_model = train(model, optim, EPOCHS, PRINT_EVERY)
    simulateWithDiffraxIntegration(integration.pendulum_ode_nn, t_stop, dt, theta_initial, omega_initial, m, b, L, G, k, umax, finished_model)

