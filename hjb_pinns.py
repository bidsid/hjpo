import equinox as eqx
import jax.numpy as jnp
import jax.lax as lax
import jax
import jax.random as jrandom
from jax import vmap, jit
from diffrax import diffeqsolve, ODETerm, SaveAt, Heun
import optax
from tqdm import tqdm
from pendulum_animation import simulateWithDiffraxIntegration, swingUpU
import integration
import jax.tree_util as jtu
import matplotlib.pyplot as plt


# hyperparameters
LAYERS = [3, 64, 64, 1] # number of neurons in each fully-connected layer
ACTIVATION_FUNC = jnp.tanh
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
STEPS = 300
PRINT_EVERY = 30
SEED = 5678
EPOCHS = 500
THETA_COST_COEFF = 1
THETADOT_COST_COEFF = 0.1
ACTION_COST_COEFF = 0.001   # R in the writeup

# simulation details
m = 1.0
b = 0.0
L = 1.0
G = 9.8
k = 1
umax = m * G * L * 1.5    # saturated at mgl
theta_initial = -jnp.pi/2
omega_initial = 0.5
theta_goal = jnp.pi
omega_goal = 0
t_stop = 10    # seconds to simulate
dt = 0.01   # time between each sample
t = jnp.arange(0, t_stop, dt)

key = jax.random.PRNGKey(SEED)

class PINNS(eqx.Module):
    # Will parametrize the value function to try and find the optimal one
    # Value function then gets converted by an equation into a policy
    layers: list

    def __init__(self, layer_sizes, activation_func, key):
        self.layers = []
        i = 0

        for (feat_in, feat_out) in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            keys = jax.random.split(key, len(LAYERS))
            
            self.layers.append(
                eqx.nn.Linear(feat_in, feat_out, use_bias=True, key=keys[i])
            )  # fully-connected layer
            i = i + 1
                
        self.layers.append(
            eqx.nn.Linear(layer_sizes[-2], layer_sizes[-1], use_bias=True, key=keys[len(LAYERS) - 1])
        )  # final layer

        assert all(layer is not None for layer in self.layers)

    def __call__(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = ACTIVATION_FUNC(x)
        x = self.layers[len(self.layers) - 1](x)    # applying the final layer

        y = jnp.squeeze(x)
        return y

def stage_cost_func(theta, omega, action):
    # goal_arr = jnp.array([theta_goal])
    theta_diff = jnp.mod(theta - theta_goal + jnp.pi, 2 * jnp.pi) - jnp.pi
    # omega_diff = jnp.array([omega_goal]) - omega
    omega_diff = omega_goal - omega

    return THETA_COST_COEFF * theta_diff**2 + THETADOT_COST_COEFF * omega_diff**2 + ACTION_COST_COEFF * action**2

def dynamics_eqn_func(theta, omega, u):
    spring_constant = m * G / L
    return (u(theta, omega) - b * omega - spring_constant * jnp.sin(theta)) / m

def simulation_ode(t, y, args):
    theta, omega = y
    fluff1, fluff2, fluff3, fluff4, fluff5, fluff6, u = args
    return dynamics_eqn_func(theta, omega, u)

def affine_dynamics_g_func(theta, omega):
    return 1 / m 

def pinns_loss_hamiltonian(pinn, theta, omega, u_func):
    # is also the hamiltonian which needs to be minimized
    # note that pinn is the neural net but due to how its __call__ method is defined
    # it also works like a function (can be called)
    u = jax.lax.stop_gradient(u_func(theta, omega))
    stage_cost = stage_cost_func(theta, omega, u)
    grad_V_wrt_unembedded_state = unembedded_grad_wrt_inputs(pinn, theta, omega)
    # grad_V_wrt_unembedded_state = jax.lax.stop_gradient(grad_V_wrt_unembedded_state)
    hamiltonian = stage_cost + jnp.inner(grad_V_wrt_unembedded_state, jnp.array([omega, dynamics_eqn_func(theta, omega, u_func)]))
    return 100 * hamiltonian**2

def project(u, umin, umax):
    return 0.5 * (umin + umax) * jnp.tanh(0.5 * u)

def grad_wrt_inputs(pinn, theta, omega):
    return jax.grad(pinn)(jnp.array([jnp.cos(theta), jnp.sin(theta), omega]))

def unembedded_grad_wrt_inputs(pinn, theta, omega):
    grad_wrt_state_fn = jax.grad(pinn)
    embed_eval = grad_wrt_state_fn(jnp.array([jnp.sin(theta), jnp.cos(theta), omega]))
    dgrad_dtheta = embed_eval[0] * jnp.cos(theta) - embed_eval[1] * jnp.sin(theta)
    unembed_eval = jnp.array([dgrad_dtheta, embed_eval[2]])
    return unembed_eval

def argmin_hamiltonian_analytic(pinn, dynamics_func_g, theta, omega):
    grad_V_wrt_unembedded_state = unembedded_grad_wrt_inputs(pinn, theta, omega)    # 2d array but only second elem nonzero
    return project(-1 * (1 / ACTION_COST_COEFF) * dynamics_func_g(theta, omega) * grad_V_wrt_unembedded_state[1], -umax, umax)

def argmin_hamiltonian(pinn, cost_func, dynamics_func, state, prevAction=0):
    min_learning_rate = 0.1
    num_steps = 50
    minimizer = optax.sgd(min_learning_rate)
    def plh(action):
        return pinns_loss_hamiltonian(pinn, cost_func, dynamics_func, state, action=action)
    min_state = minimizer.init(prevAction)
    action = prevAction
    for _ in num_steps:
        grads = jax.grad(plh)(action)
        updates, min_state = minimizer.update(grads, min_state)
        action = optax.apply_updates(action, updates)
        action = project(action, -umax, umax)
    return action

def ode(t, y, args):
    # args will be what dynamics_eqn_func needs
    # also tracks loss to see what the total loss across the trajectory is
    theta, omega = y
    u_func, pinn = args
    dtheta = omega
    domega = dynamics_eqn_func(theta, omega, u_func)
    return jnp.array([dtheta, domega])

def rolloutFixedValue(pinn):

    def optimal_action_func(theta, omega):
        # makes the action into a function even though it's solved numerically
        # this is to pass it into the diffeq solver
        return argmin_hamiltonian_analytic(pinn, affine_dynamics_g_func, theta, omega)
    
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
        args=(optimal_action_func, pinn)
    )
    
    theta = sol.ys[:,0]
    omega = sol.ys[:,1]
    times = sol.ts
    return theta, omega, times

def total_loss(pinn, theta_list, omega_list):
    actionFromValue = lambda t, o: argmin_hamiltonian_analytic(pinn, affine_dynamics_g_func, t, o) 
    # return jax.vmap(lambda theta, omega: pinn(jnp.sin(theta), jnp.cos(theta), omega))(theta_list, omega_list)
    return jnp.mean(jax.vmap(lambda theta, omega: pinns_loss_hamiltonian(pinn, theta, omega, actionFromValue))(theta_list, omega_list))


def train(pinn, optim, epochs, print_every):
    params = eqx.filter(pinn, eqx.is_array)
    opt_state = optim.init(params)
    loss = 0
    losses = [0] * epochs

    @eqx.filter_jit
    def make_step(pinn, opt_state, params):
        # the cost function here is the pinns loss
        theta_list, omega_list, time_list = rolloutFixedValue(pinn)
        loss = total_loss(pinn, theta_list, omega_list)
        grads = jax.grad(total_loss)(pinn, theta_list, omega_list)
        updates, opt_state = optim.update(grads, opt_state)
        pinn = eqx.apply_updates(pinn, updates)
        return pinn, opt_state, params, loss
    
    for epoch in range(EPOCHS):
        pinn, opt_state, params, loss = make_step(pinn, opt_state, params)
        if (epoch % print_every == 0) or (epoch == epochs - 1):
                print("Most recent loss: ", loss)
        losses[epoch] = loss
    
    return pinn, losses

if __name__ == "__main__":
    pinn = PINNS(LAYERS, ACTIVATION_FUNC, key)
    print(pinns_loss_hamiltonian(pinn, theta_initial, omega_initial, lambda t, o: 1))
    params = eqx.filter(pinn, eqx.is_array)
    optim = optax.adam(LEARNING_RATE)
    opt_state = optim.init(params)
    grads = jax.grad(lambda pinn, theta, omega: pinns_loss_hamiltonian(pinn, theta, omega, lambda t, o: 1))(pinn, theta_initial, omega_initial)
    updates, opt_state = optim.update(grads, opt_state)
    pinn = eqx.apply_updates(pinn, updates)
    print(pinns_loss_hamiltonian(pinn, theta_initial, omega_initial, lambda t, o: 1))
    
    
    optim = optax.adam(LEARNING_RATE)
    finished_value_func, losses = train(pinn, optim, EPOCHS, PRINT_EVERY)
    
    def policyFromValueFunc(theta, omega):
        return argmin_hamiltonian_analytic(pinn, affine_dynamics_g_func, theta, omega)
    def valuePolicyTorqueCalc(valuePolicy, *fluff_args, theta, omega):
        return valuePolicy(theta, omega)
    
    grid_theta = jnp.linspace(0, 2*jnp.pi, 16)
    grid_omega = jnp.linspace(0, 19.6, 16)
    xv, yv = jnp.meshgrid(grid_theta, grid_omega)
    grid_states = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)  # shape (10000, 2)
    u_vals = jax.vmap(lambda x: policyFromValueFunc(x[0], x[1]))(grid_states)  # shape (10000, act_dim)
    u_grid = u_vals.reshape(xv.shape)       # (100, 100) if scalar actions
    plt.contourf(xv, yv, u_grid, levels=50, cmap="coolwarm")
    plt.title("Control Action")
    plt.colorbar()

    simulateWithDiffraxIntegration(simulation_ode, valuePolicyTorqueCalc, t_stop, dt, 
                                   theta_initial, omega_initial, m, b, L, G, k, umax, policyFromValueFunc, 
                                   loss=None)
