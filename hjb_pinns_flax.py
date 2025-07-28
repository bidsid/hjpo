import flax.linen as nn
from typing import Sequence
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
import wandb

# hyperparameters
LAYERS = [9, 128, 128, 2] # number of neurons in each fully-connected layer
ACTIVATION_FUNC = jnp.tanh
NUM_GRID_THETA_POINTS = 100
NUM_GRID_OMEGA_POINTS = 300
NUM_GRID_POINTS = NUM_GRID_THETA_POINTS * NUM_GRID_OMEGA_POINTS
NUM_BATCHES = 10    # per epoch
BATCH_SIZE = 3000# int(NUM_GRID_POINTS / NUM_BATCHES)
LEARNING_RATE = 3e-3
STEPS = 300
PRINT_EVERY = 5
SEED = 5678
EPOCHS = 50
THETA_COST_COEFF = 1
THETADOT_COST_COEFF = 0.5
ACTION_COST_COEFF_TRAINING = 0.01   # R in the writeup
ACTION_COST_COEFF_PDE = ACTION_COST_COEFF_TRAINING
V0_COST_COEFF = 500
SLACK_WEIGHT = 1e-1
RANDOM_THETA_AMP = 0
RANDOM_OMEGA_AMP = jnp.pi

# simulation details
m = 1.0
b = 0.0
L = 1.0
G = 9.8
k = 1
umax = m * G * L /2    # saturated at mgl
NO_LIMIT = m * G * L * 10
theta_initial = jnp.pi
omega_initial = 0
theta_goal = 0
omega_goal = 0
t_stop = 10    # seconds to simulate
dt = 0.05   # time between each sample
t = jnp.arange(0, t_stop, dt)

# batch training
GRID_THETA_BOUND = jnp.pi
GRID_OMEGA_BOUND = m * G * L
EPSILON = 0
DELTA = 0
NUM_ICS = 9
initial_thetas = [2 * i * jnp.pi/NUM_ICS for i in range(NUM_ICS)]
initial_omegas = [0] * NUM_ICS
initial_conditions = jnp.stack(jnp.array([initial_thetas, jnp.zeros(NUM_ICS)]), -1)

config = {
    "hyperparameters": {
        "layers": LAYERS,
        "activation_function": ACTIVATION_FUNC,
        "batch_size": BATCH_SIZE,
        "batches_per_epoch": NUM_BATCHES,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
    },

    "sampling": {
        "num_grid_theta_points": NUM_GRID_THETA_POINTS,
        "num_grid_omega_points": NUM_GRID_OMEGA_POINTS,
        "grid_theta_bound": GRID_THETA_BOUND,
        "grid_omega_bound": GRID_OMEGA_BOUND,
        "prng_seed": SEED,
    },

    "loss": {
        "theta_cost_coefficient": THETA_COST_COEFF,
        "omega_cost_coefficient": THETADOT_COST_COEFF,
        "action_cost_coefficient": ACTION_COST_COEFF_TRAINING,
        "v0_cost_coefficient": V0_COST_COEFF,
    },

    "simulation": {
        "mass": m,
        "damping_coefficient": b,
        "length_of_pendulum": L,
        "gravitational_coefficient": G,
        "energy_swing_coefficient": k,
        "max_torque": umax,
        "theta_initial": theta_initial,
        "theta_goal": theta_goal,
        "omega_initial": omega_initial,
        "omega_goal": omega_goal,
        "simulation_duration": t_stop,
        "simulation_timestep": dt,
    },

    "other": {
        "projection": "tanh",
        "slack_weight": SLACK_WEIGHT,
    }
}


key = jax.random.PRNGKey(SEED)

class PINNS(nn.Module):
    # Will parametrize the value function to try and find the optimal one
    # Value function then gets converted by an equation into a policy
    features: Sequence[int]

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]

    # def __init__(self, layer_sizes, activation_func, key):
    #     self.layers = []
    #     i = 0

    #     for (feat_in, feat_out) in zip(layer_sizes[:-2], layer_sizes[1:-1]):
    #         keys = jax.random.split(key, len(LAYERS))
            
    #         self.layers.append(
    #             eqx.nn.Linear(feat_in, feat_out, use_bias=True, key=keys[i])
    #         )  # fully-connected layer
    #         i = i + 1
                
    #     self.layers.append(
    #         eqx.nn.Linear(layer_sizes[-2], layer_sizes[-1], use_bias=True, key=keys[len(LAYERS) - 1])
    #     )  # final layer

    #     assert all(layer is not None for layer in self.layers)

    def __call__(self, x):
        
        def inner_call(x):
            for i in range(len(self.layers) - 1):
                x = self.layers[i](x)
                x = ACTIVATION_FUNC(x)
            x = self.layers[len(self.layers) - 1](x)   # applying the final layer
            mat = jnp.outer(x, x) + SLACK_WEIGHT * jnp.eye(2)  # slack weight times I2
            return x.T @ mat @ x + jnp.inner(x, x)
        
        return inner_call(x)
    
    @staticmethod
    def feature_fn(theta, omega):
        # Core features
        sin_theta = jnp.sin(theta)
        cos_theta = jnp.cos(theta)
        
        # Useful extras
        theta_unwrapped = jnp.arctan2(sin_theta, cos_theta)
        omega_squared = omega ** 2
        sin2 = jnp.sin(2 * theta)
        cos2 = jnp.cos(2 * theta)
        phase_term = theta * omega
        bounded_omega = omega / (1 + jnp.abs(omega))
        
        return jnp.array([
            sin_theta,
            cos_theta,
            omega,
            omega_squared,
            sin2,
            cos2,
            phase_term,
            bounded_omega,
            theta_unwrapped
        ])


def stage_cost_func(theta, omega, action, stepNum):
    theta_diff = theta_goal - theta # jnp.mod(theta - theta_goal + jnp.pi, 2 * jnp.pi) - jnp.pi  # look into
    omega_diff = omega_goal - omega
    theta_part = THETA_COST_COEFF * jnp.sin(theta_diff)**2 + THETA_COST_COEFF * (jnp.cos(theta_diff) - 1)**2
    # energy = 0.5 * m * (L * omega)**2 + m * G * L * (1 - jnp.cos(theta))
    # energy_diff = 2 * m * G * L - energy
    # energy_part = (1 - curriculum_progress(stepNum)) * energy_diff**2
    return theta_part + THETADOT_COST_COEFF * omega_diff**2 + ACTION_COST_COEFF_TRAINING * action**2

# Not currently in use
def teacher_stage_cost_func(theta, omega, actionFunc):
    teacher_action = localSwingUp(theta, omega)
    return (teacher_action - actionFunc(theta, omega))**2

def dynamics_eqn_func(theta, omega, u):
    return affine_dynamics_f_func(theta, omega) + affine_dynamics_g_func(theta, omega) * u(theta, omega)

def simulation_ode(t, y, args):
    theta, omega = y
    fluff1, fluff2, fluff3, fluff4, fluff5, fluff6, u = args
    return dynamics_eqn_func(theta, omega, u)

def affine_dynamics_f_func(theta, omega):
    return jnp.array([omega, (b * omega + G * L * jnp.sin(theta)) / m])

def affine_dynamics_g_func(theta, omega):
    return jnp.array([0, 1 / (m)])


def pinns_loss_hamiltonian(pinn, params, theta, omega, u_func, stepNum):
    # is also the hamiltonian which needs to be minimized
    u = u_func(theta, omega) # where I had the stop gradient before
    grad_V_wrt_unembedded_state = unembedded_grad_wrt_inputs(pinn, params, theta, omega)
    f_part = jnp.inner(grad_V_wrt_unembedded_state, affine_dynamics_f_func(theta, omega))
    g_inner_product = jnp.inner(affine_dynamics_g_func(theta, omega), grad_V_wrt_unembedded_state * u_func(theta, omega))
    g_part = g_inner_product#  * (1 / ACTION_COST_COEFF_PDE) * g_inner_product
    hamiltonian = f_part + g_part + stage_cost_func(theta, omega, u_func(theta, omega), stepNum)
    # hamiltonian = jnp.inner(grad_V_wrt_unembedded_state, affine_dynamics_f_func(theta, omega)) - 0.5 * (1 / action)
    v0_loss = V0_COST_COEFF * (pinn.apply(params, PINNS.feature_fn(0, 0)) - 0)**2
    return hamiltonian**2 + v0_loss

# Used to enforce the torque limit
def project(u, stepNum):
    # return 0.5 * (umin + umax) * jnp.tanh(0.5 * u)    # did not work
    return umax * (jnp.tanh(u) + 1) - umax # almost worked
    # return u
    # p = curriculum_progress(stepNum)
    # appliedBound = NO_LIMIT * (1 - p) + umax * p
    # return jnp.clip(u, -umax, umax)    # almost worked
    # tanh_part = 0.5 * (umax - umin) * (jnp.tanh(u) + 1) + umin
    # clip_part = jnp.clip(u, umin, umax)
    # return 0.5 * (tanh_part + clip_part)  # combining didn't work, they fought each other
    def soft_saturate(x, limit):
        return limit * x / (1 + jnp.abs(x))
    # return umax * jnp.tanh(soft_saturate(u, umax)) # did not work
    return umax * jnp.tanh(soft_saturate(u, umax))

# outputs 2 element array [dV/dtheta, dV/domega]
# equivalent to the costate in Kaiyuan's code
def unembedded_grad_wrt_inputs(pinn, params, theta, omega):
    valueFunc = lambda theta, omega: pinn.apply(params, PINNS.feature_fn(theta, omega))
    gradV =  jax.grad(valueFunc, argnums=(0, 1))(theta, omega)
    return jnp.array([gradV[0], gradV[1]])
    # autodiff does chain rule by itself so no need to do manual unembedding
    # 0, 1 argnums to make sure it computes partials for both theta and omega

    # valueFunc = lambda sintheta, costheta, omega: pinn.apply(params, jnp.array([sintheta, costheta, omega]))
    # gradV = jax.grad(valueFunc, argnums=(0, 1, 2))
    # ff = PINNS.feature_fn(theta, omega)
    # embed_eval = gradV(ff[0], ff[1], ff[2])
    # dgrad_dtheta = embed_eval[0] * jnp.cos(theta) - embed_eval[1] * jnp.sin(theta)
    # unembed_eval = jnp.array([dgrad_dtheta, embed_eval[2]])
    # return unembed_eval

# Gives the policy as a function of the value function
def argmin_hamiltonian_analytic(pinn, params, dynamics_func_g, theta, omega, stepNum):
    # theta = theta - (5*jnp.pi/6 + 0.1)
    grad_V_wrt_unembedded_state = unembedded_grad_wrt_inputs(pinn, params, theta, omega)    # 2d array but only second elem nonzero
    # return -1 * (1 / ACTION_COST_COEFF_PDE) * jnp.inner(dynamics_func_g(theta, omega), grad_V_wrt_unembedded_state)
    return project(-1 * (1 / ACTION_COST_COEFF_PDE) * jnp.inner(dynamics_func_g(theta, omega), grad_V_wrt_unembedded_state), stepNum)  # originally negative

def ode(t, y, args):
    # args will be what dynamics_eqn_func needs
    # also tracks loss to see what the total loss across the trajectory is
    theta, omega = y
    u_func, pinn = args
    dtheta = omega
    domega = dynamics_eqn_func(theta, omega, u_func)
    return jnp.array([dtheta, domega])

# Not used
def rolloutFixedValue(pinn, batch_id):

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
        y0=jnp.array(initial_conditions[batch_id]),
        saveat=SaveAt(ts=t),
        args=(optimal_action_func, pinn)
    )
    
    theta = sol.ys[:,0]
    omega = sol.ys[:,1]
    times = sol.ts
    return theta, omega, times

def total_loss(params, input_states, stepNum):
    pinn = PINNS(features=LAYERS)
    actionFromValue = lambda t, o: argmin_hamiltonian_analytic(pinn, params, affine_dynamics_g_func, t, o, stepNum)

    original_hamiltonian_loss = jnp.mean(jax.vmap(lambda input_state: pinns_loss_hamiltonian(pinn, params, input_state[0], input_state[1], actionFromValue, stepNum))(input_states))
    # bfv = jax.nn.relu
    # bound_violation_loss = jnp.mean(jax.vmap(lambda input_state: bfv(actionFromValue(input_state[0], input_state[1]) - umax) + bfv(-umax - actionFromValue(input_state[0], input_state[1])))(input_states))
    return  original_hamiltonian_loss

# Not used
def teacher_total_loss(pinn, theta_list, omega_list):
    actionFromValue = lambda t, o: argmin_hamiltonian_analytic(pinn, affine_dynamics_g_func, t, o)
    
    # return jax.vmap(lambda theta, omega: pinn(jnp.sin(theta), jnp.cos(theta), omega))(theta_list, omega_list)
    return jnp.mean(jax.vmap(lambda theta, omega: teacher_stage_cost_func(theta, omega, actionFromValue))(theta_list, omega_list))

# Not used
def localSwingUp(theta, thetadot):
        # Energy at bottom is 0, energy at top = goal = 2mgl
        E_desired = 2 * m * G * L
        E_current = 0.5 * m * (L * thetadot)**2 + m * G * L * (1 + jnp.cos(theta))
        delta_E = E_current - E_desired
        
        output = -k * delta_E * jnp.sign(thetadot + 1e-5)
        output = jnp.clip(output, -umax, umax)
    
        return output

# Not used, tried to see if this would help learning swing up with unsaturated
# returns 1 because every change I made that relied on this function
# has no effect when the function returns 1
def curriculum_progress(step):
    return 1
    return jnp.clip(6 * step / (EPOCHS * NUM_BATCHES), 0, 1)

def train(params, optim, key, run=None):
    opt_state = optim.init(params)
    losses = [0] * EPOCHS
    NUM_RANDOM_DIM = 10
    new_key, subkey = jax.random.split(key)    

    
    @jax.jit
    def make_step(opt_state, params, random_states, stepNum):

        loss, grads = jax.value_and_grad(total_loss)(params, random_states, stepNum)
        updates, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return opt_state, params, loss

    
    def sample_online(key):
        k1, k2 = jax.random.split(key)
        # uniform randomly samples values from minval to maxval with the given shape
        ths = jax.random.uniform(k1, (BATCH_SIZE, 1), minval=-GRID_THETA_BOUND, maxval=GRID_THETA_BOUND)
        dths = jax.random.uniform(k2, (BATCH_SIZE, 1), minval=-GRID_OMEGA_BOUND, maxval=GRID_OMEGA_BOUND)
        return jnp.squeeze(jnp.stack([ths, dths], axis=-1))
    

    print("Learning from random sampling")
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for batch in range(NUM_BATCHES):
            new_key, subkey = jax.random.split(new_key)
            random_sampled_states = sample_online(subkey)
            opt_state, params, loss = make_step(opt_state, params, random_sampled_states, epoch * NUM_BATCHES + batch)
            if not run == None:
                run.log({"loss": loss})
            epoch_loss += loss

            if (batch % PRINT_EVERY == 0) or (batch == NUM_BATCHES - 1):
                    print(f"Epoch {epoch}, batch {batch}, loss: {loss}")
            if not jnp.isfinite(loss):
                print("Bad loss at batch", batch)
                break

        losses[epoch] = epoch_loss / NUM_BATCHES
    
    return params, losses

if __name__ == "__main__":
    new_key, subkey = jax.random.split(key)
    del key
    pinn = PINNS(features=LAYERS)
    init_input = jax.random.uniform(subkey, LAYERS[0])
    del subkey
    new_key2, subkey2 = jax.random.split(new_key)
    del new_key
    params = pinn.init(subkey2, init_input)
    del subkey2

    V_fn = lambda state: pinn.apply(params, state)
    grad_fn = jax.grad(V_fn)
    value_val = pinn.apply(params, PINNS.feature_fn(0, 0))
    jax.debug.print('V before training at (0, 0) = {})', value_val)
    grad_val = grad_fn(PINNS.feature_fn(0, 0))
    jax.debug.print("∇V wrt input before training at (0, 0) = {}", grad_val)
    jax.debug.print("isfinite? {}", jnp.all(jnp.isfinite(grad_val)))


    
    optim = optax.adam(LEARNING_RATE)
    new_key3, subkey3 = jax.random.split(new_key2)
    del new_key2
    wandb.login()
    run_name = "New features framework tanh projection saturated"
    run = None
    run = wandb.init(
        entity="k5wang-main-university-of-southern-california",
        project="sid-version",
        name=run_name,
        config=config,
        notes="From VSCode",
        tags=["test", "tanh", "soft-saturate"]
    )
    params, losses = train(params, optim, subkey3, run)
    print("V at (0, 0): {}", pinn.apply(params, PINNS.feature_fn(0, 0)))

    def policyFromValueFunc(theta, omega):
        return argmin_hamiltonian_analytic(pinn, params, affine_dynamics_g_func, theta, omega, EPOCHS * NUM_BATCHES)
    def valuePolicyTorqueCalc(valuePolicy, *fluff_args, theta, omega):
        # This might be returning some wrong values in extreme cases
        return valuePolicy(theta, omega)
    
    nxs, nys = 100,100
    grid_theta = jnp.linspace(-jnp.pi, jnp.pi, nxs)
    grid_omega = jnp.linspace(-m*G*L, m*G*L, nys)
    xv, yv = jnp.meshgrid(grid_theta, grid_omega)
    grid_states = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)  # shape (10000, 2)

    fig, axs = plt.subplots(2, 2, figsize=(20, 10))

    u_vals = jax.vmap(lambda x: policyFromValueFunc(x[0], x[1]))(grid_states)  # shape (10000, act_dim)
    u_grid = u_vals.reshape(xv.shape)       # (100, 100) if scalar actions
    contour1 = axs[0, 0].contourf(xv, yv, u_grid, levels=50, cmap="coolwarm")
    axs[0, 0].set_title("Control Action from Value Function")

    plt.colorbar(contour1, ax=axs[0, 0])
    swingUpUVals = jax.vmap(lambda x: localSwingUp(x[0], x[1]))(grid_states)
    su_grid = swingUpUVals.reshape(xv.shape)
    contour2 = axs[0, 1].contourf(xv, yv, su_grid, levels=50, cmap="coolwarm")
    axs[0, 1].set_title("Control Action From Swing Up Policy")

    plt.colorbar(contour2, ax=axs[0, 1])
    valueNetVals = jax.vmap(lambda x: pinn.apply(params, PINNS.feature_fn(x[0], x[1])))(grid_states)
    vn_grid = valueNetVals.reshape(xv.shape)
    contour3 = axs[1, 0].contourf(xv, yv, vn_grid, levels=50, cmap="coolwarm")
    axs[1, 0].set_title("Value network outputs on variety of states")

    plt.colorbar(contour3, ax=axs[1, 0])
    hjbLossVals = jax.vmap(lambda x: pinns_loss_hamiltonian(pinn, params, x[0], x[1], policyFromValueFunc, stepNum=(EPOCHS * NUM_BATCHES)))(grid_states)
    # hjbLossVals = jnp.log(hjbLossVals)
    hjbLoss_grid = hjbLossVals.reshape(xv.shape)
    contour4 = axs[1, 1].contourf(xv, yv, hjbLoss_grid, levels=50, cmap="coolwarm")
    axs[1, 1].contour(xv, yv, hjbLoss_grid, levels=[-1, 0], colors="black", linewidths=2)
    axs[1, 1].set_title("HJB residual on variety of states")
    plt.colorbar(contour4, ax=axs[1, 1])
    
    simulateWithDiffraxIntegration(simulation_ode, valuePolicyTorqueCalc, t_stop, dt, 
                                initial_thetas, initial_omegas, m, b, L, G, k, umax, policyFromValueFunc, 
                                run_name=run_name)
    
    if run != None:
        run.log({"simulations": wandb.Video(f"E:\\usc sure\\hjpo\\{run_name}.mp4", fps=4, format="mp4")})

    if run != None:
        wandb.log({"value, action, residual grids": wandb.Image(fig)})
        wandb.finish()