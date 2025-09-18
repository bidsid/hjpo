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
from pendulum_animation import simulateWithDiffraxIntegration, saveAnimatedFig
import sampling
import integration
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import wandb
from matplotlib.colors import LogNorm
from flax.core import freeze, unfreeze


jax.config.update("jax_enable_x64", True)
x = jax.random.uniform(jax.random.key(0), (1,), dtype=jnp.float64)

# hyperparameters
HJB_NN_LAYERS = [3, 128, 128, 1] # number of neurons in each fully-connected layer
PO_NN_LAYERS = [3, 128, 128, 1]
SKIP_CONNECTS_ENABLED = False
ACTIVATION_FUNC = nn.swish
NUM_GRID_THETA_POINTS = 100
NUM_GRID_OMEGA_POINTS = 300
NUM_GRID_POINTS = NUM_GRID_THETA_POINTS * NUM_GRID_OMEGA_POINTS
NUM_BATCHES = 10    # per epoch
BATCH_SIZE = 256# int(NUM_GRID_POINTS / NUM_BATCHES)
PO_IC_BATCH_SIZE = 10
LEARNING_RATE = 3e-3
STEPS = 300
PRINT_EVERY = 5
SEED = 5678
EPOCHS = 50
THETA_COST_COEFF = 1
THETADOT_COST_COEFF = 0.5
ACTION_COST_COEFF_TRAINING = 0.01   # R in the writeup
ACTION_COST_COEFF_PDE = ACTION_COST_COEFF_TRAINING
V0_COST_COEFF = 200
TERMINAL_COST_COEFF = 10
UPDATE_RESIDS_EVERY = 5
P_UNIFORM_SAMPLING = 0.2
SLACK_WEIGHT = 1e-1
RANDOM_THETA_AMP = 0
RANDOM_OMEGA_AMP = jnp.pi
huber_delta = 0.5
HJPO_TRADEOFF = 0.5
DISCOUNTING = 0.1

# simulation details
m = 1.0
b = 0.0
L = 1.0
G = 9.8
k = 1
umax = m * G * L / 2    # saturated at mgl
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
GRID_OMEGA_BOUND = 2 * m * G * L
EPSILON = 0
DELTA = 0
NUM_ICS = 9
initial_thetas = [2 * i * jnp.pi/NUM_ICS for i in range(NUM_ICS)]
initial_omegas = [0] * NUM_ICS
initial_conditions = jnp.stack(jnp.array([initial_thetas, jnp.zeros(NUM_ICS)]), -1)

config = {
    "hyperparameters": {
        "hjb_layers": HJB_NN_LAYERS,
        "po_layers": PO_NN_LAYERS,
        "skip_connects": SKIP_CONNECTS_ENABLED,
        "activation_function": ACTIVATION_FUNC,
        "batch_size": BATCH_SIZE,
        "batches_per_epoch": NUM_BATCHES,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "hjpo_tradeoff_factor": HJPO_TRADEOFF,
        "adaptive_recalc_resids_every": UPDATE_RESIDS_EVERY,
        "uniform_sampling_prob": P_UNIFORM_SAMPLING,
        "discounting": DISCOUNTING,
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
        "terminal_cost_coefficient": TERMINAL_COST_COEFF,
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
        "precision_level": x.dtype,
        "huber_loss_param": huber_delta,
    }
}

key = jax.random.PRNGKey(SEED)

class MLP(nn.Module):
    features: Sequence[int]

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]
    
    def __call__(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = ACTIVATION_FUNC(x)
        x = self.layers[len(self.layers) - 1](x)
        return jnp.squeeze(x)
    
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

        return jnp.array([sin_theta, cos_theta, omega])
        return jnp.array([sin_theta, cos_theta, omega, omega_squared, bounded_omega, sin2, cos2])
    

class PINNS(MLP):
    @nn.compact
    def __call__(self, x):
        
        def inner_call(x):
            temp = None
            for i in range(len(self.layers) - 1):
                if i % 2 == 0 and temp is not None:
                    if SKIP_CONNECTS_ENABLED:
                        if x.shape[-1] == temp.shape[-1]:
                            x = x + temp
                        else:
                            temp_proj = nn.Dense(x.shape[-1], name=f"skip_proj_{i}")(temp)
                            x = x + temp_proj

                x = self.layers[i](x)
                x = ACTIVATION_FUNC(x)

                if i % 2 == 0:
                    temp = x

            x = self.layers[len(self.layers) - 1](x)   # applying the final layer
            x = nn.softplus(x)  # Ensure V(x) is non-negative
            return x.squeeze(-1)
            # mat = jnp.outer(x, x) + SLACK_WEIGHT * jnp.eye(2)  # slack weight times I2
            # return x.T @ mat @ x + jnp.inner(x, x)
        
        return inner_call(x)
    
def stage_cost_func(theta, omega, action, stepNum):
    theta_diff = theta_goal - theta # jnp.mod(theta - theta_goal + jnp.pi, 2 * jnp.pi) - jnp.pi  # look into
    omega_diff = omega_goal - omega
    theta_part = THETA_COST_COEFF * jnp.sin(theta_diff)**2 + THETA_COST_COEFF * (jnp.cos(theta_diff) - 1)**2
    return theta_part + THETADOT_COST_COEFF * jnp.abs(omega_diff) + ACTION_COST_COEFF_TRAINING * action**2

def terminal_stage_cost_func(finalTheta, finalOmega, finalAction):
    return TERMINAL_COST_COEFF * stage_cost_func(finalTheta, finalOmega, finalAction, 100000)

# Not used
def teacher_stage_cost_func(theta, omega, teacherAction, action, stepNum):
    return stage_cost_func(theta, omega, action, stepNum) + (teacherAction - action)**2

def supervised_cost_func(theta, omega, teacherAction, action, stepNum):
    return (teacherAction - action)**2

def dynamics_eqn_func(theta, omega, u_func):
    return affine_dynamics_f_func(theta, omega) + affine_dynamics_g_func(theta, omega) * u_func(theta, omega)

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
    g_inner_product = jnp.inner(affine_dynamics_g_func(theta, omega), grad_V_wrt_unembedded_state * u)
    g_part = g_inner_product#  * (1 / ACTION_COST_COEFF_PDE) * g_inner_product
    hamiltonian = DISCOUNTING * (f_part + g_part) + stage_cost_func(theta, omega, u, stepNum)

    w_theta = (jnp.abs(theta) / jnp.pi)**2
    w_omega = (jnp.abs(omega) / (2 * m * G * L))**2
    weights = w_theta + w_omega

    # v0_residual = pinn.apply(params, PINNS.feature_fn(0, 0))
    # return optax.huber_loss(predictions=hamiltonian, targets=0, delta=huber_delta) + V0_COST_COEFF * optax.huber_loss(predictions=v0_residual, targets=0, delta=huber_delta)
    

    v0_loss = V0_COST_COEFF * (pinn.apply(params, PINNS.feature_fn(0, 0)) - 0)**2
    return hamiltonian**2
    return weights * hamiltonian**2 + v0_loss

# Used to enforce the torque limit
def project(u, stepNum):
    return u

# outputs 2 element array [dV/dtheta, dV/domega]
# equivalent to the costate in Kaiyuan's code
def unembedded_grad_wrt_inputs(pinn, params, theta, omega):
    valueFunc = lambda theta, omega: pinn.apply(params, PINNS.feature_fn(theta, omega))
    gradV =  jax.grad(valueFunc, argnums=(0, 1))(theta, omega)
    return jnp.array([gradV[0], gradV[1]])
    # autodiff does chain rule by itself so no need to do manual unembedding
    # 0, 1 argnums to make sure it computes partials for both theta and omega

# Gives the policy as a function of the value function
def argmin_hamiltonian_analytic(pinn, params, dynamics_func_g, theta, omega, stepNum):
    grad_V_wrt_unembedded_state = unembedded_grad_wrt_inputs(pinn, params, theta, omega)    # 2d array but only second elem nonzero
    return project(-0.5 * (1 / ACTION_COST_COEFF_PDE) * jnp.inner(dynamics_func_g(theta, omega), grad_V_wrt_unembedded_state), stepNum)  # originally negative

def augmented_ode(t, y, args):
    # args will be what dynamics_eqn_func needs
    # also tracks loss to see what the total loss across the trajectory is
    theta, omega, J = y
    policy_action_func = args
    dtheta = omega
    domega = dynamics_eqn_func(theta, omega, policy_action_func)[1]
    dJ = stage_cost_func(theta, omega, policy_action_func(theta, omega), stepNum=10000)
    return jnp.array([dtheta, domega, dJ])

def augmented_rollout_with_policy(policy_func, rollout_ic, stepNum):
    term = ODETerm(augmented_ode)
    solver = Heun()
    sol = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=t_stop,
        dt0=dt,
        y0=jnp.array([rollout_ic[0], rollout_ic[1], 0]),
        saveat=SaveAt(ts=t),
        args=(policy_func)
    )

    thetas = sol.ys[:,0]
    omegas = sol.ys[:,1]
    js = sol.ys[:,2]
    times = sol.ts

    return thetas, omegas, js, times

def rollout_terminal_cost(lastIndex, thetas, omegas, action_func):
    thetaFinal = thetas[lastIndex]
    omegaFinal = omegas[lastIndex]
    actionFinal = action_func(thetaFinal, omegaFinal)
    return terminal_stage_cost_func(thetaFinal, omegaFinal, actionFinal)

def po_loss_rollout(policy_action_func, rollout_ic, stepNum):
    
    thetas, omegas, js, times = augmented_rollout_with_policy(policy_action_func, rollout_ic, stepNum)
    
    lastIndex = len(times) - 1
    jFinal = js[lastIndex]
    #v0_manual_cost = V0_COST_COEFF * (policy_action_func(float(0), float(0)) - 0)**2
    return jFinal + rollout_terminal_cost(lastIndex, thetas, omegas, policy_action_func)

def hjb_plus_terminal_loss(pinn, params, action_from_value_func, input_state, stepNum):
    hjb_loss = pinns_loss_hamiltonian(pinn, params, input_state[0], input_state[1], action_from_value_func, stepNum)

    # thetas, omegas, _, times = augmented_rollout_with_policy(action_from_value_func, input_state, stepNum)
    # lastIndex = len(times) - 1
    # terminal_loss = rollout_terminal_cost(lastIndex, thetas, omegas, action_from_value_func)
    
    return hjb_loss#  + terminal_loss

def hjb_batched_loss(params, input_states, stepNum):
    pinn = PINNS(features=HJB_NN_LAYERS)
    actionFromValue = lambda t, o: argmin_hamiltonian_analytic(pinn, params, affine_dynamics_g_func, t, o, stepNum)

    mean_hjb_loss = jnp.mean(jax.vmap(lambda input_state: hjb_plus_terminal_loss(pinn, params, actionFromValue, input_state, stepNum))(input_states))
    # bfv = jax.nn.relu
    # bound_violation_loss = jnp.mean(jax.vmap(lambda input_state: bfv(actionFromValue(input_state[0], input_state[1]) - umax) + bfv(-umax - actionFromValue(input_state[0], input_state[1])))(input_states))
    return  mean_hjb_loss

def supervised_po_batched_loss(policy_params, pinn_params, input_states, stepNum):
    pinn = PINNS(features=HJB_NN_LAYERS)
    policy = MLP(features=PO_NN_LAYERS)
    valueActionFunc = lambda t, o: argmin_hamiltonian_analytic(pinn, pinn_params, affine_dynamics_g_func, t, o, stepNum)
    policyActionFunc = lambda t, o: policy.apply(policy_params, MLP.feature_fn(t, o))
    
    # runs the rollout and calculates the loss for each of the initial conditions specified at the top
    # Then takes the mean of that
    # so the batch is the initial conditions
    return jnp.mean(jax.vmap(lambda state: supervised_cost_func(state[0], state[1], valueActionFunc(state[0], state[1]), policyActionFunc(state[0], state[1]), stepNum))(input_states))

# Effectively sets the teacher action function in the above version to the same thing
# as the policy network's action function. Since the teacher_stage_cost_function
# only differs from stage_cost_function by the addition of a term proportional to
# teacher_action - action, if teacher_action = action then it's equivalent to just
# the stage cost and so there is no learning from a teacher.
def po_batched_loss(policy_params, batch_ics, stepNum):
    policy = MLP(features=PO_NN_LAYERS)
    policyActionFunc = lambda t, o: policy.apply(policy_params, MLP.feature_fn(t, o))
    
    return jnp.mean(jax.vmap(lambda batch_ic: po_loss_rollout(policyActionFunc, batch_ic, stepNum))(batch_ics))

def hjpo_batched_loss(params, po_ics, hjb_input_states, stepNum):
    pinn = PINNS(features=HJB_NN_LAYERS)
    actionFromValue = lambda t, o: argmin_hamiltonian_analytic(pinn, params, affine_dynamics_g_func, t, o, stepNum)

    total_hjb_loss = jnp.mean(jax.vmap(lambda input_state: pinns_loss_hamiltonian(pinn, params, input_state[0], input_state[1], actionFromValue, stepNum))(hjb_input_states))
    total_po_loss = jnp.mean(jax.vmap(lambda batch_ic: po_loss_rollout(actionFromValue, batch_ic, stepNum))(po_ics))


    # rv = 0
    # rv = jax.lax.cond(stepNum % 2 == 0, lambda: (1 - HJPO_TRADEOFF) * total_hjb_loss, lambda: HJPO_TRADEOFF * total_po_loss)
    
    # return rv
    return HJPO_TRADEOFF * total_po_loss + (1 - HJPO_TRADEOFF) * total_hjb_loss

def sample_ics(key, numICs, stepNum):
        k1, k2 = jax.random.split(key)
        # uniform randomly samples values from minval to maxval with the given shape
        # prop = curriculum_progress(stepNum) # between 0 and 1, increases
        prop = 1
        ths = jax.random.uniform(k1, (numICs, 1), minval=-GRID_THETA_BOUND * prop, maxval=GRID_THETA_BOUND * prop)
        dths = jax.random.uniform(k2, (numICs, 1), minval=-GRID_OMEGA_BOUND * prop, maxval=GRID_OMEGA_BOUND * prop)
        return jnp.squeeze(jnp.stack([ths, dths], axis=-1))
    
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
    return jnp.clip(6 * (step + 1) / (EPOCHS), 0, 1)

theta_range = (-GRID_THETA_BOUND, GRID_THETA_BOUND)
omega_range = (-GRID_OMEGA_BOUND, GRID_OMEGA_BOUND)
states, theta_lin, omega_lin, theta_grid, omega_grid = sampling.create_space_grid(theta_range, omega_range)

def train_hjb(params, optim, key, run=None):
    opt_state = optim.init(params)
    losses = [0] * EPOCHS
    NUM_RANDOM_DIM = 10
    new_key, subkey = jax.random.split(key)

    pinn = PINNS(features=HJB_NN_LAYERS)
    valuePolicy = lambda theta, omega: argmin_hamiltonian_analytic(pinn, params, affine_dynamics_g_func, theta, omega, stepNum=100000)
    residual_func = lambda state: pinns_loss_hamiltonian(pinn, params, state[0], state[1], valuePolicy, 100000)
    residuals = None  

    @jax.jit
    def make_step(opt_state, params, random_states, stepNum):

        loss, grads = jax.value_and_grad(hjb_batched_loss)(params, random_states, stepNum)
        updates, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return opt_state, params, loss
    

    print("HJB: Learning from random sampling")
    for epoch in range(EPOCHS):
        epoch_loss = 0
        if epoch % UPDATE_RESIDS_EVERY == 0:
            residuals = jax.vmap(residual_func)(states)
        for batch in range(NUM_BATCHES):
            new_key, subkey = jax.random.split(new_key)
            random_sampled_states = sampling.mixed_sample(states, residuals, theta_range, omega_range, theta_grid.shape, BATCH_SIZE, P_UNIFORM_SAMPLING, subkey)
            # random_sampled_states = sample_online(subkey, stepNum=epoch)
            opt_state, params, loss = make_step(opt_state, params, random_sampled_states, stepNum=epoch)
            if not run == None:
                run.log({"hjb_loss": loss})
            epoch_loss += loss

            if (batch % PRINT_EVERY == 0) or (batch == NUM_BATCHES - 1):
                    print(f"Epoch {epoch}, batch {batch}, loss: {loss}")
            if not jnp.isfinite(loss):
                print("Bad loss at batch", batch)
                break

        losses[epoch] = epoch_loss / NUM_BATCHES
    
    return params, losses

def train_po(params, optim, key, teacherParams=None, run=None):
    opt_state = optim.init(params)
    losses = [0] * EPOCHS
    NUM_RANDOM_DIM = 10
    new_key, subkey = jax.random.split(key)

    if not teacherParams == None:
        pinn = PINNS(features=HJB_NN_LAYERS)
        valuePolicy = lambda theta, omega: argmin_hamiltonian_analytic(pinn, teacherParams, affine_dynamics_g_func, theta, omega, stepNum=100000)
        residual_func = lambda state: pinns_loss_hamiltonian(pinn, teacherParams, state[0], state[1], valuePolicy, 100000)
        residuals = None

    @jax.jit
    def make_step(opt_state, params, sampled_ics, stepNum):

        loss, grads = jax.value_and_grad(po_batched_loss)(params, sampled_ics, stepNum)
        updates, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return opt_state, params, loss
    
    @jax.jit
    def supervised_make_step(opt_state, params, teacher_params, input_states, stepNum):

        loss, grads = jax.value_and_grad(supervised_po_batched_loss)(params, teacher_params, input_states, stepNum)
        updates, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return opt_state, params, loss
    
    addition_Str = ""
    if not teacherParams == None:
        addition_Str = " Supervised learning "
    print("PO:" + addition_Str + "then learning from rollouts")
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for batch in range(NUM_BATCHES):
            new_key, subkey = jax.random.split(new_key)
            if teacherParams == None or epoch > EPOCHS / 2:
                sampled_ics = sample_ics(subkey, PO_IC_BATCH_SIZE, stepNum=epoch)
                opt_state, params, loss = make_step(opt_state, params, sampled_ics, stepNum=epoch)
            else:
                if epoch % UPDATE_RESIDS_EVERY == 0:
                    residuals = jax.vmap(residual_func)(states)
                input_states = sampling.mixed_sample(states, residuals, theta_range, omega_range, theta_grid.shape, BATCH_SIZE, P_UNIFORM_SAMPLING, subkey)
                opt_state, params, loss = supervised_make_step(opt_state, params, teacherParams, input_states, stepNum=epoch)
            if not run == None:
                teacher_str = "" if teacherParams == None else "_supervised"
                loss_log_str = "po_loss" + teacher_str
                run.log({loss_log_str: loss})
            epoch_loss += loss

            if (batch % PRINT_EVERY == 0) or (batch == NUM_BATCHES - 1):
                    print(f"Epoch {epoch}, batch {batch}, loss: {loss}")
            if not jnp.isfinite(loss):
                print("Bad loss at batch", batch)
                break

        losses[epoch] = epoch_loss / NUM_BATCHES
    
    return params, losses

def train_hjpo(params, optim, key, run=None):
    # params in this one is the pinn's params, as that is the only neural network
    opt_state = optim.init(params)
    losses = [0] * EPOCHS
    NUM_RANDOM_DIM = 10
    new_key, subkey = jax.random.split(key)

    pinn = PINNS(features=HJB_NN_LAYERS)
    policy = lambda theta, omega: argmin_hamiltonian_analytic(pinn, params, affine_dynamics_g_func, theta, omega, stepNum=100000)
    residual_func = lambda state: pinns_loss_hamiltonian(pinn, params, state[0], state[1], policy, 100000)
    residuals = None 


    @jax.jit
    def make_step(opt_state, params, sampled_ics, sampled_input_states, stepNum):

        loss, grads = jax.value_and_grad(hjpo_batched_loss)(params, sampled_ics, sampled_input_states, stepNum)
        updates, opt_state = optim.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return opt_state, params, loss
    

    print("HJPO: Learning from rollouts and residuals")
    for epoch in range(EPOCHS):
        epoch_loss = 0
        for batch in range(NUM_BATCHES):
            if epoch % UPDATE_RESIDS_EVERY == 0:
                residuals = jax.vmap(residual_func)(states)
            new_key, subkey1, subkey2 = jax.random.split(new_key, 3)
            po_sampled_ics = sample_ics(subkey1, PO_IC_BATCH_SIZE, stepNum=epoch)
            hjb_sampled_input_states = sampling.mixed_sample(states, residuals, theta_range, omega_range, theta_grid.shape, BATCH_SIZE, P_UNIFORM_SAMPLING, subkey2)
            # hjb_sampled_input_states = sample_online(subkey2, stepNum=epoch)
            
            opt_state, params, loss = make_step(opt_state, params, po_sampled_ics, hjb_sampled_input_states, stepNum=epoch)
            if not run == None:
                loss_log_str = "hjpo_loss"
                run.log({loss_log_str: loss})
            epoch_loss += loss

            if (batch % PRINT_EVERY == 0) or (batch == NUM_BATCHES - 1):
                    print(f"Epoch {epoch}, batch {batch}, loss: {loss}")
            if not jnp.isfinite(loss):
                print("Bad loss at batch", batch)
                break

        losses[epoch] = epoch_loss / NUM_BATCHES
    
    return params, losses

def copy_flax_params_with_log(src_params, tgt_params):
    """
    Copy parameters from src_params to tgt_params with logging.
    Handles shape mismatches by partial copy (slicing).
    
    Args:
        src_params: frozen dict of source model parameters
        tgt_params: frozen dict of target model parameters
    
    Returns:
        Frozen dict of target parameters with copied values.
    """
    src = unfreeze(src_params)
    tgt = unfreeze(tgt_params)
    
    copied_full = []
    copied_partial = []
    skipped = []

    def copy_recursive(src_dict, tgt_dict, path=""):
        for key in tgt_dict:
            current_path = f"{path}/{key}" if path else key
            if key in src_dict:
                if isinstance(src_dict[key], dict):
                    copy_recursive(src_dict[key], tgt_dict[key], current_path)
                else:
                    s = src_dict[key]
                    t = tgt_dict[key]
                    if s.shape == t.shape:
                        tgt_dict[key] = s
                        copied_full.append(current_path)
                    else:
                        # Slice as much as possible along each axis
                        slices = tuple(slice(0, min(s_dim, t_dim)) for s_dim, t_dim in zip(s.shape, t.shape))
                        if all(slice_.stop > 0 for slice_ in slices):
                            tgt_dict[key] = t.at[slices].set(s[slices])
                            copied_partial.append(current_path)
                        else:
                            skipped.append(current_path)
            else:
                skipped.append(current_path)
        return tgt_dict

    new_params = copy_recursive(src, tgt)

    print(f"Fully copied layers ({len(copied_full)}): {copied_full}")
    print(f"Partially copied layers ({len(copied_partial)}): {copied_partial}")
    print(f"Skipped layers ({len(skipped)}): {skipped}")

    return new_params
    # return freeze(new_params)


if __name__ == "__main__":
    new_key, subkey = jax.random.split(key)
    del key
    pinn = PINNS(features=HJB_NN_LAYERS)
    po_net = MLP(features=PO_NN_LAYERS)
    init_input = jax.random.uniform(subkey, HJB_NN_LAYERS[0])
    del subkey
    new_key2, subkey2 = jax.random.split(new_key)
    del new_key
    # using the same key here for consistency, comparable across results
    hjb_params = pinn.init(subkey2, init_input)
    hjb_params2 = pinn.init(subkey2, init_input)
    po_params = po_net.init(subkey2, init_input)
    po_params2 = po_net.init(subkey2, init_input)
    del subkey2

    V_fn = lambda state: pinn.apply(hjb_params, state)
    grad_fn = jax.grad(V_fn)
    value_val = pinn.apply(hjb_params, PINNS.feature_fn(0, 0))
    jax.debug.print('V before training at (0, 0) = {})', value_val)
    grad_val = grad_fn(PINNS.feature_fn(0, 0))
    jax.debug.print("∇V wrt input before training at (0, 0) = {}", grad_val)
    jax.debug.print("isfinite? {}", jnp.all(jnp.isfinite(grad_val)))


    
    optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LEARNING_RATE))
    optim_po_pre = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LEARNING_RATE))
    optim_hjb = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LEARNING_RATE))
    optim_po_post = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LEARNING_RATE))
    optim_hjpo = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LEARNING_RATE))
    new_key3, subkey3 = jax.random.split(new_key2)
    del new_key2
    wandb.login()
    run_name = "Serial optimization comparison sungje architecture no clipping"
    # run_name = "HJB supervising PO adaptive sampling optax clip L1 omega norm float64 omega_squared bounded_omega fourier features no projection unsaturated 2"
    run = None
    run = wandb.init(
        entity="k5wang-main-university-of-southern-california",
        project="sid-version",
        name=run_name,
        config=config,
        notes="From VSCode",
        tags=["L1 omega norm", "float64", "adaptive sampling", "no-projection", "seven-features"]
    )

    nxs, nys = 100,100
    grid_theta = jnp.linspace(-GRID_THETA_BOUND, GRID_THETA_BOUND, nxs)
    grid_omega = jnp.linspace(-GRID_OMEGA_BOUND, GRID_OMEGA_BOUND, nys)
    xv, yv = jnp.meshgrid(grid_theta, grid_omega)
    grid_states = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)  # shape (10000, 2)

    def createPlotsAndSimulate(params, network, policyFromNetworkFunc, torqueCalcFunc, fig, axs, descr, descr2="", dualAnimation=False):
        fig.suptitle(f"Plots for {descr + descr2}")
        u_vals = jax.vmap(lambda x: policyFromNetworkFunc(x[0], x[1]))(grid_states)  # shape (10000, act_dim)
        u_grid = u_vals.reshape(xv.shape)       # (100, 100) if scalar actions
        contour1 = axs[0, 0].contourf(xv, yv, u_grid, levels=50, cmap="coolwarm")
        axs[0, 0].set_title(f"Control Action Derived From Network")

        plt.colorbar(contour1, ax=axs[0, 0])
        swingUpUVals = jax.vmap(lambda x: localSwingUp(x[0], x[1]))(grid_states)
        su_grid = swingUpUVals.reshape(xv.shape)
        contour2 = axs[0, 1].contourf(xv, yv, su_grid, levels=50, cmap="coolwarm")
        axs[0, 1].set_title("Control Action From Swing Up Policy")

        plt.colorbar(contour2, ax=axs[0, 1])
        valueNetVals = jax.vmap(lambda x: network.apply(params, PINNS.feature_fn(x[0], x[1])))(grid_states)
        vn_grid = valueNetVals.reshape(xv.shape)
        contour3 = axs[1, 0].contourf(xv, yv, vn_grid, levels=50, cmap="coolwarm")
        axs[1, 0].set_title("Network outputs on variety of states")

        plt.colorbar(contour3, ax=axs[1, 0])
        hjbLossVals = jax.vmap(lambda x: pinns_loss_hamiltonian(network, params, x[0], x[1], policyFromNetworkFunc, stepNum=(EPOCHS)))(grid_states)
        print(f"Average hjb loss across the whole grid: {jnp.mean(hjbLossVals)}")
        # hjbLossVals = jnp.log(hjbLossVals)
        
        max_r = jnp.max(hjbLossVals)
        print(f"Max residual = {max_r:.2e}")
        temp = hjbLossVals.reshape(xv.shape)
        high_idx = jnp.squeeze(jnp.array([jnp.unravel_index(jnp.argmax(temp), temp.shape)]))
        print(high_idx)
        theta_bad = xv[0][high_idx[0]]
        omega_bad = yv[0][high_idx[1]]
        print(f"Highest residual at (θ, ω) = {theta_bad}, {omega_bad}")
        network_fn = lambda x: network.apply(params, MLP.feature_fn(x[0], x[1]))
        v, grad_v = jax.value_and_grad(network_fn)(jnp.array([theta_bad, omega_bad]))
        
        print(f"Network func there: {v}, grad wrt state there: {grad_v}")
        jnp.clip(hjbLossVals, 1e-12, None)
        hjbLoss_grid = hjbLossVals.reshape(xv.shape)
        contour4 = axs[1, 1].contourf(xv, yv, hjbLoss_grid, norm=LogNorm(), levels=50, cmap="coolwarm")
        # axs[1, 1].contour(xv, yv, hjbLoss_grid, levels=[-1, 0], colors="black", linewidths=2)
        # axs[1, 1].contour(xv, yv, hjbLoss_grid, levels=[1, 2], colors="green", linewidths=2)
        axs[1, 1].set_title("HJB residual on variety of states")
        plt.colorbar(contour4, ax=axs[1, 1])

        full_run_name = run_name + "_" + descr
        video_name = "sim_" + descr
        plot_name = "plots_" + descr
        
        thetaLists, omegaLists, times = simulateWithDiffraxIntegration(simulation_ode, torqueCalcFunc, t_stop, dt, 
                                    initial_thetas, initial_omegas, m, b, L, G, k, umax, policyFromNetworkFunc, 
                                    run_name=full_run_name)
        

        
        if run != None:
            run.log({video_name: wandb.Video(f"E:\\usc sure\\hjpo\\{full_run_name}.mp4", fps=4, format="mp4")})
            run.log({plot_name: wandb.Image(fig)})
        plt.close(fig=fig)

        if dualAnimation:
            print("Generating dual animation for " + descr)
            whichListData = int(NUM_ICS / 2)
            thetaList = thetaLists[whichListData]
            thetaList = (thetaList + jnp.pi) % (2 * jnp.pi) - jnp.pi
            omegaList = omegaLists[whichListData]
            x = -L * jnp.sin(thetaList)
            y = L * jnp.cos(thetaList)

            ax = axs[0, 1]  # swapping the control policy action for the animation
            ax.clear()
            ax.set_ylim(-L, 1.)
            ax.set_xlim(-L, L)
            ax.set_aspect('equal')
            ax.grid()
            line, = ax.plot([], [], 'o-', lw=2)
            trace, = ax.plot([], [], '.-', lw=1, ms=2)

            otherAxs = [axs[0, 0], axs[1, 0], axs[1, 1]]

            def update(frame):
                artists = []
                line.set_data([0, x[frame]], [0, y[frame]])
                artists.append(line)
                artists.append(trace)
                for i, otherAxis in enumerate(otherAxs):
                    otherAxis.plot(thetaList[frame], omegaList[frame], 'yo')  # use this to plot a single point
                return artists
            
            dual_vid_name = run_name + "_" + descr + "_dual_animation"
            saveAnimatedFig(fig, update, times, interval=dt*1000, blit=True, vid_name = dual_vid_name)
            print("Finished saving animation for " + descr)

    
    # PARTIAL FUNCTIONS FOR NETWORKS
    def policyFromPolicyNetworkPre(theta, omega):
        return po_net.apply(po_params, MLP.feature_fn(theta, omega))
    def policyFromPolicyNetworkPost(theta, omega):
        return po_net.apply(po_params2, MLP.feature_fn(theta, omega))
    def networkPolicyTorqueCalc(networkPolicy, *fluff_args, theta, omega):
        return networkPolicy(theta, omega)
    def policyFromValueFunc(theta, omega):
        return argmin_hamiltonian_analytic(pinn, hjb_params, affine_dynamics_g_func, theta, omega, EPOCHS * NUM_BATCHES)
    # def hjpoPolicyFromValueFunc(theta, omega):
    #     return argmin_hamiltonian_analytic(pinn, hjpo_params, affine_dynamics_g_func, theta, omega, EPOCHS * NUM_BATCHES)
    def valuePolicyTorqueCalc(valuePolicy, *fluff_args, theta, omega):
        # This might be returning some wrong values in extreme cases
        return valuePolicy(theta, omega)

    # TRAINING PO NO TEACHER
    po_params_pre, po_losses_pre = train_po(po_params, optim_po_pre, subkey3, teacherParams=None, run=run)
    po_params = po_params_pre
    
    # TRAINING HJB
    hjb_params, hjb_losses = train_hjb(hjb_params, optim_hjb, subkey3, run=run)
    print("HJB Value Function at (0, 0): {}", pinn.apply(hjb_params, PINNS.feature_fn(0, 0)))

    # # WARM STARTING PO WITH HJB PARAMS
    # hjb_params_copy = copy_flax_params_with_log(hjb_params, po_params2)
    # po_params_post, po_losses_post = train_po(po_params2, optim_po_post, subkey3, teacherParams=None, run=run)
    # print("Policy network at (0, 0): {}", po_net.apply(po_params_post, MLP.feature_fn(0, 0)))
    # po_params2 = po_params_post 

    # TRAINING PO WITH HJB SUPERVISION
    po_params_post, po_losses_post = train_po(po_params2, optim_po_post, subkey3, teacherParams=None, run=run)
    print("Policy network at (0, 0): {}", po_net.apply(po_params_post, MLP.feature_fn(0, 0)))
    po_params2 = po_params_post

    # JOINT HJPO OPTIMIZATION
    # hjpo_params, hjpo_losses = train_hjpo(hjb_params2, optim_hjpo, subkey3, run=run)
    # print("HJPO Value Function at (0, 0): {}", pinn.apply(hjpo_params, MLP.feature_fn(0, 0)))

    # EVALUATING TOTAL STAGE COST FOR QUANTITATIVE COMPARISON
    def batched_rollout_average_loss(actionFunc, ics):
        return jnp.mean(jax.vmap(lambda batch_ic: po_loss_rollout(actionFunc, batch_ic, 10000))(ics))
    
    NUM_EVAL_ROLLOUTS = 30
    new_key4, subkey4 = jax.random.split(new_key3)
    eval_ics = sample_ics(subkey4, NUM_EVAL_ROLLOUTS, 10000)
    eval_ics = jnp.array([eval_ics[0:9], eval_ics[10:19], eval_ics[20:29]])
    hj_trained_po_average_rollout_cost = 0
    hjb_average_rollout_cost = 0
    for i in range(3):
        hj_trained_po_average_rollout_cost += batched_rollout_average_loss(policyFromPolicyNetworkPost, eval_ics[i])
        print("Done with ", (i + 1) * 10, " eval rollouts for supervised PO out of ", 30)
    for i in range(3):
        hjb_average_rollout_cost += batched_rollout_average_loss(policyFromValueFunc, eval_ics[2])
        print("Done with ", (i + 1) * 10, " eval rollouts for HJB out of ", 30)
    hj_trained_po_average_rollout_cost /= 3
    hjb_average_rollout_cost /= 3
    print("Average rollout cost of supervised PO: ", hj_trained_po_average_rollout_cost)
    print("Average rollout cost of HJB: ", hjb_average_rollout_cost)

    # DOING ALL THE PLOTTING AFTERWARD
    po_pre_fig, po_pre_axs = plt.subplots(2, 2, figsize=(20, 10))
    createPlotsAndSimulate(po_params_pre, po_net, policyFromPolicyNetworkPre, networkPolicyTorqueCalc, 
                           po_pre_fig, po_pre_axs, "Policy Optimization Without Supervision", descr2=" (ignore bottom two plots)", dualAnimation=False)
    hjb_fig, hjb_axs = plt.subplots(2, 2, figsize=(20, 10))
    createPlotsAndSimulate(hjb_params, pinn, policyFromValueFunc, valuePolicyTorqueCalc, 
                           hjb_fig, hjb_axs, "HJB With PINN", dualAnimation=False)
    po_post_fig, po_post_axs = plt.subplots(2, 2, figsize=(20, 10))
    createPlotsAndSimulate(po_params_post, po_net, policyFromPolicyNetworkPost, networkPolicyTorqueCalc, 
                           po_post_fig, po_post_axs, "Policy Optimization supervised by HJB", descr2=" (ignore bottom two plots)", dualAnimation=False)
    # hjpo_fig, hjpo_axs = plt.subplots(2, 2, figsize=(20, 10))
    # createPlotsAndSimulate(hjpo_params, pinn, hjpoPolicyFromValueFunc, valuePolicyTorqueCalc,
    #                        hjpo_fig, hjpo_axs, "Joint HJPO")
    
    if run != None:
        wandb.finish()