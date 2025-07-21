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
LAYERS = [3, 128, 128, 2] # number of neurons in each fully-connected layer
ACTIVATION_FUNC = jnp.tanh
NUM_GRID_THETA_POINTS = 100
NUM_GRID_OMEGA_POINTS = 300
NUM_GRID_POINTS = NUM_GRID_THETA_POINTS * NUM_GRID_OMEGA_POINTS
NUM_BATCHES = 10    # per epoch
BATCH_SIZE = 3000# int(NUM_GRID_POINTS / NUM_BATCHES)
LEARNING_RATE = 3e-4
STEPS = 300
PRINT_EVERY = 5
SEED = 5678
EPOCHS = 2000
THETA_COST_COEFF = 1
THETADOT_COST_COEFF = 0.5
ACTION_COST_COEFF_TRAINING = 0.1   # R in the writeup
ACTION_COST_COEFF_PDE = ACTION_COST_COEFF_TRAINING
RANDOM_THETA_AMP = 0
RANDOM_OMEGA_AMP = jnp.pi

# simulation details
m = 1.0
b = 0.0
L = 1.0
G = 9.8
k = 1
umax = m * G * L * 10    # saturated at mgl
theta_initial = -jnp.pi/2
omega_initial = 0
theta_goal = jnp.pi
omega_goal = 0
t_stop = 10    # seconds to simulate
dt = 0.01   # time between each sample
t = jnp.arange(0, t_stop, dt)

# batch training
GRID_THETA_BOUND = jnp.pi
GRID_OMEGA_BOUND = m * G * L
EPSILON = 0
DELTA = 0
initial_thetas = [i * jnp.pi/BATCH_SIZE for i in range(BATCH_SIZE)]
initial_conditions = jnp.stack(jnp.array([initial_thetas, jnp.zeros(BATCH_SIZE)]), -1)


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
        def softplus(y):
            return jnp.log(1 + jnp.e**y)
        
        def inner_call(x):
            # negX = jnp.array([-x[0], -x[1], -x[2]])
            # x = jnp.array([x[0], x[1]*x[2], x[2]**2])
            for i in range(len(self.layers) - 1):
                x = self.layers[i](x)
                # negX = self.layers[i](negX)
                x = ACTIVATION_FUNC(x)
                # negX = ACTIVATION_FUNC(negX)
            x = self.layers[len(self.layers) - 1](x)    # applying the final layer
            # negX = self.layers[len(self.layers) - 1](negX)
            # x = jax.nn.softplus(x)
            # x = 0.5 * (x + negX)
            mat = jnp.outer(x, x) + jnp.eye(2)
            SLACK_WEIGHT = 1e-1
            # y = 0.5 * (jnp.squeeze(x) + jnp.squeeze(negX))
            return x.T @ mat @ x + SLACK_WEIGHT * jnp.inner(x, x)
        
        return inner_call(x)# + jax.lax.stop_gradient(0.01 * (jnp.arctan(x[0] / x[1]))**2 + x[2]**2)

def stage_cost_func(theta, omega, action):
    # goal_arr = jnp.array([theta_goal])
    theta_diff = jnp.mod(theta - theta_goal + jnp.pi, 2 * jnp.pi) - jnp.pi
    # omega_diff = jnp.array([omega_goal]) - omega
    omega_diff = omega_goal - omega
    theta_part = THETA_COST_COEFF * jnp.sin(theta_diff - jnp.pi)**2 + THETA_COST_COEFF * (jnp.cos(theta_diff - jnp.pi) - 1)**2
    energy = 0.5 * m * (L * omega)**2 + m * G * L * (1 - jnp.cos(theta))
    energy_diff = 2 * m * G * L - energy
    return theta_part + THETADOT_COST_COEFF * omega_diff**2 + ACTION_COST_COEFF_TRAINING * jnp.tanh(action)**2

def teacher_stage_cost_func(theta, omega, actionFunc):
    teacher_action = localSwingUp(theta, omega)
    return (teacher_action - actionFunc(theta, omega))**2

def dynamics_eqn_func(theta, omega, u):
    spring_constant = m * G / L
    return (u(theta, omega) - b * omega - spring_constant * jnp.sin(theta - jnp.pi)) / m

def simulation_ode(t, y, args):
    theta, omega = y
    fluff1, fluff2, fluff3, fluff4, fluff5, fluff6, u = args
    return jnp.array([omega, dynamics_eqn_func(theta, omega, u)])

def affine_dynamics_f_func(theta, omega):
    return jnp.array([omega, (b * omega + G * L * jnp.sin(theta - jnp.pi)) / m])

def affine_dynamics_g_func(theta, omega):
    return jnp.array([0, 1 / (m)])

def pinns_loss_hamiltonian(pinn, theta, omega, u_func):
    # is also the hamiltonian which needs to be minimized
    # note that pinn is the neural net but due to how its __call__ method is defined
    # it also works like a function (can be called)
    u = u_func(theta, omega) # where I had the stop gradient before
    # u = u_func(theta, omega)
    stage_cost = stage_cost_func(theta, omega, u)
    grad_V_wrt_unembedded_state = unembedded_grad_wrt_inputs(pinn, theta, omega)
    # jax.debug.print("In pinns_loss_hamiltonian gradVwrtunembeddedstate is {x}", x=grad_V_wrt_unembedded_state)
    # grad_V_wrt_unembedded_state = jax.lax.stop_gradient(grad_V_wrt_unembedded_state)
    f_part = jnp.inner(grad_V_wrt_unembedded_state, affine_dynamics_f_func(theta, omega))
    g_inner_product = jnp.inner(affine_dynamics_g_func(theta, omega), grad_V_wrt_unembedded_state * u_func(theta, omega))
    g_part = g_inner_product#  * (1 / ACTION_COST_COEFF_PDE) * g_inner_product
    # hamiltonian = f_part - 0.5 * g_part
    hamiltonian = f_part + g_part
    # hamiltonian = jnp.inner(grad_V_wrt_unembedded_state, affine_dynamics_f_func(theta, omega)) - 0.5 * (1 / action)
    # hamiltonian = stage_cost + jnp.inner(grad_V_wrt_unembedded_state, jnp.array([omega, dynamics_eqn_func(theta, omega, u_func)]))
    v0_loss = 500 * (pinn(jnp.array([jnp.sin(0), jnp.cos(0), 0])) - 0)**2
    v0_grad_loss = 100 * (grad_wrt_inputs(pinn, theta=jnp.pi, omega=0)[2] - 0)**2
    # jax.debug.print("In pinns_loss_hamiltonian hamiltonian is {x}", x=hamiltonian)
    # jax.debug.print("In pinns_loss_hamiltonian v0 loss is {x}", x=v0_loss)
    return hamiltonian**2 + v0_loss**2

def project(u, umin, umax):
    # return 0.5 * (umin + umax) * jnp.tanh(0.5 * u)
    # return jnp.clip(u, umin, umax)
    def soft_saturate(x, limit):
        return limit * x / (1 + jnp.abs(x))
    return soft_saturate(u, umax)

def grad_wrt_inputs(pinn, theta, omega):
    return jax.grad(pinn)(jnp.array([jnp.sin(theta - jnp.pi), jnp.cos(theta - jnp.pi), omega]))

def unembedded_grad_wrt_inputs(pinn, theta, omega):
    grad_wrt_inputs_fn = jax.grad(pinn)
    embed_eval = grad_wrt_inputs_fn(jnp.array([jnp.sin(theta - jnp.pi), jnp.cos(theta - jnp.pi), omega]))
    dgrad_dtheta = embed_eval[0] * jnp.cos(theta) - embed_eval[1] * jnp.sin(theta - jnp.pi)
    unembed_eval = jnp.array([dgrad_dtheta, embed_eval[2]])
    return unembed_eval

def argmin_hamiltonian_analytic(pinn, dynamics_func_g, theta, omega):
    theta = theta - (5*jnp.pi/6 + 0.1)
    grad_V_wrt_unembedded_state = unembedded_grad_wrt_inputs(pinn, theta, omega)    # 2d array but only second elem nonzero
    return project(-1 * (1 / ACTION_COST_COEFF_PDE) * jnp.inner(dynamics_func_g(theta, omega), grad_V_wrt_unembedded_state), -umax, umax)   # originally negative

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

@eqx.filter_jit
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

@eqx.filter_jit
def total_loss(pinn, input_states):
    actionFromValue = lambda t, o: argmin_hamiltonian_analytic(pinn, affine_dynamics_g_func, t, o)
    # neg_val_penalty = jnp.mean(jnp.square(jnp.minimum(jax.vmap(lambda theta, omega: pinn(jnp.array([jnp.sin(theta), jnp.cos(theta), omega])))(theta_list, omega_list), 0.0)))
    lambda_negval = 10
    x_bottom = jnp.array([jnp.sin(0.0 - jnp.pi), jnp.cos(0.0 - jnp.pi), 0.0])  # [0, 1, 0]
    V_bottom = pinn(x_bottom)
    lambda_bottom = 50
    V_target = 5
    lambda_grad_bottom = 100
    grad_target = 5
    grad = grad_wrt_inputs(pinn, 0, 0)[2]
    lambda_symm = 100
    batched_loss_fn = jax.vmap(pinns_loss_hamiltonian, in_axes=(None, 0, 0, None))

    original_hamiltonian_loss = jnp.mean(jax.vmap(lambda input_state: pinns_loss_hamiltonian(pinn, input_state[0], input_state[1], actionFromValue))(input_states))
    # symm_hamiltonian_loss = jnp.mean(jax.vmap(lambda theta, omega: pinns_loss_hamiltonian(pinn, theta, -omega, actionFromValue))(theta_list, omega_list))
    # symm_loss = (original_hamiltonian_loss - symm_hamiltonian_loss)**2
    # return jax.vmap(lambda theta, omega: pinn(jnp.sin(theta), jnp.cos(theta), omega))(theta_list, omega_list)
    return  original_hamiltonian_loss

def teacher_total_loss(pinn, theta_list, omega_list):
    actionFromValue = lambda t, o: argmin_hamiltonian_analytic(pinn, affine_dynamics_g_func, t, o)
    
    # return jax.vmap(lambda theta, omega: pinn(jnp.sin(theta), jnp.cos(theta), omega))(theta_list, omega_list)
    return jnp.mean(jax.vmap(lambda theta, omega: teacher_stage_cost_func(theta, omega, actionFromValue))(theta_list, omega_list))

def localSwingUp(theta, thetadot):
        E_desired = 2 * m * G * L
        E_current = 0.5 * m * (L * thetadot)**2 + m * G * L * (1 - jnp.cos(theta))
        delta_E = E_current - E_desired
        
        output = -k * delta_E * jnp.sign(thetadot + 1e-5)
        output = jnp.clip(output, -umax, umax)
    
        return output 

# @eqx.filter_jit
def train(pinn, optim, epochs, print_every, key):
    params = eqx.filter(pinn, eqx.is_array)
    opt_state = optim.init(params)
    losses = [0] * EPOCHS
    NUM_RANDOM_DIM = 10
    space_between_thetas = 2 * jnp.pi / NUM_RANDOM_DIM
    shift_each_step = space_between_thetas / (EPOCHS + 1)
    # grid_theta = jnp.linspace(-GRID_THETA_BOUND + EPSILON, GRID_THETA_BOUND - EPSILON, NUM_GRID_THETA_POINTS)
    # grid_omega = jnp.linspace(-GRID_OMEGA_BOUND + DELTA, GRID_OMEGA_BOUND - DELTA, NUM_GRID_OMEGA_POINTS)
    # xv, yv = jnp.meshgrid(grid_theta, grid_omega)
    # grid_states = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)
    new_key, subkey = jax.random.split(key)    

    
    @eqx.filter_jit
    def make_step(pinn, opt_state, params, random_states):

        loss, grads = jax.value_and_grad(total_loss)(pinn, random_states)
        updates, opt_state = optim.update(grads, opt_state)
        pinn = eqx.apply_updates(pinn, updates)
        return pinn, opt_state, params, loss
    
    '''
    print("Learning from teacher")
    for epoch in range(len(losses[0])):
        pinn, opt_state, params, loss = make_step(pinn, opt_state, params, teacher=True)
        if (epoch % print_every == 0) or (epoch == epochs - 1):
                print("Most recent loss: ", loss)
        losses[0][epoch] = loss
    '''

    
    def sample_online(key):
        k1, k2 = jax.random.split(key)
        # uniform randomly samples values from minval to maxval with the given shape
        ths = jax.random.uniform(k1, (BATCH_SIZE, 1), minval=-GRID_THETA_BOUND, maxval=GRID_THETA_BOUND)
        dths = jax.random.uniform(k2, (BATCH_SIZE, 1), minval=-GRID_OMEGA_BOUND, maxval=GRID_OMEGA_BOUND)
        return jnp.squeeze(jnp.stack([ths, dths], axis=-1))

    @eqx.filter_jit
    def sampled_state_lists(batchNum, grid):
        return grid[batchNum * BATCH_SIZE: (batchNum + 1) * BATCH_SIZE]
    

    print("Learning from random sampling")
    for epoch in range(EPOCHS):
        epoch_loss = 0
        # shuffled_grid_states = jax.random.permutation(subkey, grid_states)
        for batch in range(NUM_BATCHES):
            new_key, subkey = jax.random.split(new_key)
            # random_sampled_states = sampled_state_lists(batch, shuffled_grid_states)
            random_sampled_states = sample_online(subkey)
            pinn, opt_state, params, loss = make_step(pinn, opt_state, params, random_sampled_states)
            epoch_loss += loss

            if (batch % print_every == 0) or (batch == NUM_BATCHES - 1):
                    print(f"Epoch {epoch}, batch {batch}, loss: {loss}")
            if not jnp.isfinite(loss):
                print("Bad loss at batch", batch)
                break

        losses[epoch] = epoch_loss / NUM_BATCHES
    
    # print("Learning from batched noisy policy rollouts")
    # for epoch in range(len(losses[1])):
    #     batched_rollout = jax.vmap(lambda i: rolloutFixedValue(pinn, i))
    #     theta_list, omega_list, time_list = batched_rollout(jnp.array(range(BATCH_SIZE)))
    #     theta_list = jnp.squeeze(theta_list.reshape(-1, 1))
    #     omega_list = jnp.squeeze(omega_list.reshape(-1, 1))
    #     time_list = jnp.squeeze(time_list.reshape(-1, 1))
    #     pinn, opt_state, params, loss = make_step(pinn, opt_state, params, theta_list, omega_list, time_list)
    #     if (epoch % print_every == 0) or (epoch == epochs - 1):
    #             x_bottom = jnp.array([0.0, 1.0, 0.0])
    #             print("Most recent loss: ", loss)
    #     losses[1][epoch] = loss
    
    return pinn, losses

if __name__ == "__main__":
    new_key, subkey = jax.random.split(key)
    del key
    pinn = PINNS(LAYERS, ACTIVATION_FUNC, subkey)
    del subkey
    # print(pinns_loss_hamiltonian(pinn, theta_initial, omega_initial, lambda t, o: 1))
    params = eqx.filter(pinn, eqx.is_array)
    # optim = optax.lbfgs(LEARNING_RATE)
    optim = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(LEARNING_RATE)
    )
    opt_state = optim.init(params)

    V_fn = lambda state: pinn(state)
    grad_fn = jax.grad(V_fn)

    grad_val = grad_fn(jnp.array([1, 0, -m * G * L]))
    jax.debug.print("∇V at (-π, -mgl) = {}", grad_val)
    jax.debug.print("isfinite? {}", jnp.all(jnp.isfinite(grad_val)))


    # grads = jax.grad(lambda pinn, theta, omega: pinns_loss_hamiltonian(pinn, theta, omega, lambda t, o: 1))(pinn, theta_initial, omega_initial)
    # updates, opt_state = optim.update(grads, opt_state)
    # pinn = eqx.apply_updates(pinn, updates)
    # print(pinns_loss_hamiltonian(pinn, theta_initial, omega_initial, lambda t, o: 1))
    
    
    optim = optax.adam(LEARNING_RATE)
    new_key2, subkey = jax.random.split(new_key)
    del new_key
    finished_value_func, losses = train(pinn, optim, EPOCHS, PRINT_EVERY, subkey)
    print("V at (0, 0): {}", pinn(jnp.array([0, 1, 0])))
    
    def policyFromValueFunc(theta, omega):
        return argmin_hamiltonian_analytic(pinn, affine_dynamics_g_func, theta, omega)
    def valuePolicyTorqueCalc(valuePolicy, *fluff_args, theta, omega):
        return valuePolicy(theta, omega)
    
    nxs, nys = 100,100
    grid_theta = jnp.linspace(-jnp.pi, jnp.pi, nxs)
    grid_omega = jnp.linspace(-m*G*L, m*G*L, nys)
    xv, yv = jnp.meshgrid(grid_theta, grid_omega)
    grid_states = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)  # shape (10000, 2)

    fig, axs = plt.subplots(2, 2, figsize=(20, 10))

    u_vals = jax.vmap(lambda x: policyFromValueFunc(x[0] - jnp.pi, x[1]))(grid_states)  # shape (10000, act_dim)
    u_grid = u_vals.reshape(xv.shape)       # (100, 100) if scalar actions
    contour1 = axs[0, 0].contourf(xv, yv, u_grid, levels=50, cmap="coolwarm")
    axs[0, 0].set_title("Control Action from Value Function")

    plt.colorbar(contour1, ax=axs[0, 0])
    swingUpUVals = jax.vmap(lambda x: swingUpU(None, m, G, L, b, k, umax, x[0] - jnp.pi, x[1]))(grid_states)
    su_grid = swingUpUVals.reshape(xv.shape)
    contour2 = axs[0, 1].contourf(xv, yv, su_grid, levels=50, cmap="coolwarm")
    axs[0, 1].set_title("Control Action From Swing Up Policy")

    plt.colorbar(contour2, ax=axs[0, 1])
    valueNetVals = jax.vmap(lambda x: pinn(jnp.array([jnp.sin(x[0] - jnp.pi), jnp.cos(x[0] - jnp.pi), x[1]])))(grid_states)
    vn_grid = valueNetVals.reshape(xv.shape)
    contour3 = axs[1, 0].contourf(xv, yv, vn_grid, levels=50, cmap="coolwarm")
    axs[1, 0].set_title("Value network outputs on variety of states")

    plt.colorbar(contour3, ax=axs[1, 0])
    hjbLossVals = jax.vmap(lambda x: pinns_loss_hamiltonian(pinn, x[0] - jnp.pi, x[1], policyFromValueFunc))(grid_states)
    hjbLossVals = jnp.log(hjbLossVals)
    hjbLoss_grid = hjbLossVals.reshape(xv.shape)
    contour4 = axs[1, 1].contourf(xv, yv, hjbLoss_grid, levels=50, cmap="coolwarm")
    axs[1, 1].contour(xv, yv, hjbLoss_grid, levels=[-1, 0], colors="black", linewidths=2)
    axs[1, 1].set_title("HJB residual on variety of states")
    plt.colorbar(contour4, ax=axs[1, 1])
    # dV_domega = jax.vmap(lambda theta, omega: jax.grad(pinn)(jnp.array([jnp.sin(theta), jnp.cos(theta), omega]))[2])(grid_states[:, 0], grid_states[:, 1])
    # dV_domega_grid = dV_domega.reshape(xv.shape)
    # # gradValueNetVals = jax.vmap(lambda x: jax.grad(pinn)(jnp.array([jnp.sin(x[0]), jnp.cos(x[0]), x[1]]))[:,2])(grid_states)
    # # gvn_grid = gradValueNetVals.reshape(xv.shape)
    # contour3 = axs[1, 1].contourf(xv, yv, dV_domega_grid, levels=50, cmap="coolwarm")
    # axs[1, 1].set_title("Value gradient on variety of states")
    # plt.colorbar(contour3, ax=axs[1, 1])   
    
    simulateWithDiffraxIntegration(simulation_ode, valuePolicyTorqueCalc, t_stop, dt, 
                                   theta_initial, omega_initial, m, b, L, G, k, umax, policyFromValueFunc, 
                                   loss=losses)