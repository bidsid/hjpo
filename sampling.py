import jax
import jax.numpy as jnp

# The next few functions evaluate residuals over the state space
# And also help find where the high residuals are
# And also help to plot the residual map (used for after training as usual
# but also possible to do it during training now and see it update)
def create_space_grid(theta_range, omega_range, grid_size=(50, 50)):
    theta_lin = jnp.linspace(*theta_range, grid_size[0])
    omega_lin = jnp.linspace(*omega_range, grid_size[1])
    theta_grid, omega_grid = jnp.meshgrid(theta_lin, omega_lin, indexing='ij')
    states = jnp.stack([theta_grid.ravel(), omega_grid.ravel()], axis=-1)
    return states, theta_lin, omega_lin, theta_grid, omega_grid
# residuals = residual_fn(states)
# residuals_grid = residuals.reshape(grid_size)
# do these after create_space_grid to get it ready for plotting

def continuous_residual_density(x, grid_states, residuals, bandwidth=0.5):
    # RBF kernel weight for each sample x vs each grid point
    # x: [N, 2], grid_states: [G, 2], residuals: [G]
    
    @jax.jit
    def rbf(xi):
        dists = jnp.linalg.norm(grid_states - xi, axis=1)  # [G]
        weights = jnp.exp(-0.5 * (dists / bandwidth) ** 2)  # [G]
        return jnp.sum(weights * residuals)

    return jax.vmap(rbf)(x)  # returns [N]

@jax.jit
def fast_continuous_residual_density(xs, grid_states, residuals, bandwidth):
    diff = xs[:, None, :] - grid_states[None, :, :]
    sq_distances = jnp.sum(diff ** 2, axis=-1)
    weights = jnp.exp(-0.5 * sq_distances / (bandwidth ** 2))
    return weights @ residuals

@jax.jit
def inverse_distance_weighting(xs, grid_states, residuals, p=2, epsilon=1e-6):
    # xs: [N, 2], grid_states: [G, 2], residuals: [G]
    diff = xs[:, None, :] - grid_states[None, :, :]  # [N, G, 2]
    dists = jnp.linalg.norm(diff, axis=-1) + epsilon  # [N, G]
    weights = 1.0 / (dists ** p)  # decay ~ 1/r^p
    weighted_sum = weights @ residuals  # [N]
    norm = jnp.sum(weights, axis=1)
    return weighted_sum / norm

def sample_weighted_from_domain(grid_states, residuals, theta_range, omega_range, grid_size, num_samples, key, bandwidth=0.5):

    # Sample uniform candidates
    key1, key2 = jax.random.split(key)
    theta_samples = jax.random.uniform(key1, (num_samples * 5,), minval=theta_range[0], maxval=theta_range[1])
    omega_samples = jax.random.uniform(key2, (num_samples * 5,), minval=omega_range[0], maxval=omega_range[1])
    candidates = jnp.stack([theta_samples, omega_samples], axis=-1)  # [M, 2]

    # Compute smooth density
    density = inverse_distance_weighting(candidates, grid_states, residuals)
    # density = continuous_residual_density(candidates, grid_states, residuals, bandwidth=bandwidth)
    # density = density / jnp.max(density)  # Normalize to [0, 1]

    # Rejection sampling
    key_reject = jax.random.PRNGKey(999)
    accept_prob = jax.random.uniform(key_reject, (len(candidates),))
    accepted = candidates[accept_prob < density]
    
    # Return first N accepted samples
    return accepted[:num_samples]

# def compute_sampling_weights(residuals, threshold_ratio=0.7):
#     threshold = jnp.quantile(residuals, threshold_ratio)
#     weights = jnp.where(residuals > threshold, residuals, 0.0)
#     weights = weights / jnp.sum(weights)
#     return weights

# def weighted_sample(states, weights, num_samples, key):
#     idx = jax.random.choice(key, len(states), shape=(num_samples,), p=weights)
#     return states[idx]

def uniform_sample(theta_range, omega_range, num_samples, key):
    k1, k2 = jax.random.split(key)
    ths = jax.random.uniform(k1, (num_samples, 1), minval=theta_range[0], maxval=theta_range[1])
    dths = jax.random.uniform(k2, (num_samples, 1), minval=omega_range[0], maxval=omega_range[1])
    return jnp.squeeze(jnp.stack([ths, dths], axis=-1))

def mixed_sample(states, residuals, theta_range, omega_range, grid_size, num_samples, uniform_prop, key):
    w_key, u_key = jax.random.split(key)
    num_uniform = int(uniform_prop * num_samples)
    num_weighted = num_samples - num_uniform

    weighted = sample_weighted_from_domain(states, residuals, theta_range, omega_range, grid_size, num_weighted, w_key)

    uniform = uniform_sample(theta_range, omega_range, num_uniform, u_key)

    mixed = jnp.concatenate([weighted, uniform], axis=0)
    return mixed