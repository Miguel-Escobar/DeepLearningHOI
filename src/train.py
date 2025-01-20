from src.model import Student, Teacher
import jax, optax, jax.numpy as jnp
from flax import nnx
from tqdm import tqdm, trange

def generate_x_data(n_samples, n_slow_bits, n_fast_bits, switch_every = 10_000, add_bias = True, seed = 420):
    add_bias *= 1
    x_data = jnp.ones((n_samples, n_slow_bits + n_fast_bits + add_bias))
    key = jax.random.key(seed)
    x_data = x_data.at[:, add_bias:(n_fast_bits+add_bias)].set(jax.random.bernoulli(key, 0.5, (n_samples, n_fast_bits)))

    slow_index = 0
    while True:
        key, subkey = jax.random.split(key)
        x_data = x_data.at[slow_index:min(n_samples, slow_index + switch_every), (n_fast_bits+add_bias):].set(jax.random.bernoulli(subkey, 0.5, (1, n_slow_bits)))
        if slow_index + switch_every >= n_samples:
            break
        slow_index += switch_every
    return x_data

def batch_x_data(x_data, batch_size):
    return jnp.split(x_data, batch_size)

def generate_and_batch_y_data(teacher, x_data, batch_size):
    y_data = teacher(x_data)
    return jnp.split(y_data, batch_size)

def mse_loss(model, x_batch, y_values):
    model_output = model(x_batch)
    loss = optax.squared_error(model_output, y_values).mean()
    return loss, model_output

# @nnx.jit
# def train_step(model: Student, teacher: Teacher, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, x_batch):
#     teacher_values = teacher(x_batch)
#     grad_loss_fn = nnx.value_and_grad(mse_loss, has_aux=True)
#     (loss, model_output), grads = grad_loss_fn(model, x_batch, teacher_values)
#     metrics.update(loss=loss)
#     optimizer.update(grads)
#     return None

@nnx.jit
def train_step(model: Student, teacher_values, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, x_batch):
    grad_loss_fn = nnx.value_and_grad(mse_loss, has_aux=True)
    (loss, model_output), grads = grad_loss_fn(model, x_batch, teacher_values)
    metrics.update(loss=loss)
    optimizer.update(grads)
    return None

def eval_step(model: Student, teacher: Teacher, metrics: nnx.MultiMetric, x_test_data):
    teacher_values = teacher(x_test_data)
    loss, model_output = mse_loss(model, x_test_data, teacher_values)
    metrics.update(loss=loss)
    return None

def train_model(model: Student, teacher: Teacher, x_data, x_data_switch_every, n_epochs, batch_size = 10, learning_rate = 0.005, momentum = 0.9):
    x_batches = batch_x_data(x_data, batch_size)
    teacher_batches = generate_and_batch_y_data(teacher, x_data, batch_size)
    metrics_history = {
    'train_loss': [],
    'test_loss': [],
    }
    n_batches = len(x_batches)
    metrics = nnx.MultiMetric(
    loss=nnx.metrics.Average('loss'),
    )

    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))

    for step in trange(n_epochs):
        # train_step(model, teacher, optimizer, metrics, x_batches[step % n_batches])
        train_step(model, teacher_batches[step % n_batches], optimizer, metrics, x_batches[step % n_batches])
        if (step * batch_size) % x_data_switch_every == 0: # Entramos en un caso de evaluación del modelo, que todavía no sé hacer legit la verdad
            # current_datapoints_start = (step*batch_size) % n_data_points
            # current_datapoints_end = current_datapoints_start + x_data_switch_every
            # eval_step(model, teacher, metrics, x_data[current_datapoints_start:current_datapoints_end])
            for metric, value in metrics.compute().items():  # Compute the metrics.
                metrics_history[f'train_{metric}'].append(value)  # Record the metrics.
            metrics.reset()  # Reset the metrics for the test set.

    return metrics_history