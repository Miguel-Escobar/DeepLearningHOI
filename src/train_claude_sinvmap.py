from functools import partial
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from typing import Tuple, Any
from flax.training import train_state
import jax.random
from model import Student, Teacher

@partial(jax.jit, static_argnums=(1, 2, 3))
def generate_x_data(key, n_samples: int, n_slow_bits: int, n_fast_bits: int, 
                   switch_every: int = 10_000, add_bias: bool = True):
    """Vectorized data generation using JAX's vmap."""
    add_bias = int(add_bias)
    total_features = n_slow_bits + n_fast_bits + add_bias
    
    # Generate base array
    x_data = jnp.ones((n_samples, total_features))
    
    # Generate fast bits all at once
    key, subkey = jax.random.split(key)
    fast_bits = jax.random.bernoulli(subkey, 0.5, (n_samples, n_fast_bits))
    x_data = x_data.at[:, add_bias:(n_fast_bits+add_bias)].set(fast_bits)
    
    # Generate slow bits more efficiently
    n_switches = (n_samples + switch_every - 1) // switch_every
    key, subkey = jax.random.split(key)
    slow_bits = jax.random.bernoulli(subkey, 0.5, (n_switches, n_slow_bits))
    
    # Create indices for updating slow bits
    indices = jnp.arange(n_samples)
    switch_indices = indices // switch_every
    
    # Use advanced indexing to set slow bits
    x_data = x_data.at[:, (n_fast_bits+add_bias):].set(slow_bits[switch_indices])
    
    return x_data

@jax.jit
def batch_x_data(x_data: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    """More efficient batching using JAX's reshape."""
    n_samples = len(x_data)
    n_complete_batches = n_samples // batch_size
    return x_data[:n_complete_batches * batch_size].reshape((n_complete_batches, batch_size, -1))

@jax.jit
def mse_loss(params: Any, model: Student, x_batch: jnp.ndarray, 
             y_values: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
    """Compute MSE loss using model parameters directly."""
    model_output = model.apply(params, x_batch)
    loss = jnp.mean(jnp.square(model_output - y_values))
    return loss, model_output

@partial(jax.jit, static_argnums=(1, 2))
def train_step(state: Any, model: Student, teacher: Teacher, x_batch: jnp.ndarray) -> Tuple[Any, dict]:
    """Single training step with improved JAX transformations."""
    def loss_fn(params):
        teacher_values = teacher(x_batch)
        loss, model_output = mse_loss(params, model, x_batch, teacher_values)
        return loss, (loss, model_output, teacher_values)
    
    (loss, (_, model_output, teacher_values)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    metrics = {
        'loss': loss,
        'model_output': model_output,
        'teacher_output': teacher_values
    }
    
    return state, metrics

def train_model(model: Student, teacher: Teacher, x_data: jnp.ndarray, 
                n_epochs: int, batch_size: int = 10, 
                learning_rate: float = 0.005, momentum: float = 0.9) -> dict:
    """Optimized training loop with improved metrics tracking."""
    # Initialize training state
    tx = optax.adamw(learning_rate=learning_rate, b1=momentum)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(jax.random.PRNGKey(0), x_data[:1]),
        tx=tx,
    )
    
    # Prepare batches
    batched_data = batch_x_data(x_data, batch_size)
    
    # Initialize metrics
    metrics_history = {
        'train_loss': [],
        'model_outputs': [],
        'teacher_outputs': []
    }
    
    # Training loop with pmap for multi-device training if available
    devices = jax.local_device_count()
    if devices > 1:
        state = jax.device_put_replicated(state, jax.local_devices())
        train_step_pmap = jax.pmap(train_step, axis_name='batch')
    
    for epoch in range(n_epochs):
        for batch_idx in range(len(batched_data)):
            x_batch = batched_data[batch_idx]
            
            if devices > 1:
                state, batch_metrics = train_step_pmap(state, model, teacher, x_batch)
            else:
                state, batch_metrics = train_step(state, model, teacher, x_batch)
            
            # Update metrics
            metrics_history['train_loss'].append(batch_metrics['loss'])
            metrics_history['model_outputs'].append(batch_metrics['model_output'])
            metrics_history['teacher_outputs'].append(batch_metrics['teacher_output'])
    
    return metrics_history