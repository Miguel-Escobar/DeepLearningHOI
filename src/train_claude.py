from functools import partial
import jax
import jax.numpy as jnp
import optax
from flax import nnx
from typing import Tuple, Any
from flax.training import train_state
import jax.random
from model import Student, Teacher

def _generate_slow_bits(key, n_slow_bits: int) -> jnp.ndarray:
    """Generate slow bits for a single switch period."""
    return jax.random.bernoulli(key, 0.5, (1, n_slow_bits))

@partial(jax.jit, static_argnums=(1, 2, 3))
def generate_x_data(key, n_samples: int, n_slow_bits: int, n_fast_bits: int, 
                   switch_every: int = 10_000, add_bias: bool = True):
    """Vectorized data generation using vmap."""
    add_bias = int(add_bias)
    total_features = n_slow_bits + n_fast_bits + add_bias
    
    # Generate base array
    x_data = jnp.ones((n_samples, total_features))
    
    # Generate fast bits all at once
    key, subkey = jax.random.split(key)
    fast_bits = jax.random.bernoulli(subkey, 0.5, (n_samples, n_fast_bits))
    x_data = x_data.at[:, add_bias:(n_fast_bits+add_bias)].set(fast_bits)
    
    # Generate keys for slow bits
    n_switches = (n_samples + switch_every - 1) // switch_every
    keys = jax.random.split(key, n_switches)
    
    # Use vmap to generate slow bits in parallel
    vmapped_generate = jax.vmap(_generate_slow_bits, in_axes=(0, None))
    slow_bits = vmapped_generate(keys, n_slow_bits)
    slow_bits = slow_bits.reshape((n_switches, n_slow_bits))
    
    # Create indices for updating slow bits
    indices = jnp.arange(n_samples)
    switch_indices = indices // switch_every
    
    # Use advanced indexing to set slow bits
    x_data = x_data.at[:, (n_fast_bits+add_bias):].set(slow_bits[switch_indices])
    
    return x_data

@partial(jax.jit, static_argnums=(1,))
def process_batch(params: Any, model: Student, x_batch: jnp.ndarray) -> jnp.ndarray:
    """Single batch processing function to be vmapped."""
    return model.apply(params, x_batch)

@jax.jit
def batch_x_data(x_data: jnp.ndarray, batch_size: int) -> jnp.ndarray:
    """Efficient batching using reshape."""
    n_samples = len(x_data)
    n_complete_batches = n_samples // batch_size
    return x_data[:n_complete_batches * batch_size].reshape((n_complete_batches, batch_size, -1))

@jax.jit
def mse_loss(params: Any, model: Student, x_batch: jnp.ndarray, 
             y_values: jnp.ndarray) -> Tuple[float, jnp.ndarray]:
    """Compute MSE loss using vmapped model application."""
    # Vmap the model application over the batch dimension
    batch_process = jax.vmap(partial(process_batch, params, model))
    model_output = batch_process(x_batch)
    loss = jnp.mean(jnp.square(model_output - y_values))
    return loss, model_output

@partial(jax.jit, static_argnums=(1, 2))
def train_step(state: Any, model: Student, teacher: Teacher, x_batch: jnp.ndarray) -> Tuple[Any, dict]:
    """Single training step with vmapped forward passes."""
    def loss_fn(params):
        # Vmap teacher prediction
        teacher_pred = jax.vmap(teacher)(x_batch)
        loss, model_output = mse_loss(params, model, x_batch, teacher_pred)
        return loss, (loss, model_output, teacher_pred)
    
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
    """Training loop with vmapped operations."""
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
        # Vmap over devices first, then over batch dimensions
        state = jax.device_put_replicated(state, jax.local_devices())
        train_step_pmap = jax.pmap(jax.vmap(train_step, in_axes=(None, None, None, 0)), 
                                 axis_name='batch')
    
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