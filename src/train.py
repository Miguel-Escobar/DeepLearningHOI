from model import Student, Teacher
import jax, optax, jax.numpy as jnp
from flax import nnx

def mse_loss(model, x_batch, y_values):
    model_output = model(x_batch)
    loss = optax.squared_error(model_output, y_values).mean()
    return loss, model_output

def generate_x_data(n_slow_bits, n_fast_bits, n_samples, rngs, slow_switching = True, switch_every = 10_000, add_bias = True):
    x_data = jnp.zeros((n_samples, n_slow_bits + n_fast_bits + add_bias))
    x_data[:min(n_samples, switch_every), :n_slow_bits] = jax.random.bernoulli(#WEAS
                                                                               
    """
    ESTO ESTÁ PENDIENTE!!!
    
    la idea es que generemos la secuencia de unos y zeros con random.bernoulli (ojo que hay que generar las random keys para que funcione), sampleando los bits rápidos cada vez, mientras que los lentos sólo cada switch_every sampleos. Esto no debiera ser muy raro. Hay que revisar tb lo de rngs en flax.nnx aparte de jax.random.
    """
    
    return

def train_step(model: Student, teacher: Teacher, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, x_batch):
    teacher_values = teacher(x_batch)
    grad_loss_fn = nnx.value_and_grad(mse_loss)
    (loss, model_output), grads = grad_loss_fn(model, x_batch, teacher_values)
    metrics.update(loss=loss, model_output=model_output, teacher_output=teacher_values)
    optimizer.update(grads)
    
    """
    técnicamente esto debiera funcionar d pana. Puede que la grad loss fn no esté vectorizada pero mi intuición es que sí
    """
    return None

def eval_step(model: Student, teacher: Teacher, metrics: nnx.MultiMetric, x_batch):
    teacher_values = teacher(x_batch)
    loss, model_output = mse_loss(model, x_batch, teacher_values)
    metrics.update(loss=loss, model_output=model_output, teacher_output=teacher_values)
    
    """
    esto es para ir recopilando la performance a cada iteración. Luego vamos a promediar cada 10_000 iteraciones o más
    """
    return None    