from flax import nnx
import jax.numpy as jnp
from jax import random


def threshold(x, tau):
    return 2.0 * (x > tau) - 1.0


def teacher_initializer(key, shape, dtype=jnp.float32):
    return 2.0 * random.bernoulli(key, 0.5, shape) - 1.0


class Student(nnx.Module):
    def __init__(
        self, input_dim, layer_width, activationfn, rng
    ):  # Me encantaría que pueda darle un número de capas variable también, pero flax no tiene algo como nn.ModuleList y de la documentación no me queda claro cómo hacerlo.
        super().__init__()
        self.input_dim = input_dim
        self.layerwidth = layer_width
        self.activation = activationfn
        self.first_layer = nnx.Linear(self.input_dim, self.layerwidth, rngs=rng)
        self.hidden_layer = nnx.Linear(
            self.layerwidth, 1, rngs=rng
        )  # Acumulo el output de la capa lineal en una sola neurona

    def __call__(self, x):
        x = self.first_layer(x)
        x = self.activation(x)
        x = self.hidden_layer(x)
        return x


class Teacher(nnx.Module):
    def __init__(self, input_dim, layer_width, rng):
        super().__init__()
        self.input_dim = input_dim
        self.layerwidth = layer_width
        self.first_layer = nnx.Linear(
            self.input_dim,
            self.layerwidth,
            kernel_init=teacher_initializer,
            bias_init=teacher_initializer,
            rngs=rng,
        )
        self.hidden_layer = nnx.Linear(
            self.layerwidth,
            1,
            kernel_init=teacher_initializer,
            bias_init=teacher_initializer,
            rngs=rng,
        )  # Acumulo el output de la capa lineal en una sola neurona
        # S = (self.input_dim - jnp.sum(self.first_layer.kernel.value, axis=0) + self.first_layer.bias.value) / 2
        # self.threshold_value = 0.75 * (self.input_dim + 1) - S # Beta sacado del repo del paper
        
        self.activation = lambda x: 2.0 * (x > (0.75 * (self.input_dim + 1) - (self.input_dim - jnp.sum(self.first_layer.kernel.value, axis=0) + self.first_layer.bias.value) / 2)) - 1.0

    def __call__(self, x):
        x = self.first_layer(x)
        x = self.activation(x)#threshold(x, self.threshold_value)
        x = self.hidden_layer(x)
        return x
