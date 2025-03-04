import numpy as np

import flax.linen as nn
from einops import rearrange


class AbsolutePositionalEncoder(nn.Module):
    d_model: int
    max_len: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x):
        # encoding
        encoding = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len)
        # must be in the shape max_len, 1
        position = rearrange(position, "max_len -> max_len 1")

        factor = np.exp(
            np.arange(0, self.d_model, 2) * (-np.log(np.array([1.0e4])) / self.d_model)
        )

        # encoding for odd and even positions
        # even, 0::2
        encoding[:, 0::2] = np.sin(position * factor)
        # odd, 1::2
        encoding[:, 1::2] = np.cos(position * factor)

        # reshape
        encoding = rearrange(encoding, "s dmodel -> 1 s dmodel")

        encoded_x = x + encoding[:, : x.shape[1]]
        # apply dropout
        encoded_x = nn.Dropout(self.dropout, deterministic=True)(encoded_x)

        return encoded_x
