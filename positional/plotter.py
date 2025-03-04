# same as shown in the annotated transformers notebook
# https://nlp.seas.harvard.edu/annotated-transformer/


import pandas as pd
import altair as alt
from typing import List
import jax.numpy as jnp
import numpy as np


def plot_encoding(y: jnp.array, max_len: int, dim_range: List[int]):
    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": np.array(y)[0, :, dim],
                    "dimension": dim,
                    "position": list(range(max_len)),
                }
            )
            for dim in dim_range
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )
