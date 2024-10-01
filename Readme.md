# Safety Interpretability

## Quick Start

`pip install requirements.txt`

## Ships On Specific Harmful Queries

See `Ships_quick_start.ipynb`

## Ships On Dataset Level

See `Generalized_Ships.ipynb`

## Ablated Safety Attention Head
By *Ships* or *Generalized Ships*, we can attribute safety heads. Then, we can ablate safety head following `Surgery.ipynb` to obtain an ablated LLMs. The weights also can be load from `transformers.AutoModel` instead of `custommodel`.

