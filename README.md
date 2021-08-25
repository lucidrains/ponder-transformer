## Ponder(ing) Transformer (wip)

Implementation of a Transformer that learns to adapt the number of computational steps it takes depending on the difficulty of the input sequence, using the scheme from the <a href="https://arxiv.org/abs/2107.05407">PonderNet</a> paper. If possible, it will be extended to a per-token basis (like <a href="https://arxiv.org/abs/1807.03819">Universal Transformers</a>). Will also try to abstract out a pondering module that can be used with any block that returns an output with the halting probability.

This repository would not have been possible without repeated viewings of <a href="https://www.youtube.com/watch?v=nQDZmf2Yb9k">Yannic's educational video</a>

## Citations

```bibtex
@misc{banino2021pondernet,
    title   = {PonderNet: Learning to Ponder}, 
    author  = {Andrea Banino and Jan Balaguer and Charles Blundell},
    year    = {2021},
    eprint  = {2107.05407},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
