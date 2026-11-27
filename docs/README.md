# Documentation

| # | Chapter | Read it when you want to know … |
|---|---|---|
| 1 | [Architecture and module reference](01-architecture.md) | What each module is, with its generics, ports, timing and internal structure |
| 2 | [Execution and data flow: the road map](02-execution-flow.md) | Where a run starts, what happens in which clock, where every value lives, and where it ends |
| 3 | [Fixed-point arithmetic and the training math](03-fixed-point-and-math.md) | Number formats, rounding, the backprop derivation, and a worked numeric example of one training step |
| 4 | [Training: datasets, procedure and results](04-training-and-datasets.md) | The five datasets, how training is run and judged, measured results, and adding a dataset |
| 5 | [Verification and simulation](05-verification-and-simulation.md) | How correctness is proven, the test benches, and running GHDL / ModelSim / Vivado |
| 6 | [Extending the design](06-extending.md) | Other topologies, activations, formats, bigger networks, putting it on a board |

**Suggested reading order:**

- New to the project: 1.1–1.2, then chapter 2, then chapter 4.
- Reviewing the hardware: chapters 1, 3 and 5.

**Glossary**

| Term | Meaning |
|---|---|
| level | A vector of values: level 0 = inputs, level L = outputs |
| layer *l* | The weights and neurons that map level *l* to level *l+1* |
| master weight | Q4.12 copy of a weight that training updates. The forward pass uses it rounded to Q4.4. |
| delta *d* | ∂loss/∂(pre-activation) of one neuron |
| training step | One sample: forward + backprop + SGD update (one `start` pulse with `train = '1'`) |
| epoch | Every sample trained once, then every sample evaluated once |
| LSB | One unit of the last bit. For Q4.4 that is 1/16. |
