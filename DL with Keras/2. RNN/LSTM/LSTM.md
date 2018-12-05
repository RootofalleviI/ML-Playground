# LSTM

## Pseudocode

```python
output_t = activation(dot(state_t, Uo) + dot(input_t, Wo) + doc(C_t, Vo) + bo)

i_t = activation(dot(state_t, Ui) + dot(input_t, Wi) + bi)
f_t = activation(dot(state_t, Uf) + dot(input_t, Wf) + bf)
k_t = activation(dot(state_t, Uk) + dot(input_t, Wk) + bk)

c_t+1 = i_t * k_t + c_t * f_t
```

**Explanation**
- Recall from SimpleRNN, we had `output_t = activation(dot(state_t, U) + dot(input_t, W) + b)`. The `o` in `Uo`, `Wo` and `bo` stand for "output", as they deal with the ouput. 
- We add an additional data flow, `c` (for carry), which carries information across timesteps. This will be combined with the input connection and the recurrent connection via a dense tranformation -- a dot product with a weight matrix followed by a bias add and the applicaation of an activation function -- and it will affect the state being sent to the next step via an activation function and a multiplication operation. We call its weight matrix `V` (more specific, `Vo`).
- Next, the way the next value of carry dataflow is computed. It involves three distinct transformations, each has the form of a simple RNN cell, `y = activation(dot(state_t, U) + dot(input_t, W) + b)`. Since each has its own weight matrices, we index them with subscripts `i`, `j`, and `k`. We obtain the new carry state, `c_t+1`, by combining `i_t`, `f_t`, and `k_t`.

## Meaning of operations

While you could get philosophical and interpret each of the operation as a *forget* gate or *update* gate, what these operations actually do is determined by the contents of the weights parameterizing them; and the weights are learned in an end-to-end fashion, starting over with each training round, making it impossible to credit this or taht operation with a specific purpose. The specification of an RNN cell determines your hypothesis space -- the space in which you'll search for a good model configuration during training -- but it doesn't determine what the cell does; that is up to the cell weights as the same cell with different weights can be doing very different thiings. So the combination of operations making up an RNN cell is better interpreted as a set of *constraints* on your search, not as a *design* in an engineering sense. 

It shouldn't be your job to understand what an RNN cell actually does. Just keep in mind that, an LSTM cell allows past information to be reinjected at a later time, thus fighting the vanishing-gradient prolblem.
