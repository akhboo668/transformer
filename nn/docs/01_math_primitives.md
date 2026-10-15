# Step 1: Math Primitives

> **Status**: In progress
> **Paper sections**: Prerequisites (Linear, Softmax, ReLU), Section 3.1 (LayerNorm), Section 5.4 (Dropout), Section 3.4 (Embedding)
> **Implements**: `nn/src/layers.py`

[< Overview](00_overview.md) | [Next: Positional Encoding >](02_positional_encoding.md)

---

## What We Build

The atomic operations that everything else composes. Each needs both `forward()` and `backward()`.

| Layer | Key Equation | Learnable? | Status |
|-------|-------------|------------|--------|
| Linear | y = xW + b | Yes: W, b | Documented |
| Softmax | softmax(x_i) = e^{x_i} / sum(e^{x_j}) | No | Documented |
| ReLU | max(0, x) | No | Documented |
| LayerNorm | gamma * (x - mean) / sqrt(var + eps) + beta | Yes: gamma, beta | Documented |
| Dropout | x * random_mask / (1 - p) | No | Documented  |
| Embedding | lookup table indexed by token IDs | Yes: embedding matrix | Documented ++ |

---

## Why This Step First

Every Transformer component is built from these pieces. Multi-Head Attention uses Linear layers
for projections. The FFN is two Linear layers with a ReLU. Residual connections go through
LayerNorm. If the gradients through these primitives are wrong, nothing downstream will train.

---

## Foundation: Forward and Backward Passes

Before any specific layer, you need to understand the pattern that **every component** follows.

### What is a Neural Network, Really?

A neural network is a chain of mathematical functions. Data flows **forward** through them to
produce a prediction, then a **loss** (error signal) flows **backward** through them to update
the weights.

```
FORWARD PASS (left to right):
Input  -->  [Layer 1]  -->  [Layer 2]  -->  [Layer 3]  -->  Prediction
                                                                |
                                                          compare with target
                                                                |
                                                              Loss
                                                                |
BACKWARD PASS (right to left):
Loss   <--  [Layer 1]  <--  [Layer 2]  <--  [Layer 3]  <--  gradient
(update W)  (update W)      (update W)
```

- **Forward pass**: compute output from input, layer by layer, left to right.
- **Backward pass**: compute how much each weight contributed to the error, layer by layer,
  right to left.

### The Chain Rule -- the Engine of Backpropagation

If you have y = f(g(x)), the derivative of y with respect to x is:

```
dy/dx = dy/dg * dg/dx
```

Each layer computes its **own local gradient** and multiplies it by the **gradient coming from
above**. This is why every layer needs a `backward()` method -- it receives the gradient from
the layer above and passes its gradient down to the layer below.

### The Layer Interface (our base pattern)

Every layer we build follows this exact interface:

```python
class Layer:
    def forward(self, x):
        # 1. Compute output from input
        # 2. Cache anything needed for backward (x, intermediates)
        # 3. Return output

    def backward(self, grad_output):
        # 1. Receive gradient of loss w.r.t. this layer's OUTPUT
        # 2. Compute gradient of loss w.r.t. this layer's PARAMETERS (if any)
        # 3. Compute gradient of loss w.r.t. this layer's INPUT
        # 4. Return gradient of input (to pass to the layer below)
```

### Concrete Example: Two Simple Layers

Layer 1: f(x) = 2x   (multiply by 2)
Layer 2: g(y) = y + 3 (add 3)

**Forward** (x = 5):
- Layer 1: y = 2 * 5 = 10
- Layer 2: z = 10 + 3 = 13

**Backward** (assume dL/dz = 1, i.e., loss gradient arriving from above):
- Layer 2: local gradient dz/dy = 1, so dL/dy = dL/dz * dz/dy = 1 * 1 = **1**
- Layer 1: local gradient dy/dx = 2, so dL/dx = dL/dy * dy/dx = 1 * 2 = **2**

Key insight: each layer only needs to know:
1. Its own local derivative (dz/dy or dy/dx)
2. The gradient arriving from the layer above (dL/dz or dL/dy)

Multiply them together and pass the result down. That's the chain rule in action.

---

## 1a. Linear Layer

The most fundamental operation in deep learning. Every time a network "thinks," it's doing this.

### The Equation

```
y = xW + b
```

- **x**: input -- shape (batch_size, in_features)
- **W**: weight matrix -- shape (in_features, out_features) -- LEARNABLE
- **b**: bias vector -- shape (out_features,) -- LEARNABLE
- **y**: output -- shape (batch_size, out_features)

### What Each Piece Means

**W (weights)**: one weight per connection between input and output nodes. W[i, j] is the
strength of the connection from input node i to output node j.

**b (bias)**: one bias per OUTPUT node (not input). The bias is the number of output nodes,
which is the number of nodes in the next layer. Each output node gets its own personal offset
that shifts its activation up or down regardless of the input.

### Visualizing the Connections

```
Input layer: 2 nodes           Output layer: 3 nodes

   x[0] ----\
              >----> y[0] = x[0]*W[0,0] + x[1]*W[1,0] + b[0]
   x[1] ----/

   x[0] ----\
              >----> y[1] = x[0]*W[0,1] + x[1]*W[1,1] + b[1]
   x[1] ----/

   x[0] ----\
              >----> y[2] = x[0]*W[0,2] + x[1]*W[1,2] + b[2]
   x[1] ----/

W shape: (2, 3)  -- 2 input nodes x 3 output nodes = 6 connections
b shape: (3,)    -- one bias per output node
```

### Numerical Forward Example

```
x = [2, 1]                    shape: (1, 2) -- 1 sample, 2 features

W = [[0.5, -1.0, 0.3],        shape: (2, 3) -- 2 inputs -> 3 outputs
     [0.2,  0.8, -0.5]]

b = [0.1, 0.0, -0.2]          shape: (3,) -- one bias per output node

Step 1 -- matrix multiply (x @ W):
  y[0] = 2*0.5 + 1*0.2  = 1.2
  y[1] = 2*(-1) + 1*0.8 = -1.2
  y[2] = 2*0.3 + 1*(-0.5) = 0.1
  x @ W = [1.2, -1.2, 0.1]    shape: (1, 3)

Step 2 -- add bias:
  y = [1.2+0.1, -1.2+0.0, 0.1+(-0.2)]
    = [1.3, -1.2, -0.1]       shape: (1, 3)
```

### Shape Rule for Matrix Multiply

```
x:     (batch_size, in_features)
W:     (in_features, out_features)
x @ W: (batch_size, out_features)    -- inner dimensions must match, outer dims remain
```

The in_features dimension "cancels out" (like multiplying fractions), leaving
(batch_size, out_features).

### Broadcasting: How NumPy Adds Different Shapes

After the matrix multiply, we need to add bias b to the result:

```
x @ W  shape: (1, 3)     -- 2D matrix
b      shape: (3,)        -- 1D vector
```

These are different shapes. NumPy uses "broadcasting" to make them compatible.

**Rule 1**: Pad the smaller array with 1s on the LEFT until dimensions match.

```
x @ W  shape: (1, 3)
b      shape:    (3,)  -->  (1, 3)    -- padded with 1 on the left
```

**Rule 2**: If a dimension has size 1, stretch it to match the other array.

After padding, both are (1, 3) -- they match. Addition works element-by-element.

**Where it really matters -- batches**: with 3 samples in a batch:

```
x @ W = [[1.2, -1.2,  0.1],     shape: (3, 3) -- 3 samples
         [0.8,  0.5, -0.3],
         [2.0, -0.7,  0.4]]

b     = [0.1,  0.0, -0.2]       shape: (3,)
                                   --> (1, 3)     -- Rule 1: pad
                                   --> (3, 3)     -- Rule 2: stretch 1 -> 3
```

NumPy treats b as if it were repeated into 3 identical rows:

```
b "becomes" [[0.1,  0.0, -0.2],     -- same bias added to every sample
             [0.1,  0.0, -0.2],
             [0.1,  0.0, -0.2]]
```

The same bias gets added to every sample in the batch. That's correct -- the bias is a
property of the layer, not of any particular input.

**Quick broadcasting rule of thumb**: line up shapes from the RIGHT. Each pair of
dimensions must be either equal, or one of them must be 1.

```
(3, 5, 32) + (32,)      works:  (32) matches (32)
(3, 5, 32) + (5, 32)    works:  (5,32) matches (5,32)
(3, 5, 32) + (5, 1)     works:  1 stretches to 32
(3, 5, 32) + (3, 32)    FAILS:  5 != 3 and neither is 1
```

Broadcasting shows up throughout the Transformer -- attention masking, positional encoding
addition, and bias addition all rely on it.

### Where Linear Appears in the Transformer

Everywhere -- it's the workhorse:
- Each attention head: 4 Linear layers (W^Q, W^K, W^V, W^O)
- Each FFN: 2 Linear layers (W1, W2)
- Final output projection: 1 Linear layer
- With h=4 heads and N=2 layers: roughly 30+ Linear layers total

### Backward Pass

During backward, we receive `grad_output` (dL/dy) and must compute three things:

```
Given:  y = xW + b   and   grad_output = dL/dy

1. dL/dW = x^T @ grad_output       -- how to update weights
2. dL/db = sum(grad_output, axis=0) -- how to update biases
3. dL/dx = grad_output @ W^T        -- gradient to pass to layer below
```

**Why these formulas?** From the chain rule:
- dL/dW: since y = xW, the derivative of y w.r.t. W involves x. The transpose and order
  ensure the shapes align: (in, batch)^T @ (batch, out) = (in, out) = shape of W.
- dL/db: since y = ... + b, the derivative of y w.r.t. b is 1. Sum across batch because
  every sample in the batch contributes to the bias gradient.
- dL/dx: since y = xW, the derivative of y w.r.t. x involves W. Transposing W reverses
  the direction: (batch, out) @ (out, in) = (batch, in) = shape of x.

**Key symmetry**: every gradient has the SAME shape as the thing it's a gradient of.

### Numerical Backward Example

Continuing from the forward example:

```
grad_output (dL/dy) = [0.5, -0.3, 0.8]   shape: (1, 3)

dL/dW = x^T @ grad_output
      = [[2],  @ [[0.5, -0.3, 0.8]]
         [1]]
      = [[1.0, -0.6, 1.6],     shape: (2, 3) -- same as W!
         [0.5, -0.3, 0.8]]

dL/db = [0.5, -0.3, 0.8]       shape: (3,)   -- same as b!

dL/dx = grad_output @ W^T
      = [0.5, -0.3, 0.8] @ [[0.5,  0.2],
                              [-1.0, 0.8],
                              [0.3, -0.5]]
      = [0.5*0.5 + (-0.3)*(-1.0) + 0.8*0.3,    = 0.79
         0.5*0.2 + (-0.3)*0.8    + 0.8*(-0.5)]  = -0.54
      = [0.79, -0.54]           shape: (1, 2) -- same as x!
```

### In the Transformer Context

In our model, inputs won't be simple (1, 2) vectors. They'll be 3D tensors:

```
x shape: (batch_size, seq_len, d_model) = (32, 5, 32)
```

The Linear layer operates on the LAST dimension -- transforming each position's d_model=32
features independently. For example, projecting to d_ff=64 in the FFN:

```
W shape: (32, 64)   -- d_model -> d_ff
b shape: (64,)      -- one bias per output node
y shape: (32, 5, 64) -- same batch and seq_len, new feature dim
```

---

## 1b. Softmax

Converts raw numbers (logits) into probabilities that sum to 1.

### Where It Appears in the Transformer

1. **Inside attention** (Equation 1): `softmax(QK^T / sqrt(d_k))` -- turns raw attention
   scores into attention weights (how much to focus on each position)
2. **Final output layer**: turns the decoder's raw scores into probabilities over the
   vocabulary (which token to predict next)

### The Equation

```
softmax(x_i) = e^(x_i) / sum(e^(x_j))   for all j
```

Take each number, raise e (2.718...) to its power, then divide by the sum of all of them.

### Numerical Forward Example

```
Input (logits):  x = [2.0, 1.0, 0.1]

Step 1 -- exponentiate each:
  e^2.0 = 7.389
  e^1.0 = 2.718
  e^0.1 = 1.105

Step 2 -- sum them:
  7.389 + 2.718 + 1.105 = 11.212

Step 3 -- divide each by the sum:
  softmax = [7.389/11.212,  2.718/11.212,  1.105/11.212]
          = [0.659,          0.242,          0.099]

Check: 0.659 + 0.242 + 0.099 = 1.000  ✓
```

What happened:
- The largest input (2.0) got the highest probability (0.659)
- Differences get amplified -- softmax is a "soft" version of argmax
- All outputs are positive and sum to 1 -- valid probabilities

### Why We Need Probabilities (Not Raw Logits)

Logits are raw unconstrained numbers from the network's last layer. They have three problems:

**Problem 1 -- No meaningful scale**: a logit of 3.8 doesn't tell you if the model is
38% sure or 99% sure. After softmax, 0.87 means "87% confident" -- a meaningful quantity.

**Problem 2 -- Can't compare across positions**: position 1 might have logits in range
[-2, 5] and position 2 in range [-10, -3]. Softmax normalizes each independently.

**Problem 3 -- Loss function needs probabilities**: cross-entropy loss is defined as:

```
loss = -log(probability of correct token)

If correct token has probability 0.87:  loss = -log(0.87) = 0.14  (low -- good!)
If correct token has probability 0.10:  loss = -log(0.10) = 2.30  (high -- bad!)
If correct token has probability 0.01:  loss = -log(0.01) = 4.61  (very high -- terrible!)
```

High probability for the correct answer = low loss = model is right = small weight update.
Low probability = high loss = model is wrong = large weight update. This is how models learn.
Without softmax, log(logit) could be undefined (log of a negative) or meaningless.

**The full pipeline**:

```
Decoder output       Softmax             Loss function
(raw logits)    -->  (probabilities) --> (how wrong?)   --> backward --> learn
[-1.2, 3.8, ...]    [0.02, 0.87, ...]   -log(0.87)=0.14
```

### The Numerical Stability Problem

What if inputs are large?

```
x = [1000, 1001, 999]
e^1000 = ??? (infinity -- too large for a computer!)
```

NumPy gives `inf`, then `inf / inf = NaN`. Model breaks.

### The Fix: Subtract the Max

Mathematical property: subtracting a constant from all inputs doesn't change the output.

```
softmax(x_i) = softmax(x_i - max(x))
```

Why: subtracting constant c means e^c appears in both numerator and denominator, cancels out.

```
x = [1000, 1001, 999]
max(x) = 1001
x - max = [-1, 0, -2]

e^(-1) = 0.368
e^(0)  = 1.000
e^(-2) = 0.135
sum = 1.503

softmax = [0.245, 0.665, 0.090]    (no overflow!)
```

This is called **numerically stable softmax**. Every real implementation uses it.

### Backward Pass -- Full Calculus Derivation

The trickiest backward of all our primitives. Let's derive it from scratch.

**Setup**: we have softmax outputs for a 3-element vector:

```
s_1 = e^(x_1) / S
s_2 = e^(x_2) / S
s_3 = e^(x_3) / S

where S = e^(x_1) + e^(x_2) + e^(x_3)    (the denominator)
```

We need: how does each output s_i change when we nudge each input x_j?

**Prerequisite -- the Quotient Rule**: for f(x)/g(x):

```
d/dx [f/g] = (f' * g  -  f * g') / g^2
```

#### Case 1: i == j (how does s_1 change when we nudge x_1?)

Computing ds_1/dx_1 where s_1 = e^(x_1) / S:

```
f  = e^(x_1)        f' = e^(x_1)       (derivative of e^x is e^x)
g  = S               g' = e^(x_1)       (only the x_1 term in S depends on x_1)

ds_1/dx_1 = (e^(x_1) * S  -  e^(x_1) * e^(x_1)) / S^2
          = e^(x_1) * (S - e^(x_1)) / S^2
```

Split into two fractions:

```
          = (e^(x_1) / S) * ((S - e^(x_1)) / S)
          =      s_1       * (1 - e^(x_1)/S)
          =      s_1       * (1 - s_1)
```

**Result: ds_i/dx_i = s_i * (1 - s_i)**

With our numbers: s_1 = 0.659, so ds_1/dx_1 = 0.659 * 0.341 = 0.225

#### Case 2: i != j (how does s_1 change when we nudge x_2?)

Computing ds_1/dx_2 where s_1 = e^(x_1) / S:

```
f  = e^(x_1)        f' = 0              (e^(x_1) does NOT depend on x_2)
g  = S               g' = e^(x_2)       (only the x_2 term in S depends on x_2)

ds_1/dx_2 = (0 * S  -  e^(x_1) * e^(x_2)) / S^2
          = -e^(x_1) * e^(x_2) / S^2
          = -(e^(x_1)/S) * (e^(x_2)/S)
          = -s_1 * s_2
```

**Result: ds_i/dx_j = -s_i * s_j (when i != j)**

With our numbers: ds_1/dx_2 = -0.659 * 0.242 = -0.159

Intuition: increasing x_2 pushes s_2 UP, so s_1 must go DOWN (they sum to 1). Hence negative.

#### Unified Formula

Both cases in one expression:

```
ds_i/dx_j = s_i * (delta_ij - s_j)

where delta_ij = 1 if i==j, 0 if i!=j   (the "Kronecker delta")
```

Verify:
- i==j: s_i * (1 - s_i)       -- matches Case 1  ✓
- i!=j: s_i * (0 - s_j) = -s_i * s_j  -- matches Case 2  ✓

#### The Full Jacobian Matrix

All partial derivatives form a 3x3 matrix (the "Jacobian"):

```
              dx_1              dx_2              dx_3
           ┌──────────────────────────────────────────────┐
   ds_1    │  s_1*(1-s_1)      -s_1*s_2          -s_1*s_3 │
   ds_2    │ -s_2*s_1           s_2*(1-s_2)      -s_2*s_3 │
   ds_3    │ -s_3*s_1          -s_3*s_2           s_3*(1-s_3)│
           └──────────────────────────────────────────────┘

Plugging in s = [0.659, 0.242, 0.099]:

           ┌──────────────────────────────────┐
           │  0.225      -0.159      -0.065   │
           │ -0.159       0.183      -0.024   │
           │ -0.065      -0.024       0.089   │
           └──────────────────────────────────┘
```

#### From Jacobian to Efficient Backward Formula

The backward pass receives grad_output (dL/ds) and needs dL/dx.

By chain rule: dL/dx_i = sum_j (ds_j/dx_i * dL/ds_j)

That's a Jacobian-vector product. But computing the full Jacobian is O(n^2). We can simplify.

Substitute our formula for ds_j/dx_i:

```
dL/dx_i = ds_i/dx_i * dL/ds_i  +  sum_{j!=i} ds_j/dx_i * dL/ds_j

        = s_i*(1-s_i) * dL/ds_i  +  sum_{j!=i} (-s_j*s_i) * dL/ds_j
```

Expand the first term:

```
        = s_i * dL/ds_i  -  s_i*s_i * dL/ds_i  -  s_i * sum_{j!=i} s_j * dL/ds_j
```

Factor out s_i:

```
        = s_i * [ dL/ds_i  -  s_i * dL/ds_i  -  sum_{j!=i} s_j * dL/ds_j ]

        = s_i * [ dL/ds_i  -  (s_i * dL/ds_i  +  sum_{j!=i} s_j * dL/ds_j) ]
```

The parenthesized expression is the sum over ALL j (including i):

```
        = s_i * [ dL/ds_i  -  sum_j (s_j * dL/ds_j) ]
```

That sum is just the dot product of s and grad_output -- a single number!

```
dot = sum(s * grad_output)                  -- one number, computed once

dL/dx_i = s_i * (grad_output_i - dot)       -- for each element i
```

In vector form:

```
dL/dx = s * (grad_output - dot)
```

No Jacobian matrix needed. O(n) instead of O(n^2).

### Numerical Backward Example

```
s = [0.659, 0.242, 0.099]               (softmax output from forward)
grad_output = [1.0, 0.0, 0.0]           (pretend loss only cares about first element)

dot = sum(s * grad_output)
    = 0.659*1.0 + 0.242*0.0 + 0.099*0.0
    = 0.659

dL/dx[0] = 0.659 * (1.0 - 0.659) = 0.659 * 0.341  =  0.225   ✓
dL/dx[1] = 0.242 * (0.0 - 0.659) = 0.242 * (-0.659) = -0.159  ✓
dL/dx[2] = 0.099 * (0.0 - 0.659) = 0.099 * (-0.659) = -0.065  ✓

dL/dx = [0.225, -0.159, -0.065]
```

The gradients sum to approximately 0. This makes sense: since softmax outputs always sum to 1,
pushing one probability up must push others down.

### Alternatives to Softmax

| Method | Outputs sum to 1? | Exact zeros? | Differentiable? | Use case |
|--------|---|---|---|---|
| **Softmax** | Yes | No (approaches 0) | Yes | Single-choice: "which ONE token?" |
| Hardmax | Yes | Yes | **No** | Inference only (can't backprop) |
| Sigmoid | **No** | No | Yes | Multi-label: "which items apply?" |
| Sparsemax | Yes | Yes | Yes | Sparse attention |
| Gumbel-Softmax | Approximately | No | Yes | Discrete choices with gradients |
| Log-Softmax | N/A (log space) | N/A | Yes | Numerically stable loss |

**Hardmax (argmax)**: picks the biggest, outputs [1, 0, 0]. Used during inference (just give
me the best token). NOT usable for training -- zero gradient everywhere (step function).
Softmax is the "soft" differentiable version of hardmax.

**Sigmoid**: `1 / (1 + e^(-x))`. Maps each element independently to (0, 1), but outputs
do NOT sum to 1. Used for multi-label classification ("this image has a cat AND a dog" --
multiple things can be true). Not suitable for "pick one token" where we need exactly one answer.

**Sparsemax**: like softmax but can produce exact zeros. Useful for sharp attention (attend to
only a few positions). More expensive to compute. Some Transformer variants use it.

**Gumbel-Softmax**: adds random noise to approximate hard choices while staying differentiable.
Not used in standard Transformers.

**Log-Softmax**: `log(softmax(x)) = x - log(sum(e^x))`. Numerically more stable when
combined with cross-entropy loss. PyTorch recommends this over separate softmax + log.

**Why softmax won for Transformers**: attention needs weights that are (a) between 0 and 1,
(b) sum to 1 (distribute 100% of attention), (c) differentiable. Softmax checks all three.

### Temperature: Controlling Sharpness

Dividing inputs by a temperature T before softmax controls how "peaked" the output is:

```
softmax(x / T)

T = 1.0 (normal):   [2.0, 1.0, 0.1] --> [0.659, 0.242, 0.099]  (spread out)
T = 0.5 (sharp):    [4.0, 2.0, 0.2] --> [0.844, 0.114, 0.042]  (concentrated)
T = 2.0 (smooth):   [1.0, 0.5, 0.05]--> [0.423, 0.323, 0.254]  (more uniform)
```

- **Low temperature** = sharper, more confident (approaches hardmax)
- **High temperature** = smoother, more spread out (approaches uniform distribution)
- The `sqrt(d_k)` in attention acts as a temperature that normalizes for dimension size,
  so softmax behaves well regardless of how big d_k is. This is exactly why the paper
  scales by `1/sqrt(d_k)` in Equation 1.

---

## 1c. ReLU (Rectified Linear Unit)

The simplest activation function. Applied after a Linear layer to introduce non-linearity.

### What is an Activation Function?

An activation function is applied after a Linear layer. It decides how much signal passes
through and introduces non-linearity.

**The neuron analogy**: biological neurons receive signals, sum them, and "fire" if the total
exceeds a threshold. An artificial neuron does: Linear(xW + b) then Activation(decides how
strongly to fire).

**Why non-linearity is essential**: without it, stacking Linear layers is pointless:

```
Layer 1:  y = xW1 + b1
Layer 2:  z = yW2 + b2 = (xW1+b1)W2+b2 = x(W1*W2)+(b1*W2+b2) = xW3+b3

Collapse! Two layers = one layer. The network can't learn complex patterns.
```

Activation functions bend and fold the space between layers:
- **Linear layer**: rotates and stretches the data
- **Activation function**: bends and folds the data
- **Together**: the network can mold any shape

### The ReLU Equation

```
ReLU(x) = max(0, x)

If positive: keep it.
If negative: make it zero.
```

### Numerical Example

```
Input:  x = [0.5, -1.2, 3.0, -0.1, 0.0, 2.7]
Output: y = [0.5,  0.0, 3.0,  0.0, 0.0, 2.7]
              ↑     ↑    ↑     ↑    ↑    ↑
            keep  zero  keep  zero zero  keep
```

### Where ReLU Appears in the Transformer

One place only: inside the FFN (Paper Section 3.3, Equation 2):

```
FFN(x) = max(0, xW1 + b1) W2 + b2
          ^^^
          ReLU here, between the two Linear layers
```

### Backward Pass

The simplest backward of all primitives -- just a mask:

```
If x > 0:   gradient passes through (multiply by 1)
If x <= 0:  gradient is blocked     (multiply by 0)
```

### Numerical Backward Example

```
Forward:
  x = [0.5, -1.2, 3.0, -0.1, 0.0, 2.7]
  y = [0.5,  0.0, 3.0,  0.0, 0.0, 2.7]

Backward:
  grad_output = [0.3, -0.5, 0.1, 0.8, 0.2, -0.4]
  mask =        [  1,    0,   1,   0,   0,    1]     (1 where x > 0, else 0)

  dL/dx = grad_output * mask
        = [0.3,  0.0, 0.1, 0.0, 0.0, -0.4]
```

Gradients flow where input was positive, blocked where negative. Like a gate.

### The "Dying ReLU" Problem

If a neuron's output is always negative (bad initialization or large learning rate), its
gradient is always zero. It never updates. It's "dead" forever.

Rarely a problem in Transformers because:
- Residual connections provide alternative gradient paths
- FFN inner dimension (d_ff=64) is large -- even if some neurons die, plenty remain

### All Activation Functions Compared

| Function | Equation | Range | Smooth? | Cost | Dead neurons? |
|----------|----------|-------|---------|------|---------------|
| Sigmoid | 1/(1+e^(-x)) | (0, 1) | Yes | Medium | No, but vanishing gradient |
| Tanh | (e^x-e^(-x))/(e^x+e^(-x)) | (-1, 1) | Yes | Medium | No, but vanishing gradient |
| **ReLU** | **max(0, x)** | **[0, inf)** | **No** | **Cheapest** | **Yes** |
| Leaky ReLU | max(0.01x, x) | (-inf, inf) | No | Cheap | No |
| PReLU | max(ax, x), a learned | (-inf, inf) | No | Cheap | No |
| ELU | x if x>0, a(e^x-1) if x<=0 | (-a, inf) | Yes | Medium | No |
| GELU | x * Phi(x) | ≈[0, inf) | Yes | Medium | No |
| Swish | x * sigmoid(x) | ≈(-0.28, inf) | Yes | Medium | No |

**Sigmoid**: S-curve mapping to (0,1). Used for binary classification and LSTM gates.
Problem: gradients vanish for large inputs -- learning stalls in deep networks.

**Tanh**: like sigmoid but centered at zero (-1 to 1). Used in RNNs/LSTMs.
Same vanishing gradient problem at extremes.

**Leaky ReLU**: fixes dying ReLU by letting a small gradient (0.01) through for negatives.

**PReLU**: like Leaky ReLU but the negative slope is LEARNED, not hardcoded.

**ELU**: smooth at zero (unlike ReLU's sharp corner), pushes mean activation toward zero.

**GELU**: smooth ReLU-like curve. Empirically works better than ReLU for Transformers.
Used in BERT, GPT-2, GPT-3, most modern Transformers.

**Swish**: x * sigmoid(x). Non-monotonic. Used in EfficientNet and some Transformer variants.

### How to Choose an Activation Function

**By task type (output layer)**:
- Binary yes/no --> Sigmoid
- Pick one of N --> Softmax
- Multiple labels --> Sigmoid per label
- Hidden layers --> ReLU, GELU, or Swish

**By gradient flow (hidden layers)**:
- Sigmoid/Tanh: gradients shrink in deep networks (vanishing gradient)
- ReLU: gradients survive (=1) but can die (=0)
- GELU/Swish: best of both -- gradients flow well AND no dead neurons

**By compute cost**:
- Cheapest: ReLU, Leaky ReLU (just a comparison)
- Medium: GELU, Swish, ELU (need e^x or tanh)

**Historical trend in Transformers**:
- Original Transformer (2017): ReLU
- BERT (2018): GELU
- GPT-2, GPT-3 (2019-2020): GELU
- LLaMA, modern LLMs (2023+): SwiGLU (variant of Swish + gating)

**Our choice**: ReLU -- matches the original paper, simplest to implement, trivial backward
pass. For our tiny model the difference is negligible. Swapping to GELU later is one line.

---

## 1d. Layer Normalization

Normalizes activations to keep them in a well-behaved range. Appears after every sub-layer
in both the encoder and decoder (Paper Section 3.1, referencing Ba et al. 2016 [1]).

### The Problem It Solves

As data flows through many layers, numbers can drift wildly:

```
Layer 1 output:   values in range [-1, 1]          (reasonable)
Layer 5 output:   values in range [-50, 50]        (growing)
Layer 10 output:  values in range [-5000, 5000]    (out of control)
```

A learning rate that works for Layer 1 is way too big for Layer 10. Training diverges.
LayerNorm keeps activations stable at every step.

### The Equation

```
LayerNorm(x) = gamma * (x - mean) / sqrt(variance + epsilon) + beta
```

- **mean**: average of x across features (the last dimension)
- **variance**: variance of x across features
- **epsilon**: tiny number (1e-5) to prevent division by zero
- **gamma**: learnable scale parameter, shape (d_model,), initialized to 1
- **beta**: learnable shift parameter, shape (d_model,), initialized to 0

### Numerical Forward Example

```
x = [4.0, 2.0, 0.0]       (one position's features, d_model=3 for simplicity)

Step 1 -- compute mean:
  mean = (4.0 + 2.0 + 0.0) / 3 = 2.0

Step 2 -- subtract mean (center at zero):
  x - mean = [2.0, 0.0, -2.0]

Step 3 -- compute variance:
  variance = ((2.0)^2 + (0.0)^2 + (-2.0)^2) / 3
           = (4 + 0 + 4) / 3 = 2.667

Step 4 -- divide by standard deviation:
  x_norm = [2.0, 0.0, -2.0] / sqrt(2.667 + 1e-5)
         = [2.0, 0.0, -2.0] / 1.633
         = [1.225, 0.0, -1.225]

Step 5 -- scale and shift (gamma=1, beta=0 initially):
  output = 1 * [1.225, 0.0, -1.225] + 0
         = [1.225, 0.0, -1.225]

Check: mean ≈ 0  ✓   variance ≈ 1  ✓
```

The numbers are now well-behaved regardless of what the previous layer produced.

### Why Gamma and Beta?

We forced mean=0 and var=1, but what if the next layer NEEDS a different distribution?

Gamma and beta give back that freedom. They're learnable -- the network can learn to undo
the normalization if needed. Initially gamma=1, beta=0 (identity), but training adjusts them.

### Where LayerNorm Appears in the Transformer

After EVERY sub-layer (Paper Section 3.1):

```
Encoder layer:
  x = LayerNorm(x + SelfAttention(x))        <-- here
  x = LayerNorm(x + FFN(x))                  <-- and here

Decoder layer:
  x = LayerNorm(x + MaskedSelfAttention(x))  <-- here
  x = LayerNorm(x + CrossAttention(x))       <-- here
  x = LayerNorm(x + FFN(x))                  <-- and here
```

Encoder: 2 sub-layers x 2 layers = 4 LayerNorms
Decoder: 3 sub-layers x 2 layers = 6 LayerNorms
Total: 10 LayerNorms in our model

### Post-Norm vs. Pre-Norm: Where to Put LayerNorm

The placement of LayerNorm has evolved since the original paper.

#### Post-Norm (what the 2017 paper does):

```
output = LayerNorm(x + Sublayer(x))

  x --> Sublayer(x) --> Add(x + result) --> LayerNorm --> output
    \__________________↗
       residual connection
```

- Original design from the paper
- Harder to train in deep models
- Requires careful warmup schedule (paper Section 5.3)
- Gradient must pass through LayerNorm on the main path

#### Pre-Norm (modern default -- GPT-2, GPT-3, LLaMA):

```
output = x + Sublayer(LayerNorm(x))

  x --> LayerNorm --> Sublayer(x) --> Add(x + result) --> output
    \______________________________________↗
             residual connection (clean, direct path)
```

- Much easier to train -- stable without warmup tricks
- Clean gradient highway through the residual connection
- Gradient can flow directly through the addition without LayerNorm
- Used by most modern Transformers

The key difference: in Pre-Norm, the residual path is CLEAN. The gradient has a direct,
unobstructed route from output back to input. In Post-Norm, the gradient must pass through
LayerNorm, which can distort gradient magnitudes in deep models.

### Can You Remove LayerNorm Entirely?

The model won't train well. Without normalization:

```
Layer 1 output:   range [-1, 1]
Layer 2 output:   range [-5, 5]        (growing)
Layer 3 output:   range [-50, 50]      (exploding)
...softmax gets inputs of 1000+  -->   NaN
```

Activations blow up, softmax saturates, gradients become zero or infinity. Training collapses.
LayerNorm is essential glue.

### Layer Norm vs. Batch Norm

```
Input shape: (batch=32, seq_len=5, d_model=32)

Batch Norm: for each feature, compute mean/var across all 32 SAMPLES
            --> depends on batch size
            --> different behavior at training vs. inference
            --> unreliable for variable-length sequences

Layer Norm: for each sample at each position, compute mean/var across 32 FEATURES
            --> independent of batch size
            --> same behavior at training and inference
            --> each position normalized independently
```

The Transformer uses Layer Norm because sequences have variable lengths, each position
should be normalized independently, and behavior should be identical at train and inference.

### RMSNorm: The Simplified LayerNorm

RMSNorm (Root Mean Square Normalization) skips the mean subtraction step. The insight:
maybe we don't need to center the data at zero -- just controlling the scale is enough.

**LayerNorm (5 steps):**

```
1. Compute mean:        mean = sum(x) / n
2. Subtract mean:       x_centered = x - mean
3. Compute variance:    var = sum(x_centered^2) / n
4. Divide by std:       x_norm = x_centered / sqrt(var + eps)
5. Scale and shift:     output = gamma * x_norm + beta
```

**RMSNorm (3 steps):**

```
1. Compute RMS:    rms = sqrt(sum(x^2) / n)
2. Divide by RMS:  x_norm = x / rms
3. Scale only:     output = gamma * x_norm          (no beta!)
```

#### Numerical Comparison

```
x = [4.0, 2.0, 0.0]

LayerNorm:
  mean = 2.0
  x - mean = [2.0, 0.0, -2.0]
  var = 2.667
  x_norm = [1.225, 0.0, -1.225]       (centered AND scaled)

RMSNorm:
  rms = sqrt((16 + 4 + 0) / 3) = sqrt(6.667) = 2.582
  x_norm = [4.0/2.582, 2.0/2.582, 0.0/2.582]
         = [1.549, 0.775, 0.0]        (scaled only, NOT centered)
```

RMSNorm doesn't center at zero -- output still has a positive bias. But in practice this
doesn't hurt quality.

#### Why Use RMSNorm?

```
LayerNorm:  mean, variance, subtraction, division  -->  ~5 operations
RMSNorm:    sum of squares, division                -->  ~3 operations (40% less)
```

For huge models with billions of parameters, that savings adds up. LLaMA uses RMSNorm
throughout and found no loss in quality compared to LayerNorm.

RMSNorm's backward pass is also simpler -- no gradient through mean subtraction.

### DeepNorm: Normalization for Extreme Depth

DeepNorm solves training instability in very deep Transformers (100+ layers, even 1000+).

#### The Problem

Even with Pre-Norm, very deep models are unstable:

```
2 layers:     trains fine
6 layers:     trains fine (original paper)
24 layers:    needs careful tuning
100 layers:   very hard to train
1000 layers:  diverges
```

Residual connections accumulate contributions from every layer. After 1000 layers, the
residual stream is the sum of 1000 sub-layer outputs. The scale grows unbounded.

#### How DeepNorm Works

**Trick 1 -- Amplify the residual path:**

```
Normal:    output = LayerNorm(x + Sublayer(x))
DeepNorm:  output = LayerNorm(alpha * x + Sublayer(x))     (alpha > 1)
```

Where alpha = (2N)^0.25 for N layers. The identity path (x) is stronger, so the model
starts closer to an identity function and gradually learns to deviate.

**Trick 2 -- Shrink initial weights:**

```
Normal init:    W ~ Normal(0, 1/sqrt(d_model))
DeepNorm init:  W ~ Normal(0, beta/sqrt(d_model))     (beta < 1)
```

Initial sub-layer outputs are tiny. Combined with amplified residual, very stable at init.

#### Numerical Intuition

```
Normal residual (100 layers, each contributing ~1.0):
  After 100 layers: x ≈ 0 + 1 + 1 + ... + 1 = ~100   (too large!)

DeepNorm (alpha=3, sub-layers start tiny ~0.01):
  After 100 layers: gradual controlled growth            (stable)
```

The paper "DeepNet: Scaling Transformers to 1,000 Layers" (Microsoft, 2022) showed it could
train a 1000-layer Transformer that was previously impossible.

### The Normalization Family Tree

```
                    Normalization
                    /          \
            Batch Norm      Layer Norm
            (2015)          (2016)
            |               /      \
            |          RMSNorm    DeepNorm
            |          (2019)     (2022)
            |
      (not used in Transformers --
       depends on batch size)
```

### When to Use What

```
Standard Transformer (our project):    LayerNorm   (matches the paper)
Modern LLMs (LLaMA, Gemma):            RMSNorm     (faster, same quality)
Very deep models (100+ layers):         DeepNorm    (stability for extreme depth)
CNNs (image models):                    BatchNorm   (different domain entirely)
```

### Backward Pass

LayerNorm has two learnable parameters (gamma, beta) and we need gradients for the input.
This is one of the harder backwards because mean and variance depend on all inputs.

Given x_norm = (x - mean) / sqrt(var + eps), and output = gamma * x_norm + beta:

```
Step 1 -- gradients for learnable parameters:
  dL/dgamma = sum(grad_output * x_norm)     (across batch and positions)
  dL/dbeta  = sum(grad_output)              (across batch and positions)

Step 2 -- gradient through scale:
  dL/dx_norm = grad_output * gamma

Step 3 -- gradient through variance:
  dL/dvar = sum(dL/dx_norm * (x - mean)) * -0.5 * (var + eps)^(-1.5)

Step 4 -- gradient through mean:
  dL/dmean = sum(dL/dx_norm) * -1/sqrt(var + eps)
           + dL/dvar * sum(-2*(x - mean)) / n

Step 5 -- gradient through input:
  dL/dx = dL/dx_norm / sqrt(var + eps)
        + dL/dvar * 2*(x - mean) / n
        + dL/dmean / n
```

Looks complex, but each step reuses cached values from the forward pass. The key is that
both mean and variance are FUNCTIONS of x, so gradients must flow through both paths.

### Our Choice

We implement Post-Norm LayerNorm matching the original paper exactly. Our model has only
2 layers, so we don't need DeepNorm. Understanding LayerNorm makes RMSNorm trivial later
(just remove the mean subtraction step).

---

## 1e. Dropout

A regularization technique that prevents overfitting by randomly silencing neurons during
training. (Paper Section 5.4, referencing Srivastava et al. 2014 [33])

> **NOTE**: This concept may feel abstract now. It will click during Step 8 (Training) when
> we see the concrete difference between dropout=0 (overfitting) and dropout=0.1 (generalizing).
> Revisit this section after training the model.

### The Core Idea

During training, randomly zero out neurons with probability p. Each forward pass uses a
different random subset of the network.

```
Without dropout:   [0.5, -1.2, 3.0, -0.1, 2.7]    (all neurons active)

With dropout (p=0.3, 30% chance of dropping each):
  Random mask:     [ 1,    0,   1,    1,   0  ]    (0 = dropped)
  Result:          [0.5, 0.0, 3.0, -0.1, 0.0]     (some neurons silenced)
```

Every training step, a DIFFERENT random set gets silenced. The network can never rely on
any single neuron -- it must spread knowledge across many neurons.

### Analogy

A team of 10 people on a project. If the same 3 do all the work, the team is fragile (one
gets sick, project fails). Dropout randomly sends different people home each day, forcing
cross-training. Everyone must be capable. The result is a more resilient team.

### Inverted Dropout: The Scale Fix

If 30% of neurons are zeroed during training but all are active at inference, the total
magnitude is different:

```
Training:   7 of 10 active  -->  smaller sum
Inference:  10 of 10 active -->  bigger sum (43% bigger!)
```

Fix: scale UP surviving neurons during training by 1/(1-p):

```
p = 0.3  -->  scale = 1/(1-0.3) = 1/0.7 = 1.429

Training:
  Input:      [0.5, -1.2, 3.0, -0.1, 2.7]
  Mask:       [ 1,    0,   1,    1,   0  ]
  After mask: [0.5,  0.0, 3.0, -0.1, 0.0]
  Scale up:   [0.71, 0.0, 4.29,-0.14, 0.0]    (* 1.429)

Inference:
  Input:      [0.5, -1.2, 3.0, -0.1, 2.7]     (no dropout, use as-is)
```

Expected values stay the same between training and inference. No adjustment at inference.

### Training vs. Inference (Critical Distinction)

```
Training mode:   randomly zero out + scale survivors
Inference mode:  do NOTHING -- pass input through unchanged
```

Our layers need a `training` flag. During inference we want the full network.

### Where Dropout Appears in the Transformer

Three places (Paper Section 5.4):

```
1. After each sub-layer output, before residual addition:
     x = LayerNorm(x + Dropout(Sublayer(x)))

2. After adding positional encoding to embeddings (both encoder and decoder):
     x = Dropout(TokenEmbedding(x) + PositionalEncoding(x))

3. Inside attention weights (optional, common in practice, not in original paper)
```

The paper uses P_drop = 0.1 (drop 10%). We use the same.

### Backward Pass

Same mask, same scaling -- the simplest backward:

```
Forward:   output = (x * mask) / (1 - p)
Backward:  dL/dx  = (grad_output * mask) / (1 - p)
```

Dropped neurons get zero gradient (didn't contribute, shouldn't be updated).
Active neurons get scaled gradient.

### Numerical Backward Example

```
Forward (p = 0.3):
  x =           [0.5, -1.2, 3.0, -0.1, 2.7]
  mask =        [ 1,    0,   1,    1,   0  ]
  output =      [0.71, 0.0, 4.29,-0.14, 0.0]

Backward:
  grad_output = [0.3, -0.5, 0.1,  0.8, 0.2]
  dL/dx = grad_output * mask / (1 - p)
        = [0.3*1, -0.5*0, 0.1*1, 0.8*1, 0.2*0] / 0.7
        = [0.429,  0.0,   0.143, 1.143, 0.0]
```

### Does Dropout Always Help?

```
Small model, lots of data:    may not be needed (not overfitting anyway)
Large model, little data:     very helpful (prevents memorization)
Very large models (GPT-3+):   some use it, some don't
Our toy model:                p=0.1 matches paper, mild regularization
```

---

## 1f. Embedding

Converts token IDs (integers) into vectors that the network can do math with.
(Paper Section 3.4)

### The Problem

The Transformer works with vectors and matrices. But our input is just integers:

```
Input: [3, 1, 4, 1, 5]     -- five integers, not vectors
```

How does the network turn the number 3 into something it can multiply, add, and attend over?

### What an Embedding Does

It's a **lookup table**. Each token ID maps to a vector of d_model numbers:

```
Embedding matrix E, shape (vocab_size, d_model) = (13, 32):

       dim0    dim1    dim2   ...  dim31
  0: [ 0.12, -0.84,  0.33,  ...,  0.56]    <-- vector for digit "0"
  1: [-0.21,  0.67, -0.45,  ...,  0.11]    <-- vector for digit "1"
  2: [ 0.88,  0.02,  0.71,  ..., -0.39]    <-- vector for digit "2"
  3: [-0.54,  0.91, -0.12,  ...,  0.43]    <-- vector for digit "3"
  ...
 12: [ 0.33, -0.71,  0.15,  ...,  0.62]    <-- vector for EOS
```

Forward pass: just pick the rows corresponding to the input token IDs.

### Numerical Example

```
Embedding matrix E, shape (5, 4):    (5 tokens, 4 dims for simplicity)

         dim0   dim1   dim2   dim3
  0:   [ 0.12, -0.84,  0.33,  0.56]
  1:   [-0.21,  0.67, -0.45,  0.11]
  2:   [ 0.88,  0.02,  0.71, -0.39]
  3:   [-0.54,  0.91, -0.12,  0.43]
  4:   [ 0.33, -0.15,  0.62,  0.77]

Input token IDs: [3, 1, 4]

Output: pick rows 3, 1, 4:
  token 3 --> [-0.54,  0.91, -0.12,  0.43]
  token 1 --> [-0.21,  0.67, -0.45,  0.11]
  token 4 --> [ 0.33, -0.15,  0.62,  0.77]

Output shape: (3, 4)    -- 3 tokens, each a 4-dim vector
```

No multiplication, no addition -- just row lookups.

### What Do These Numbers Mean?

Initially, the embedding vectors are RANDOM. Token 3's vector [-0.54, 0.91, ...] means
nothing at first.

During training, the network adjusts these vectors so that:
- Tokens that behave similarly --> similar vectors (close in space)
- Tokens that behave differently --> different vectors (far apart)

For sorting, the network might learn that 1 and 2 are "small numbers" (nearby vectors)
while 8 and 9 are "large numbers" (nearby to each other, far from 1 and 2).

### Do Random Initial Vectors Collide?

No. In 32 dimensions with only 13 tokens, collisions are essentially impossible.

In 1D: 13 points on a number line -- crowded, likely collisions.
In 2D: 13 points on a plane -- much more room.
In 32D: 13 points in 32-dimensional space -- astronomically far apart.

Probability of two random 32-dim vectors being close in ALL dimensions: ~(0.1)^32 ≈ 0.
With only 13 tokens in 32 dimensions, each token has ~2.5 dimensions of "personal space."

Even if two tokens started similar by accident, training would push them apart if they
behave differently.

Rule of thumb: safe as long as d_model >> log2(vocab_size). For us: 32 >> 3.7 ✓

### The Paper's Scaling Trick (Section 3.4)

The paper multiplies embeddings by sqrt(d_model):

```
output = Embedding(token_ids) * sqrt(d_model)
```

Why? Embeddings are initialized with small random values (variance ~1/d_model). Positional
encodings (added next) have variance ~1. Without scaling, positional encoding would dominate.
Multiplying by sqrt(d_model) = sqrt(32) ≈ 5.66 brings embedding scale to match PE scale.

### Backward Pass

No matrix multiply in forward, so backward is a scatter operation:

```
Forward:  picked rows [3, 1, 4] from embedding matrix

Backward:
  Row 3 gets gradient from position 0
  Row 1 gets gradient from position 1
  Row 4 gets gradient from position 2
  All other rows (0, 2, 5-12) get zero gradient (weren't used)
```

If the same token appears twice:

```
Input: [3, 1, 3]     -- token 3 at positions 0 and 2

Backward: row 3 gets gradients from BOTH positions, summed:
  grad_E[3] = grad_from_position_0 + grad_from_position_2
```

Gradients accumulate. If a token was used twice, it contributed to the loss twice.

### Token IDs Are Just Addresses

Token IDs never participate in any math. They're array indices -- nothing more:

```
Token ID 3 does NOT mean "the value three"
Token ID 3 means "go to row 3 of the embedding table and grab that vector"
```

The ID is an address. The vector at that address is the actual data:

```
Token ID 3  -->  Embedding[3]  -->  [-0.54, 0.91, -0.12, ..., 0.43]
                  ↑                  ↑
              address/index        actual data that flows through the model
              (never used           (this gets multiplied, added, attended)
               in any math)
```

After lookup, the token ID is GONE. The Transformer never sees it again. Token IDs only
appear at the very beginning (input) and the very end (argmax over output probabilities
to pick a predicted token).

### How Large Is the Embedding Table?

```
Size = vocab_size * d_model * 4 bytes (float32)

Our model:   13 * 32 * 4       =        1,664 bytes =   1.6 KB
Transformer: 37,000 * 512 * 4  =   75,776,000 bytes =  72 MB
GPT-2:       50,257 * 768 * 4  =  154,389,504 bytes = 147 MB
LLaMA-7B:    32,000 * 4096 * 4 =  524,288,000 bytes = 500 MB
GPT-3:       50,257 * 12288 * 4= 2.47 billion bytes = 2.3 GB
```

GPT-3's embedding table is 2.3 GB, but that's only ~1.4% of its total parameters.
Most parameters are in the Linear layers of attention and FFN.

The table is loaded into memory once. During a forward pass, only rows for the current
batch's tokens are accessed (sparse access).

### Which Rows Get Updated During Training?

Only rows for tokens that APPEARED in the batch get gradients:

```
Batch 1: "the cat sat on the mat"
  "the" (row 456) --> gets gradient (appeared twice, gradients summed)
  "cat" (row 892) --> gets gradient
  "sat" (row 1203) --> gets gradient
  All other 50,252 rows: ZERO gradient, NOT updated
  
Batch 2: "the dog is on the bed"
  "the" (row 456) --> gets gradient AGAIN
  "dog" (row 1567) --> gets gradient
  "is" (row 89) --> gets gradient
```

Common tokens like "the" appear in almost every batch --> updated almost every step.
Rare tokens like "defenestration" appear once in the whole dataset --> updated once total.

```
Token              Frequency        Updated how often?
─────              ─────────        ──────────────────
"the"              everywhere       almost every batch --> very well-trained
"is"               very common      almost every batch --> very well-trained
"transformer"      moderate         every few batches  --> decently trained
"defenestration"   extremely rare   maybe once total   --> poorly trained
```

### "If the Embedding Gets Updated, Isn't It No Longer 'the'?"

This is a key insight: the embedding vector IS the model's definition of "the". There is no
external fixed definition. When the vector changes, the model's understanding improves.

```
Step 1:     "the" = [-0.54, 0.91, ...]   Random -- means nothing
Step 100:   "the" = [0.23, -0.11, ...]    Learning: common word before nouns
Step 1000:  "the" = [0.31, -0.08, ...]    Knows: indicates a specific noun follows
Step 5000:  "the" = [0.32, -0.07, ...]    Refined understanding
Step 10000: "the" = [0.32, -0.07, ...]    Converged -- barely changes anymore
```

**Analogy: learning a foreign language.** Your understanding of "すみません" (sumimasen)
evolves:
- Day 1: you think it means "sorry"
- Day 30: you learn it also means "excuse me"
- Day 90: you realize it's used to get attention ("hey")
- Day 180: you understand the full nuance -- a social lubricant

Did "すみません" stop being "すみません" because your understanding changed? No. Your
representation got richer and more accurate. Same with embeddings.

### What Drives the Updates?

Updates are guided by the loss, not random:

```
Model sees: "the cat sat on the [???]"
            If model predicts "mat" correctly:
              --> small loss, tiny gradient, "the" barely moves
              
            If model predicts "dog" instead of "mat":
              --> big loss, larger gradient
              --> "the" gets pushed toward a representation that
                  would have helped predict "mat"
```

Every update says: "adjust so the model would have predicted better." The embedding
moves TOWARD being more useful, not away.

### Conflicting Pulls Converge to Meaning

"the" appears in millions of different contexts:

```
"the cat"        --> pulls "the" toward "precedes animals"
"the algorithm"  --> pulls "the" toward "precedes technical terms"
"the big red"    --> pulls "the" toward "precedes adjectives"
"the end"        --> pulls "the" toward "precedes abstract nouns"
```

Each pull is slightly different. But the AVERAGE across millions of examples converges
to a vector encoding what all contexts share:

```
"the" ≈ "a specific thing follows"
```

The 32+ dimensions encode this nuanced meaning. Some dimensions might capture
"precedes a noun," others "grammatically important," others "extremely common word."

For our sort task, "3" appears in many contexts and converges to:
```
"3" ≈ "I am the number three. Larger than 0,1,2. Smaller than 4,5,6,7,8,9."
```

Nobody told the model what 3 means. It discovered this from examples and loss signals.

### The Embedding Is Not a Dictionary

```
The embedding table is NOT a dictionary with fixed definitions.
It's a LEARNED representation that starts random and gradually
becomes meaningful through training.

Updating it doesn't break it -- updating IS the learning.
Updates stop being significant once the model has learned,
because loss becomes small and gradients become tiny.
```

### How Do We Know a Vector Is "the"? (And Inference Vocab)

**The string is not read out of the vector.** The floats in a row do not spell "the" or
encode letters in a human-readable way. The link is by **table index**:

```
text "the"  -->  tokenizer  -->  token id (e.g. 456)  -->  Embedding[456]  -->  vector
                     ↑                    ↑                      ↑
              fixed mapping         same id every time      same row every time
              for this checkpoint   for this piece          (shared weights)
```

So **every time** the tokenizer emits id 456 for the piece meaning "the", the model
looks up **the same row** of the embedding matrix. There is **no** new random row per
sentence and **no** new token id for the same tokenizer output. What *can* differ
between two occurrences of "the" in the same sequence is the **position**: positional
encoding (or position-aware attention) is added on top of the **same** token embedding
at each position.

**Test / generation time (frozen weights):**

- The **vocabulary is fixed**: only ids in `0 .. vocab_size-1` are valid. The embedding
  table and the final logits layer are sized for exactly that set of token types.
- You are **not** adding new vocabulary entries at inference. If someone swaps in a
  tokenizer or vocab that does not match the checkpoint, ids and rows no longer align
  -- that is a setup bug, not normal behavior.
- **"New text" is not the same as "new vocab."** With **subword** (BPE, SentencePiece) or
  **byte-level** tokenization, a rare or unseen **word** is usually expressed as a
  **sequence of existing** subword or byte tokens. The model still only ever sees known
  ids; it does not grow the table on the fly.

**How generated text becomes words again:** The Transformer ends with scores over
every vocab id (logits). You pick an id (argmax or sampling). The tokenizer's **id →
subword/string** map turns that id back into text. The model does not "recognize" the
word by inspecting the embedding vector at decode time; it **chooses an id**, and the
tokenizer **labels** that id.

### Managing Large Embedding Tables in Production

For very large vocabularies, practical challenges and solutions:

**Rare token problem**: rare tokens don't get enough updates to converge.
Solutions:
- BPE tokenization: "defenestration" = ["def","en","est","ration"] -- each piece
  appears in other words too, so they all get trained well
- Weight tying (Section 3.4): same matrix for input and output embeddings. Tokens
  get signal as input ("I saw this") AND output ("I need to predict this")

**Memory**: GPT-3's embedding is 2.3 GB but only ~8% of rows are accessed per batch.

**Embedding factorization** (ALBERT): instead of (vocab, d_model), use two smaller
matrices (vocab, 128) then (128, d_model). 6x fewer parameters.

**Adaptive embeddings**: common tokens get full dimensions, rare tokens get fewer.

### Weight Sharing (Paper Section 3.4)

The paper shares the SAME weight matrix for three purposes:
1. Input embedding (encoder)
2. Output embedding (decoder)
3. Final linear projection (decoder output -> vocab logits)

This ties "what a token means as input" to "what a token means as output" and reduces
parameters. We'll implement this sharing.

### The Dimension Expansion

Embedding is where the data goes from 2D to 3D:

```
                                    Shape
                                    ─────
Raw input (token IDs):              (32, 5)            -- 2D: batch x sequence
        |
After embedding:                    (32, 5, 32)        -- 3D: batch x seq x d_model
        |
Stays 3D through ENTIRE model:     (32, 5, 32)
        |
Final projection:                   (32, 7, 13)        -- 3D: batch x seq x vocab_size
```

Once embedding adds the third dimension, it stays 3D all the way through. The shape is
always (batch_size, seq_len, features). Only the last dimension changes.

### Features = Embedding Dimensions = d_model

These are all the same thing, different names depending on context:

```
"embedding dimensions"  -- when talking about the Embedding layer
"features"              -- when talking about Linear, LayerNorm
"d_model"               -- when talking about the Transformer architecture
"hidden size"           -- in some frameworks (like HuggingFace)
"representation"        -- in academic papers
```

All refer to the same 32 numbers that describe one token at one position.

### How Features Evolve Through Layers

The embedding creates initial 32 features. Every layer transforms them:

```
After embedding:        [-0.54, 0.91, -0.12, ..., 0.77]   (raw)
                        "I am the number 3"

After encoder layer 1:  [0.33, -0.18, 0.95, ..., -0.42]   (enriched)
                        "I am 3, larger than the 1 next to me"

After encoder layer 2:  [1.21, 0.05, -0.67, ..., 0.89]    (fully processed)
                        "I am 3, I belong at position 3 in sorted output"
```

Same 32 numbers, but the MEANING gets richer layer by layer.

### Where Each Layer Operates on the 3D Tensor

```
Shape: (batch_size, seq_len, d_model) = (32, 5, 32)
             ↑          ↑        ↑
             |          |        └── Linear operates HERE (transforms features)
             |          |            LayerNorm normalizes HERE
             |          |
             |          └── Attention operates HERE (tokens attend to each other)
             |
             └── Batch dim: everything independent per sample
```

### Choosing d_model Size

```
Model                 d_model    Vocab     Task
─────                 ───────    ─────     ────
Our toy model            32       13       Sort 5 numbers
Original Transformer    512     37,000     Translation
BERT-base               768     30,522     Language understanding
GPT-2 small             768     50,257     Text generation
GPT-3                 12288     50,257     Text generation
LLaMA-7B              4096     32,000     Text generation
LLaMA-70B             8192     32,000     Text generation
```

Selection criteria:
1. Task complexity: simple=32-64, serious NLP=512-1024, LLM=2048-12288
2. Must be divisible by h (number of heads): 32/4=8 ✓
3. Prefer multiples of 64 for GPU efficiency
4. d_model >> log2(vocab_size)
5. Budget: total params ≈ 12 * N * d_model^2

d_model=32 is perfect for us: ~30K total params, trains in seconds, can trace by hand.

### Vocabulary: Definition and How It's Built

See [reference_tokenization.md](reference_tokenization.md) for full details on vocabulary
construction, BPE algorithm, and real-world examples.

Our vocabulary is simple and hand-defined:

```
ID    Token    Meaning
──    ─────    ───────
0-9   digits   the actual numbers to sort
10    PAD      padding for variable-length batches
11    SOS      Start Of Sequence (decoder begin signal)
12    EOS      End Of Sequence (decoder stop signal)
```

Vocab size = 13. The model can only read and produce these 13 tokens.

### Attention Heads: Definition

Each attention head is one independent attention computation. Multi-Head Attention runs
h=4 of these in parallel, each looking at the data from a different "perspective."

```
d_model=32 split across h=4 heads:
  Head 1 (d_k=8): might learn "attend to nearby positions"
  Head 2 (d_k=8): might learn "attend to smaller numbers"
  Head 3 (d_k=8): might learn "attend to duplicates"
  Head 4 (d_k=8): might learn "attend to sequence start"
```

Each head projects the full 32 dims into 8 dims via learned W^Q, W^K, W^V matrices,
computes attention in that 8-dim subspace, then all heads' outputs are concatenated back
to 32 dims. Covered fully in Step 4.

---

## Step 1 Summary

All six primitives documented:

| Layer | Status | Key Takeaway |
|-------|--------|-------------|
| Linear | Documented | y = xW + b, the workhorse (30+ in our model) |
| Softmax | Documented | logits -> probabilities, numerically stabilized |
| ReLU | Documented | max(0,x), introduces non-linearity between Linear layers |
| LayerNorm | Documented | Keeps activations stable, appears after every sub-layer |
| Dropout | Documented | Prevents overfitting by randomly silencing neurons |
| Embedding | Documented | Lookup table: token IDs -> d_model vectors |

Next: implement all six in code (nn/src/layers.py)

---

## Discussion Log

A chronological log of every conceptual exchange. Append, do not edit.

### 2026-04-28 -- Embeddings are learnable; dropout intuitions; fancy-indexing return type (Q&A from chat)

Three follow-up questions came up during a debug-session walk-through
of the toy training run. Each strengthens material that was implicit or
incomplete in the topical sections above. Captured here near-verbatim
so the reasoning isn't lost.

#### Q1: "Are embedding vectors learnable?"

**Yes, completely.** Embeddings are ordinary trainable parameters; the
`Embedding` class participates in the same three-stage lifecycle as
every other learnable layer in the codebase. The point is implicit in
section 1f and the `Embedding.params_and_grads` source code, but it
was not stated as a headline. Stating it here for clarity:

1. **Initialization.** `self.weight = np.random.randn(V, d_model) * 0.02`
   -- a small Gaussian init. At step 0, the embedding for "0" looks
   essentially the same as the embedding for "9": both are noise. The
   model has no idea yet that 0 and 9 are digits; it doesn't know
   anything about its tokens. The factor `0.02` is a transformer
   convention -- bigger inits explode early activations, smaller inits
   train glacially.
2. **Backward (gradient flow).** When the loss has been computed, the
   chain rule walks gradients all the way back to the embedding.
   `np.add.at(self.grad_weight, token_ids, grad_output)` scatters
   per-position `(B, n, d_model)` gradients back into the `(V, d_model)`
   table -- specifically, into the rows whose token ids appeared in
   the batch. Rows for tokens that didn't appear in the batch get zero
   gradient and don't move. The unbuffered scatter-add (vs naive `+=`)
   is essential because real batches contain repeated tokens; see
   `reference_numpy.md` Section 11 for the worked example of the
   silent bug naive `+=` causes.
3. **Optimizer step.** `params_and_grads()` exposes the embedding to
   the optimizer as a `(param, grad, name)` triple, structurally
   indistinguishable from a `Linear` weight or a `LayerNorm` gain.
   `Adam.step()` walks this list and updates `self.weight` in place
   on every training step. The optimizer does not know or care that
   it's an embedding; the abstraction is the whole point.

**Geometric interpretation (the deeper "why this matters").** After
training, the `(13, 32)` embedding table for our sort task has
rearranged itself into a layout where:

- adjacent digits (`weight[0]` and `weight[1]`) are *close* in the
  32-dim space,
- distant digits (`weight[0]` and `weight[9]`) are *far apart*,
- special tokens (`PAD`, `SOS`, `EOS`) end up in their own region,
  not interfering with the digit cluster.

The model learns this geometry purely from the gradient signal of "did
I sort correctly?" -- because attention is computed via dot products
between (projected) embeddings, and the only embedding layouts that
*make sorting tractable* are the ones where digit ordering is
encoded as geometric proximity. This is the fundamental insight from
the original Transformer's Section 3.4 and from Mikolov-style
distributed-representation work: even though the input is a discrete
integer, the model gets to choose a continuous geometric meaning for
it, and that geometry is what makes everything downstream work.

**Verification trick.** A growing `weight.std()` over training is a
fingerprint that this parameter is learning. At init it's `0.02` by
construction; after training a sort model it's typically `0.3`-`0.6`:

```python
import numpy as np
data = np.load("nn/numpy-transformer/train_model/sort_model.npz")
emb = data["params.encoder.embedding.weight"]
print(emb.std())   # ~0.02 at init -- ~0.3-0.6 after training
```

**"What if" experiments worth running later:**

- *Freeze the embeddings.* Set `grad_weight = None` after backward so
  they never update. Watch the model fail to learn, or learn much more
  slowly. This proves the embedding's contribution rather than just
  asserting it.
- *Hand-craft a 1D ramp init.* Set `weight[i] = [i, i, ..., i] * 0.1`
  for digits 0-9 so the geometry is given, not learned. Watch
  training converge dramatically faster. This isolates "discovering the
  ordering" from "learning to use the ordering."

#### Q2: "What is the purpose of dropout, deeper than the team analogy?"

Section 1e gives the mechanics (random zero, inverted-dropout
rescaling, train/eval flag) and the team analogy. The team analogy is
correct but covers only one of the three accepted explanations for
*why* dropout works. All three are simultaneously true:

1. **Don't co-adapt** (the team analogy). Without dropout, two neurons
   can develop a brittle partnership -- A says "I'll detect feature X
   only if B remembers to subtract C." Dropout breaks the partnership
   randomly, forcing each neuron to be useful *on its own*. Result:
   features are simpler, more local, more robust.
2. **Free ensemble.** A network with `N` dropoutable neurons can be
   viewed as `2^N` overlapping sub-networks (each sub-network is a
   random pruning). Every training step computes a gradient for one
   randomly chosen sub-network. After many steps, the weights settle
   at a value that works well for *all* sub-networks on average. At
   test time -- with dropout off -- the full network is approximately
   a geometric-mean ensemble over those sub-networks, and ensembles
   reliably generalize better than single models. This is the most
   rigorous justification, due to Hinton et al. 2012.
3. **Flat-minima bias.** Multiplicative noise during training nudges
   optimization toward weight settings that *don't care* about exact
   activation values -- i.e. flat minima. Flat minima generalize
   better than sharp minima (a deep result on its own). Dropout is one
   way to inject the noise that biases optimization in this direction.

#### Q3: "Where in our codebase does dropout appear?"

A useful x-ray of the architecture. Per paper Section 5.4, dropout
shows up at four kinds of sites:

- After the embedding-plus-positional-encoding sum (input dropout),
  in both the encoder and the decoder.
- After every encoder sub-layer's output (multi-head attention and
  feed-forward), before the residual + LayerNorm.
- After every decoder sub-layer's output (masked self-attention,
  cross-attention, and feed-forward), before the residual + LayerNorm.

Counted concretely for our toy build (2 encoder layers + 2 decoder
layers):

```text
input dropouts:                 1 (encoder) + 1 (decoder)              = 2
per-encoder-block (2 sublayers): 2 layers x 2 = 4
per-decoder-block (3 sublayers): 2 layers x 3 = 6
                                                            total      = 12
```

Twelve `Dropout` layer instances are walked through on a single
forward pass, each independently sampling a 10% mask. They share the
same `training` flag (propagated via `transformer.train(mode)`) but
not the same masks -- each is its own independent stochastic event.

#### Q4: "What does `self.weight[token_ids]` return -- a single array or a dict?"

A beginner-friendly NumPy clarification triggered by the user reading
`Embedding.forward`. `self.weight[token_ids]` is a NumPy *fancy
indexing* expression and returns a **single `ndarray`**, not a
dictionary or anything else. Specifically with `weight: (V, d_model)`
and `token_ids: (B, n)`, the output has shape `(B, n, d_model)`. The
general shape rule is captured in `reference_numpy.md` Section 11.

Worth distinguishing -- the line **above** the return,
`self._cache['token_ids'] = token_ids`, is what populates a dict.
That's the layer's per-call cache for backward. The return value of
`forward()` is unrelated to that dict; it's just the looked-up
`(B, n, d_model)` tensor.

#### Cross-references

- `Embedding.params_and_grads()` and `np.add.at` correctness: this
  doc Section 1f, plus `reference_numpy.md` Section 11.
- Adam updating embeddings via `params_and_grads()`:
  `08_training.md` Section 6.
- Tied input/output embeddings (where the same matrix receives
  *two* gradient contributions per backward): `07_decoder.md`
  Section 6.
- Dropout-off-at-eval determinism guarantee: tests
  `_dropout_off_at_eval_is_deterministic` in
  `tests/test_layers.py`.
