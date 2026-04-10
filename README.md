# Neural ODEs Project 1

This repository contains code and report for Project 1 of the Neural ODEs course. The network and training code is contained in the `neural_ode` directory, while experiments and visualizations are in `node.ipynb` notebook.

## Setup

See `requirements.txt` for the list of requirements.

## Task

Using a Neural ODE, reconstruct the function

$$
f:[-1,1] \to {0,1,2,3,4,5,6,7,8,9}
$$

where $f(x)$ is the first digit after the decimal point.

For example:
- $f(0.0014)=0$
- $f(-0.38)=3$
- $f(0.456)=4$

This can be done using the adjoint method or Double Dimension Differential Equations (D3E).

Please generate the training set yourself, with 200, 2000, and 20000 samples.

Visualizations are welcome. Please keep in mind that this cannot work in “one” dimension.