# ThermoPINN Theory

## Overview

ThermoPINN is a Physics-Informed Neural Network framework developed for solving heat transfer and thermal diffusion problems commonly encountered in semiconductor devices.

Unlike traditional numerical methods such as Finite Difference Methods (FDM) or Finite Element Methods (FEM), PINNs embed the governing physical equations directly into the neural network loss function.

The model learns temperature fields while satisfying the underlying physics.

---

## Governing Equations

### 1D Steady-State Heat Equation

d²T/dx² = 0

### 1D Heat Equation with Internal Heat Generation

d²T/dx² + Q/k = 0

### 1D Transient Heat Equation

∂T/∂t = α ∂²T/∂x²

### 2D Steady-State Heat Equation

T_xx + T_yy + Q(x,y)/k = 0

### 2D Transient Heat Equation

∂T/∂t = α(T_xx + T_yy) + S(x,y)

---

## Physics-Informed Neural Networks

A PINN approximates the solution field using a neural network:

Tθ(x)

Tθ(x,t)

Tθ(x,y)

Tθ(x,y,t)

Automatic differentiation is used to compute the derivatives required by the governing equations.

The PDE residual becomes part of the training objective.

Loss = PDE Loss + Boundary Loss + Initial Condition Loss

---

## Hard Constraint Enforcement

Boundary and initial conditions are embedded directly into the network architecture.

Example:

T(x,y,t)
=
T0
+
(1-e^{-t})
x(1-x)
y(1-y)
Nθ(x,y,t)

This guarantees:

T(x,y,0)=T0

and

T=T0

on all boundaries.

---

## Benchmarking

PINN solutions are validated against finite difference solutions.

Evaluation metrics include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Maximum Error
- Relative L2 Error
