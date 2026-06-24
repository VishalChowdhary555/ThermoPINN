# ThermoPINN
Physics-Informed Neural Network framework for heat transfer and thermal diffusion in semiconductor devices.
# ThermoPINN

Physics-Informed Neural Network Framework for Heat Transfer and Thermal Diffusion in Semiconductor Devices

---

## Overview

ThermoPINN is a modular Physics-Informed Neural Network (PINN) framework developed for solving heat transfer and thermal diffusion problems commonly encountered in semiconductor devices and microelectronic systems.

The framework combines deep learning, automatic differentiation, and governing heat-transfer equations to model temperature fields without relying solely on traditional numerical discretization techniques.

PINN predictions are benchmarked against analytical solutions and finite difference methods (FDM) to evaluate accuracy and physical consistency.

---

## Key Features

- Physics-Informed Neural Networks (PINNs)
- Automatic differentiation for PDE residual computation
- Steady-state and transient heat-transfer modeling
- Semiconductor hotspot simulation
- Hard enforcement of boundary and initial conditions
- Finite Difference Method (FDM) benchmarking
- Modular and extensible architecture
- GPU acceleration using PyTorch

---

## Implemented Models

### 1D Steady-State Heat Conduction

Governing Equation:

d²T/dx² = 0

Validation:
Analytical solution

---

### 1D Heat Conduction with Internal Heat Generation

Governing Equation:

d²T/dx² + Q/k = 0

Validation:
Analytical solution

---

### 1D Transient Heat Diffusion

Governing Equation:

∂T/∂t = α ∂²T/∂x²

Validation:
Analytical solution

---

### 2D Steady-State Semiconductor Hotspot

Governing Equation:

T_xx + T_yy + Q(x,y)/k = 0

Validation:
Finite Difference Method (FDM)

---

### 2D Transient Semiconductor Hotspot

Governing Equation:

∂T/∂t = α(T_xx + T_yy) + S(x,y)

Validation:
Finite Difference Method (FDM)

---

## Results

### 1D Heat Generation Benchmark

MAE ≈ 4 × 10⁻⁷

RMSE ≈ 3 × 10⁻⁶

---

### 1D Transient Heat Diffusion

MAE ≈ 6.7 × 10⁻⁴

RMSE ≈ 7.8 × 10⁻⁴

---

### 2D Transient Semiconductor Hotspot

At t = 1.0:

MAE ≈ 1 × 10⁻³ K

RMSE ≈ 1.6 × 10⁻³ K

Relative L2 Error ≈ 5 × 10⁻⁶

PINN and FDM solutions show excellent agreement.

---

## Repository Structure

```text
ThermoPINN/
│
├── notebooks/
├── src/
├── docs/
├── requirements.txt
└── README.md
