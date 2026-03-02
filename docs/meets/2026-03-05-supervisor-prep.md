# Preparation for Supervisor Meeting

*This document outlines key scientific and implementation questions to align on before fully executing the remaining experimental phase.*

## 1. The "Weak Baseline" Concern
Currently, we are training an MLP to perform regression on highly non-convex functions (Ackley, Levy).
*   **The Risk:** Standard MLPs naturally suffer from "spectral bias" (they cannot easily learn high-frequency surfaces). If our Baseline (no curriculum) fails, reviewers may argue: *"The baseline was just a bad architecture, your curriculum only compensated for it."*
*   **The Goal:** To prove the meta-learning curriculum works, it must beat an *objectively strong*, peer-reviewed baseline in a domain where progressive learning is already known to be difficult.

## 2. Potential Application Pivots
To inherit a strong baseline, I propose pivoting the application from "arbitrary 3D functions" to a domain where progressively hardening the loss is mathematically natural and extensively studied.

*   **Option A: Physics-Informed Neural Networks (PINNs) / Fluid Dynamics**
    *   *The Task:* Instead of fitting Ackley, the network solves Burgers' Equation.
    *   *The "Smoothing" Equivalent:* The equation's "viscosity" parameter $\nu$. High viscosity creates a smooth, easy curve ($L_1$). Low viscosity creates a violent shockwave ($L_N$).
    *   *The Benefit:* The PINN baseline is standard. The literature explicitly states that standard PINNs fail at low viscosity and *require* a curriculum. My meta-learning algorithm perfectly solves this known gap.
*   **Option B: Standard Image Vision**
    *   *The Task:* CIFAR-10 with ResNet-18 (a highly indisputable baseline).
    *   *The "Smoothing" Equivalent:* Progressively un-blurring the images during training.
*   **Option C: Knowledge Distillation**
    *   *The Task:* Student / Teacher knowledge transfer.
    *   *The "Smoothing" Equivalent:* The Softmax "Temperature". High temperature is an easy target; low temperature is an exact target.

## 3. Decisions & Questions for the Supervisor

1.  **Do you agree with the concern regarding the "Weak Baseline Trap"?** 
2.  **Which Pivot Option do you prefer?** (PINNs, Vision, or KD?)
3.  **Baseline Comparisons:** Are there any specific alternate pacing/curriculum methods you want included in the final experiments?
