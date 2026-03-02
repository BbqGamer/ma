# Implementation and Experimentation Plan (Draft)

*This document outlines the proposed meta-learning curriculum and the required scientific baselines. Specific technical details and domains are pending supervisor approval.*

---

## 1. Experimental Design: Minimizing Variables

To isolate the proposed meta-optimizer (with the Softmax Bridge and Monotonic Regularizer) as the true source of improvement, all competing methods must be trained under identical conditions.

### The "Weak Baseline" Protocol
To ensure scientific rigor, we must first establish and tune a standard baseline model. This guarantees that our method improves upon an optimized architecture, rather than merely compensating for poorly chosen hyperparameters.

If we remain in the current math-function regression domain, this requires a "Phase 0" grid search on model capacity and learning rate. *Alternatively, we may pivot to a different domain (e.g., PINNs, Vision) with universally accepted hyperparameters to bypass this risk (pending supervisor discussion).*

### The 5 Proposed Experimental Arms

Whether we use math functions, PINNs, or Vision, we will test these 5 conditions:

1.  **Baseline 1: No Curriculum** (Standard training on the hardest task/target)
2.  **Baseline 2: Static Multi-Loss** (Equal weighting of all smoothed tasks)
3.  **Baseline 3: hard-coded Sequential Curriculum** 
4.  **Ablation:** Meta-Curriculum WITHOUT the Monotonic Gradient Penalty ($\lambda = 0$)
5.  **Proposed:** Full Dynamic Multi-Loss Curriculum ($\lambda > 0$)

---

## 2. Core Implementation Strategy: The Meta-Optimization Loop

The goal is to implement the bi-level optimization loop proposed in `2026-01-15-proposal.md`.

*   **Simultaneous Loaders:** Datasets must yield the full stack of $N$ smoothed targets per batch, rather than reading them one level at a time.
*   **Unconstrained Parameters:** Introduce a parameter vector $u$ inside the training loop to derive the loss weights via Softmax.
*   **The Meta-Update:** We must differentiate the validation loss with respect to $u$. Because the inner model parameters $\theta$ have already been updated using $u$, this requires unrolling the inner-loop optimizer. We propose using the `higher` library for this differentiable capability.

---

## 3. High-Level Execution Workflow

1.  **Phase A (Alignment):** Agree with the supervisor on the final application domain (Math Functions vs. PINNs vs. Deep Vision) to ensure strong baselines.
2.  **Phase B (Infrastructure):** Refactor datasets to allow simultaneous target loading. Fix random seeds to ensure perfectly identical $80/20$ splits across all baseline runs.
3.  **Phase C (The Engine):** Implement the unrolling and the monotonic regularization penalty.
4.  **Phase D (Execution & Reporting):** Evaluate the 5 experimental arms, ensuring MLflow automatically logs the learned weights $w_i$ over time to visually verify the curriculum schedule.
