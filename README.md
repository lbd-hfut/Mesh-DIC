# Mesh-DIC

## Introduction

Mesh-DIC is a **hybrid Digital Image Correlation (DIC) framework** that combines **traditional finite-element-based DIC algorithms** with **neural network–assisted optimization**. By integrating the physical interpretability and numerical stability of classical Mesh-DIC with the strong nonlinear mapping capability of neural networks, this approach aims to improve robustness, efficiency, and convergence behavior in complex deformation scenarios.

Unlike Subset-DIC, Mesh-DIC represents the displacement field in a **global, continuous manner** using finite element shape functions defined on a mesh. The proposed framework further introduces neural networks to accelerate and enhance the solution process, especially in handling the mapping between global and local coordinates and optimizing nodal degrees of freedom.

---

## Method Overview

The Mesh-DIC framework consists of two complementary components:

### 1. Traditional Mesh-DIC Algorithm

In the classical formulation, Mesh-DIC is solved through an iterative, physics-consistent procedure:

- For each integration point (or pixel) in the global coordinate system, the corresponding **local (isoparametric) coordinates** are determined by solving a nonlinear mapping using **Jacobian-based iterative schemes**.
- Based on the obtained local coordinates, **finite element shape functions** and their gradients are evaluated.
- The **global stiffness matrix** \( \mathbf{H} \) and the **residual vector** \( \mathbf{b} \) are assembled from all elements by enforcing the grayscale consistency assumption.
- The nodal displacement increment is obtained by solving the linearized system:
  \[
  \mathbf{H} \Delta \mathbf{u} = \mathbf{b}
  \]
- The nodal displacements are updated iteratively until convergence.

This traditional approach ensures strong physical interpretability and accuracy but may suffer from high computational cost and sensitivity to initial guesses, especially for distorted meshes or complex boundaries.

---

### 2. Neural Network–Assisted Mesh-DIC Algorithm

To overcome the limitations of purely traditional solvers, a neural network–based strategy is introduced:

- A **neural network is employed to learn the mapping relationship between global coordinates and local element coordinates** within each finite element.
- The **nodal displacements of the mesh are treated as trainable parameters**, analogous to learnable weights in a neural network.
- The grayscale residual defined by the DIC formulation is used as the **loss function**.
- Using **backpropagation**, the neural network propagates gradients through the coordinate-mapping network to the nodal displacement parameters.
- The nodal displacements are optimized directly via gradient-based methods, leading to a solution of the DIC problem.

This neural formulation avoids explicit Jacobian iterations for coordinate inversion and provides improved flexibility and robustness, particularly in cases involving mesh distortion, irregular boundaries, or large deformation gradients.

---

## Key Features

- Hybrid Mesh-DIC framework combining **classical numerical methods** and **neural network optimization**
- Physics-consistent formulation based on grayscale conservation
- Neural network–based global-to-local coordinate mapping
- Nodal displacements treated as trainable parameters
- End-to-end optimization via backpropagation
- Suitable for complex geometries and irregular boundaries

---

## Typical Applications

- Full-field deformation measurement with complex boundaries
- Large-deformation or highly nonuniform strain fields
- Mesh-based DIC acceleration and robustness enhancement
- Comparative studies between classical FEM-DIC and learning-based DIC methods

---

## Notes

- The traditional solver provides a strong baseline and physical reference.
- The neural network component can be used as an accelerator or as an alternative solver.
- Mesh quality and element type (e.g., linear or quadratic elements) influence accuracy and convergence.
- The framework is designed for research and educational purposes.

---

## License

This project is intended for **research and educational use only**.
