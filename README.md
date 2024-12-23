# Separable Gaussian Neural Network (SGNN)

## Paper
For detailed methodology, structure, and analysis, refer to our published paper:  
[S. Xing and J.Q. Sun, 2023, Separable Gaussian Neural Networks: Structure, Analysis, and Function Approximations, *Algorithm*, 16(10)](https://www.mdpi.com/1999-4893/16/10/453)

---

## Features
- **Interpretable Neural Network Design**: Leverages the separable property of Gaussian radial-basis functions (GRBFs) to reduce model complexity and improve interpretability.  
- **Computational Efficiency**: Achieves linear scaling with input dimensions by reducing neurons from \(O(N^d)\) (traditional GRBFNN) to \(O(dN)\), significantly enhancing computational speed.  
- **High Accuracy**: Maintains the dominant subspace of the Hessian matrix during gradient descent, achieving results comparable to GRBFNNs while outperforming ReLU-based DNNs by three orders of magnitude for complex geometries.  
- **Improved Trainability**: Simplified tuning process compared to DNNs using ReLU or Sigmoid activation functions.  
- **Versatile Applications**: Effective in interpolation, classification, and function approximations, especially in high-dimensional spaces or datasets with intricate geometric structures.

---

## Prerequisites
To use SGNN, ensure the following dependencies are installed:
- **Python**: Version 3.08 or higher  
- **TensorFlow**: Version 2.10 or higher  (tested on 2.10 and 2.17)

---

## License
This project is licensed under the **MIT License**, allowing free usage, modification, and distribution under the license terms. Refer to the LICENSE file for more details.

---

## Citation
If you use this work in your research, please cite the following paper:

```bibtex
@Article{XingandSunSGNN2023,
  Title   = {Separable Gaussian Neural Networks: Structure, Analysis, and Function Approximations},
  Author  = {Siyuan Xing and Jianqiao Sun},
  Year    = {2023},
  Issue   = {10},
  Volume  = {16},
  Pages   = {453 (19 pages)},
  Journal = {Algorithm},
  DOI     = {10.3390/a16100453}
}

