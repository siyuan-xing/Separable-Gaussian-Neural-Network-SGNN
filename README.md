# Separable Gaussian Neural Network (SGNN)
 
# Paper
- [Separable Gaussian Neural Networks: Structure, Analysis, and Function Approximations](https://www.mdpi.com/1999-4893/16/10/453)  

## Features
- **Interpretable Neural Network Design**: Utilizes the separable property of Gaussian radial-basis functions to reduce complexity and enhance interpretability.
- **Efficiency**: Reduces the number of neurons from \( O(N^d) \) (GRBFNN) to \( O(dN) \), exponentially improving computational speed and scaling linearly with input dimensions.
- **High Accuracy**: Preserves the dominant subspace of the Hessian matrix during gradient descent, achieving comparable accuracy to GRBFNN and delivering three orders of magnitude greater accuracy than ReLU-based DNNs for complex geometries.
- **Trainability**: Easier to tune compared to DNNs with ReLU or Sigmoid activation functions.
- **Applications**: Effective for interpolation, classification, and function approximations, particularly in high-dimensional and complex geometric data.

---

## Prerequisites
Ensure you have the following installed:
- Python: 3.12
- Tensorflow: >2.10 


## License
This project is licensed under the MIT License. You are free to use, modify, and distribute this software in accordance with the terms of the license.

## Citation
  @Article{XingandSunSGNN2023,
   Title                    = {Separable Gaussian Neural Networks: Structure, Analysis, and Function Approximations},
    Author                   = {Siyuan Xing and Jianqiao Sun},
    Year                     = {2023},
    Issue						={10},
    Volume 					=  {16},
    Pages						={453 (19 pages)},
    Journal                  = {Algorithm},
    doi 					={10.3390/a16100453},
  }


