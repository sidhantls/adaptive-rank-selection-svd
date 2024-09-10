# Adaptive Rank Selection for Low-Rank Approximation of Language Models
This repository contains the implementation (unofficial) of the paper "Adaptive Rank Selections for Low-Rank Approximation of Language Models". The method introduces learnable neural networks to predict optimal decomposition ranks. This repo only implements the rank selection step - i.e it only implements [Algorithm 1: Adaptive Rank Selection](https://aclanthology.org/2024.naacl-long.13.pdf). Fine-tuning after rank selection is not implemented

<p align="center">
<img src="outline.png" alt="Outline Image" width="50%" />
  <p style="font-size: 14px; color: gray;">
    <a href="https://aclanthology.org/2024.naacl-long.13.pdf">Figure 1</a>
  </p>
</p>

## Implementation Caveats:
* Implementation of SVD: ASVD and Fisher SVD is implemented, doesn't implement IWSVD. IWSVD is utilized in final paper.
