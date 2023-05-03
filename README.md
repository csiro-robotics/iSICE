# Learning Partial Correlation based Deep Visual Representation for Visual Classification

#### Saimunur Rahman, Piotr Koniusz, Lei Wang, Luping Zhou, Peyman Moghadam, Changming Sun <br/> CSIRO Data61, University of Wollongong, University of Sydney

## Introduction
The official repository for paper "Learning Partial Correlation based Deep Visual Representation for Image Classification" To appear in 2023 The IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR).

![iSICE](isice.png)

Our partial correlation based deep visual representation method iSICE is implemented using PyTorch as a meta-layer. iSICE is fully differentiable and optimisable with torch autograd package. Our codebase designed in a flexible manner so that any change in the pipeline can be done more convaniently. Specifically, we considered separating backone networks, global image representation and classification steps so that each step can be changed without affecting others. We find that such desgin is helpful to conduct hassle-free experiments and reduce issues during future extensions.
