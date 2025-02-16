# Knowledge-Guided-learning

## A brief introduction to Knowledge-Guided-learning: Deep learning methods for singular variational problems with Lavrentiev phenomenon
* Deep learning has made significant advancements in the field of scientific computing, particularly in its application to solving problems involving differential operators using deep neural networks. However, the use of neural networks to address problems with singularities remains challenging. Specifically, we discuss the limitations of deep learning methods for singular variational problems that exhibit the Lavrentiev phenomenon. For such problems, we demonstrate that the standard deep Ritz method and some of its variants fail to capture singular minimizers.

* To address this issue, we introduce a guiding term and a scheduling technique into the training process, which encourages the neural network to explore desired solutions. Numerical experiments demonstrate that this method achieves significantly better approximations compared to conventional approaches. Furthermore, we apply the same algorithm to problems with regular solutions to illustrate the robustness of the proposed method.

## Paper
* This repository contains the code in the paper 'Deep learning methods for singular variational problems with Lavrentiev phenomenon' by Dong Wang, Xianmin Xu, and Boyi Zou.

## About the code
* The code in this repository was written in '[Python 3.11](https://www.python.org/downloads/)' and '[PyTorch 2.5.1](https://pytorch.org/get-started/locally/)'. (or it can be directly run on a [Colab](https://colab.google/) Notebook.)

* Here, we provide the code of this paper and have added annotations to certain sections to enhance understanding and facilitate learning. We use 'KGL.py' in 'Mania-1d' as an example to provide annotations for interested scholars to study, and the remaining examples follow a similar format.

* One should keep in mind that this paper primarily proposes the method and does not focus on fine-tuning parameters. Additionally, we employ a larger depth and width to ensure sufficient approximation capability, highlighting the challenges associated with such problems. Similar results can be achieved with fewer parameters. And one may need to readjust the parameters when using this code to solve different problems.
