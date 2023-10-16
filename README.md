
# This is an Implementation for [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502) 
![](https://github.com/mohame54/DDIM-/blob/main/gifs/test.gif)


DDIM is one of the denoising diffusion probabilistic models family but the key difference here it doesn't require **a large reverse diffusion  time steps** to produce the samples or images as you can see from above this gif was created with 25 reverse diffusion time steps.

------------------------------------------------------
**if you want to train your own model for a specific dataset this colab is for you.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cKkB6RXqs67SmAlRduYuST02hrcCRzG5?usp=sharing)

**if you want to try the pretrained model this colab is for you**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bP1K73ep6MfF-fZGJA26rSqOlY2YO85F?usp=sharing)

## Content:
* An implementation of [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502) with continuous time. All variables are properly named and the code is densely commented.
* A pretrained weights for a model trained on [Flowers Dataset from the  university of oxford](https://www.robots.ox.ac.uk/~vgg/data/flowers/) . 
* A Diffusion model combining all the important parts to generate samples
* Some helper methods for visualizing the generated samples 
--------------------------------------------------------
## Code to try the pretrained model:
```python
from Diffusion import DiffusionModel, DiffUnet 
import matplotlib.pyplot as plt

net = DiffUnet(block_depth=2) # the Unet model for the diffusion module
model = DiffusionModel(net, num_steps=1000, input_res=[64,64])  # initialize the module with 1000 training steps and img size (64, 64)

model.load("Pretrained") # load the pretrained weights from the weights directory
sample = model.generate(num_samples=1, num_infer_steps=25).cpu().numpy().squeeze() # generate one sample with 25 reverse diffusion steps

sample = model.inverse_transform(sample) # reverse the transformation to better visualize the image

# visualize the image
plt.imshow(sample)
plt.axis("off");
```
