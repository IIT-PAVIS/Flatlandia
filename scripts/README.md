# Flatlandia Examples


[video3.mp4](https://user-images.githubusercontent.com/32576285/232072824-d7c11658-e005-4d13-84bb-9940716e7769.mp4)


## Environment setup
To log and visualize training information, the code we provide relies on the [Weights&Biases](https://wandb.ai/site).
To use the code,
- create an account on [Weights&Biases](https://wandb.ai/site)
- set it up on the computer (https://docs.wandb.ai/quickstart)
- add the W&B username to `scripts/utils/common.py` under `wandb_user`.

## The Flatlandia tasks

We propose to approach the visual localization problem in Flatlandia as two separate tasks:

- Coarse Map Localization: localizing a single image observing a set of objects in respect to a 2D map of object landmarks
- Fine-grained 3DoF Localization: estimating latitude, longitude, and orientation of the image within a 2D map


### Coarse localization

[video4.mp4](https://user-images.githubusercontent.com/32576285/232072884-ea2e2cf3-c8cc-475c-b953-ecbee195a9c6.mp4)


### Fine-grained localization

[video5.webm](https://user-images.githubusercontent.com/32576285/229822512-e6e6bf8f-2fe6-4852-bef1-26d41f87493f.webm)

We provide baselines for the fine-grained localization tasks, exemplified in `demp_fine_localization.py`.
The code allows to train the different models described in the accompanying paper, and defined in `utils/models.py`. 
We refer to the paper and to the configuration section of `demp_fine_localization.py` for additional details.
