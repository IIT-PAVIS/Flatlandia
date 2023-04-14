# Flatlandia Examples

[video3.webm](https://user-images.githubusercontent.com/32576285/229822233-c51203e1-a5af-4a2c-bae9-dae42d5b0834.webm)

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

[video4.webm](https://user-images.githubusercontent.com/32576285/229822420-b849c3b8-f4a9-499f-ac65-3b9966955c28.webm)


### Fine-grained localization

[video5.webm](https://user-images.githubusercontent.com/32576285/229822512-e6e6bf8f-2fe6-4852-bef1-26d41f87493f.webm)

We provide baselines for the fine-grained localization tasks, exemplified in `demp_fine_localization.py`.
The code allows to train the different models described in the accompanying paper, and defined in `utils/models.py`. 
We refer to the paper and to the configuration section of `demp_fine_localization.py` for additional details.
