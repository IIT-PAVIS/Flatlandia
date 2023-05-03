# The Flatlandia Dataset

## Introduction


[intro.webm](https://user-images.githubusercontent.com/32576285/231718784-c65424f5-91fe-4659-b1df-6e50b40f3ef8.webm)



We introduce the Flatlandia dataset, a novel problem for visual localization from object detections and annotated object 
maps: given a visual in which common urban objects (e.g., benches, streetlights, signs) are detected, and given a 2D map
of the area, annotated with the location of similar urban objects, we want to recover the location of the visual query on
the map, expressed as a 2D location (latitude/longitude) and an angle (orientation). 

Solving these problems would allow to better exploit the wide availability of open urban maps annotated with GPS 
locations of common objects (e.g., via surveying or crowd-sourced). Such maps are also more storage-friendly than 
standard large-scale 3D models often used in visual localization while additionally being privacy-preserving.
As existing datasets are unsuited for the proposed problem, we designed a novel dataset for 3DoF 
visual localization, based on the crowd-sourced data available in [Mapillary](https://www.mapillary.com/app/) for five 
European cities. 

The code in this repository is part of the paper:
<br>
**["You are here! Finding position and orientation on a 2D map from a single image: The Flatlandia localization problem and dataset."](https://arxiv.org/abs/2304.06373)**
<br>
 <a href='https://pavis.iit.it/web/pattern-analysis-and-computer-vision/people-details/-/people/matteo-toso'>Matteo Toso</a>,
 <a href='https://pavis.iit.it/web/pattern-analysis-and-computer-vision/people-details/-/people/matteo-taiana'>Matteo Taiana</a>,
 <a href='https://pavis.iit.it/web/pattern-analysis-and-computer-vision/people-details/-/people/stuart-james'>Stuart James</a> and
 <a href='https://pavis.iit.it/web/pattern-analysis-and-computer-vision/people-details/-/people/alessio-delbue'>Alessio Del Bue</a>.
<br>
arXiv preprint [arXiv:2304.06373 (2023)](https://arxiv.org/abs/2304.06373).

[video1bis.webm](https://user-images.githubusercontent.com/32576285/229825564-a3061b61-9b86-44c6-8be9-bf1ca655fdbd.webm)



Further details about the dataset and the proposed tasks can be found on [Arxive](https://arxiv.org/abs/2304.06373)

The Flatlandia data set is published under MIT license.

If you use this code in your research, please acknowledge it as:

@article{toso2023you,
  title={You are here! Finding position and orientation on a 2D map from a single image: The Flatlandia localization problem and dataset},
  author={Toso, Matteo and Taiana, Matteo and James, Stuart and Del Bue, Alessio},
  journal={arXiv preprint arXiv:2304.06373},
  year={2023}
}

## Project set up
We developed Flatlandia using the Ubuntu operative system, but we expect that it can be run on other operating systems.
### Set up the Conda environment
```
git clone git@github.com:IIT-PAVIS/Flatlandia
cd Flatlandia
conda create -n flatlandia python=3.7
conda activate flatlandia
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
pip install matplotlib mapillary 
conda install -c dglteam dgl-cuda11.6 
conda install wandb --channel conda-forge
```

### Set up Mapillary access
The visual queries used in the Flatlandia dataset were obtained from, and belong to, [Mapillary](https://www.mapillary.com/). 
To visualize them, we rely on the official Mapillary API; follow the instruction provided at [Mapillary API](https://blog.mapillary.com/update/2021/06/23/getting-started-with-the-new-mapillary-api-v4.html)
to obtain an access token. This token then has to be added to `scripts/utils/common.py` under `mapillary_access_token`.


## The Flatlandia dataset


[video2.webm](https://user-images.githubusercontent.com/32576285/229826247-e8cceb5f-5775-46d9-90f9-c73c062f298c.webm)



The Flatlandia dataset provides a series of visual queries sampled from crowd-sourced street-level Mapillary images, 
each annotated with a set of object detections (2D bounding boxes and class labels). These queries are sampled from 
20 areas across Europe, and for each area we provide a reference map: a 2D map with the location (latitude and longitude)
and class of the objects present in the scene. We here provide an example of a reference map in Vienna (Left), a visual 
query present in it (Top Right), and a zoomed-in, camera-centric map with only the objects observed in the query.
![example_plot](https://user-images.githubusercontent.com/32576285/230586376-d7be61a2-ceaf-42de-9b78-980f31d5ac86.png)


The core Flatlandia dataset is stored in json format under `data/flatlandia.json`, and can easily be accessed with a torch dataloader:

```
from scripts.utils.dataloader import FlatlandiaLoader 
dataset = FlatlandiaLoader()
for problem in dataset:
    ...
```

Each dataset entry is a json file containing:

- *reference_map*: the id of the Flatlandia scene (integer in the range 0-19),
- *reference_xy*: the latitude and longitude of each object in the reference map
- *reference_class*: the class label of each object, encoded as an integer (see `scripts/dataloader.id_to_scene` for conversion)
- *query_token*: the unique Mapillary token associated with the visual query,
- *query_xy*: the location of the camera in the reference map
- *query_theta*: the orientation of the camera
- *query_matches*: the index of the detected objects in the list of reference map objects
- *query_detections*: location of the detections on the image, as the top left and bottom right corners of a bounding box
- *intrinsics*: intrinsic parameters of the camera that acquired the visual query

Each dataset entry can be visualized, as in the above image, using the function `visualize_problem(x)` defined 
in `scripts.utils.dataloader`.

## Additional content
In addition to the Flatlandia dataset, we provide:
- SfM reconstructions of the Flatlandia scenes (`data/README.MD`)
- Code exemplifying the use of the dataset (`scripts/README.MD`)

## Acknowledgement
This code was developed as part of the MEMEX project, and has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 870743.
