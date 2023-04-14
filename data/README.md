# Additional Flatlandia dataset information

## Data provided

In addition to the core Flatlandia dataset, we provide additional data that might be helpful to the community:

- `data/maps` :: for each *reference_map* we include a directory with:
    - *SfM* the Colmap reconstruction of the scene; for efficiency, the image binaries images.bin have been compressed.  
    - *tokens.txt* a file with all the Mapillary tokens of the images used for the scene reconstruction
    - *transformations.json* contains the transformation matrix to align the Colmap point cloud so that the principal plane 
      is horizontal ('to_principal_plane'), the rotation, translation and scaling to align the point cloud to GPS coordinates
      ('to_mapillary') and the scaling factor to go from GPS coordinates to the Flatlandia maps ('to_flatlandia'). 
        The latter is needed, as for increased numerical stability we normalize the Flatlandia maps so that the largest 
        one fits in the range (-1,1)
    - *class_segmented_point_cloud.ply* contains the SfM point cloud, with each point assigned to a class or the background
    - *bbs_for_segmented_objects.ply* contains a 3D bounding box in the SfM point cloud space for each map object

- `data/local_maps.json` :: a dictionary of pre-computed local maps, indexed by 'reference_map' and 'query_token'. For each query we provide two maps:
  - *GT* is obtained transforming the object coordinates in the reference map to the camera reference system
  - *depth* is obtained using camera intrinsics and monocular depth estimation to project the detections

- `data/region_proposal.json` :: pre-computed data needed to replicate the fine-grained localization results (see `scripts/README`)

- `data/coarseloc_similarity` :: pre-computed data needed to replicate the coarse localization results (see `scripts/README`)

## Accessing the data
The SfM scenes contained in `data/maps` can be visualized using [Colmap](https://colmap.github.io/), using the `import model`
option to load the `SfM` directory, after decompressing `images.zip`.  