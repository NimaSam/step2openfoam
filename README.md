# step2openfoam
A python script collection to automate the import and processing of STEP files for the use in OpenFOAM simulations.

`Step2OpenFOAM.py` creates a pipeline using the Blender addon STEPper and snappyhexmeshgui to automatically create blockMeshDict and snappyHexMeshDict files, as well as a triangulated .stl geometry. Furthermore, the script automatically determines a locationInMesh specified in snappyHexMeshDict using Blender's raycasting engine. The principle is described in more detail [below](#find-point-inside-mesh) . 

## Prerequisites:

- Windows 10+, 64-bit
- Blender 2.8+ with Python 3.7+ [blender.org ](https://www.blender.org/)
- STEPper addon [ambient.gumroad.com/l/stepper](https://ambient.gumroad.com/l/stepper)
- snappyhexmeshgui addon [github.com/tkeskita/snappyhexmesh_gui](https://github.com/tkeskita/snappyhexmesh_gui)

## Setup:

1. Install Blender
2. Add blender.exe to PATH *OR* copy path to blender.exe
   - Default install location: C:/Program Files/Blender Foundation/Blender X/blender.exe
3. Set up config.json (see explanation of values below)
4. Run the script using blender from the console:
``` 
blender.exe --background --python Step2OpenFOAM.py
```


## Config

#### Import Settings for STEPper

- "**stepper_import_filepath**": *"./step_samples/"*,
  - Path to the base directory where the .step/.stp file is located
- "**stepper_import_file**": *"Oppo_reno_10_pro_5G.step"*,
  - Name of the file
- "**stepper_import_detail_level**": *1000*,
  - How detailed the imported STEP file wil be meshed. This value is an input to the STEPper addon. Higher numbers create more vertices.

#### Export Settings for SnappyHexMeshGUI

- "**snappyhex_export_filepath**": *"./export/"*,
  - Path to the folder where snappyhexmeshgui will export its files.
- "**snappyhex_no_cpus**": *4*,
  - No. CPUs for decomposeParDict
- "**snappyhex_cell_length**": *0.1*,
  - Length of Base Block Mesh Cell Size
- "**snappyhex_surface_refinement_min**": *2*,
  - Minimum cell refinement level for surface
- "**snappyhex_surface_refinement_max**": *2*,
  - Maximum cell refinement level for surface
- "**snappyhex_cleanup_distance**": *1e-5*,
  - Maximum distance for merging closeby vertices
- "**snappyhex_feature_edge_level**": *0*,
  - Feature edge refinemenet level for surface

#### Settings for the Find Point Inside Mesh subfunction

- "**pointInMesh_delta**": *0.001*,
  - Minimum space to surrounding geometry for target point
- "**pointInMesh_deterministic**": *1*,
  - If set to 1 the random functions will be deterministic (only applies to determination of locationInMesh)
- "**pointInMesh_maxiter**": *500*,
  - Maximum iterations the script will run before aborting
- "**pointInMesh_rays_primary**": *50*,
  - No. of rays for first raycast. This should be a lower number (20 - 500)
- "**pointInMesh_rays_secondary**": *5000*,
  - No. of rays for secondary raycast. This value can be higher (500 - 10000). Higher numbers mean a more accurate result, but more computational time.
- "**pointInMesh_seed**": *42*
  - Seed for the random functions if set to deterministic



 
# Find Point Inside Mesh

The goal of this script is to quickly and efficiently find a point inside the mesh with a certain `delta` of space in all directions to neighboring geometry. I.e. a sphere with radius `delta` around the point should not touch or cross the base geometry. 

To find a point inside the mesh, the script uses the following algorithm:

1. Specify search parameters `delta=0.001`, `maxiter=50`,  `no_rays=50` and `no_rays_secondary=500`.
2. Cast a ray from a random face inside the mesh. Get hit location. Determine midway point.
3. From that midway point, fire `no_rays` rays spatially in all directions. Determine smallest ray length.
4. If that length is smaller than `delta`, repeat with another random face. (It means that there already is geometry closer than `delta` to the point.)
5. If not, repeat the raycast with `no_rays_secondary` and get the minimum ray length
6. Repeat step 4.
7. If still no geometry violates the `delta` threshold, return that point.
8. Repeat until a point is found or `maxiter` iterations are surpassed.

If no point can be found, the possible causes are:
- `delta` is too large: The mesh may be too small to find any point that fulfills the `delta` requirement.
- `maxiter` is too low: If no point is found, the iterations can be increased. This can be useful for very fine meshes, but it will icnrease computation time.
- The mesh geometry is non-manifold or has holes. This is problematic for many other reasons, and may be a result from the STEPper import, but it can also cause this script to fail.



