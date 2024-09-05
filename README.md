# Step2OpenFOAM
A python script collection to automate the import and processing of STEP files for the use in OpenFOAM simulations.

`Step2OpenFOAM.py` creates a pipeline to automatically generate a clean STL file as well as a snappyHexMeshDict and a blockMeshDict for openFOAM simulations based on a `.step` input file. The script will also extract feature edges and supply them as a VTK file.

The determination of locationInMesh specified in snappyHexMeshDict is done using Blender's raycasting engine. The principle to locate this point is described in more detail [below](#find-point-inside-mesh) . 

## Dependencies

- python 3.10
- bpy module for python
- FreeCAD libraries

## Setup:

1. Setup python 3.10
2. Install FreeCAD
   2.1 LINUX: Install it via `apt install freecad`, set `FREECADPATH = '/usr/lib/freecad-python3/lib/'`
   2.2 WINDOWS: Install the .exe via the official website, set `FREECADPATH` to where `FreeCAD.pyd` is located, typically located in `FREECADPATH = 'C:/Program Files/FreeCAD 0.XX/bin/'`
3. Make sure `bpy` is installed for python, `pip install bpy`
4. Set up config.json (see explanation of values below)
5. Run the script from the console while supplying a valid config file
``` 
python3.10 STEP2OpenFOAM.py /path/to/config.json
```


## Config

#### General Settings

- "**project_directory**": *"./path/to/project_basedir/"*,
  - Path to the base directory of the project

#### Settings for blockMeshDict generation

- "**blockmesh_ndim**": *100*,
  - This is the highest resolution of the backgroundMesh in the direction of greatest extent of the input geometry.
  - Remaining resolutions will be calculated based on this parameter such that blocks are perfect cubes. 
- "**blockmesh_buffer**": *0.1*,
  - Additional 'flesh' around the object for blockMesh generation. E.g. 0.1 = 10% extra space around the object, while 0 = 0% means that the blockMesh is snug against the geometry in the direction of greatest extent.


#### Settings for mesh Cleaning

- "**cleanmesh_merge_threshold**": *1e-7*,
  - Tolerance for merging closeby and duplicate vertices.
  - A higher number will fix larger holes, but it may also delete (smoothe) high detail areas.
  - Adapt this number to the size of your geometry.


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



# Notes

- The STEP sample file has been downloaded from https://grabcad.com/library/3d-printable-phone-case-1, original author: Ines Slimani
