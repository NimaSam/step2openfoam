# stl2foam
This `README.md` file provides comprehensive documentation for `stl2foam` repository,
including installation instructions, usage examples, options, features. 


This repository contains a Python script for analyzing STL geometry files using VTK. The script computes various geometric properties such as:

 - volume
 - surface area
 - curvature, and more. 
 
 It also identifies points inside and outside the geometry and generates *blockMeshDict* and *snappyHexMeshDict* for OpenFOAM. The initial repo is taken from:

[STLtoOpenFOAM](https://github.com/Amr-Emad994/STLtoOpenFOAM)

## Author

- Amr Emad
- Nima Samkhaniani

## Dependencies

- VTK (Visualization Toolkit)
- NumPy
- math

## Installation

You can install the required dependencies using pip:

`sh
    pip install vtk numpy
`
## Usage

Run the script from the command line as follows:

`
python stl2foam.py tut/examples/cad.stl --volume --surface_area --center_of_mass --bounding_box --curvature --surface_normals --facet_areas --edge_lengths --aspect_ratios --inside_outside_points --generate_blockMeshDict --cellRefinement 1 --buffer 10 --max_buffer 20
Options
--volume: Compute volume
--surface_area: Compute surface area
--center_of_mass: Compute center of mass
--bounding_box: Compute bounding box
--curvature: Compute curvature
--surface_normals: Compute surface normals
--facet_areas: Compute facet areas
--edge_lengths: Compute edge lengths
--aspect_ratios: Compute aspect ratios
--inside_outside_points: Compute inside and outside points
--generate_blockMeshDict: Generate blockMeshDict for OpenFOAM
--buffer <value>: Buffer size to add around the bounding box (default: 10)
--max_buffer <value>: Maximum buffer size allowed for the bounding box (default: 20)
--cellRefinement <value>: basic mesh refinement in blockMeshDict (value>1 finer mesh and value<1 coarser mesh)
`

Populare usage:

`
stl2foam tut/examples/cad.stl --bounding_box --inside_outside_points --generate_blockMeshDict --cellRefinement 0.5 --generate_snappyHexMeshDict
`




## Features

- #### Volume Calculation 
Calculate the volume of the STL geometry.
- #### Surface Area Calculation
Calculate the surface area of the STL geometry.
- #### Center of Mass Calculation
Compute the center of mass of the STL geometry.
- #### Bounding Box Calculation 
Determine the minimum and maximum bounds of the STL geometry.
- #### Curvature Calculation 
Calculate the curvature of the STL geometry. Supports mean, Gaussian, maximum, and minimum curvatures.
- #### Surface Normals Calculation 
Compute the surface normals and count the number of outward-facing and inward-facing normals.
- #### Facet Areas Calculation 
Determine the minimum and maximum facet areas in the STL geometry.
- #### Edge Lengths Calculation 
Calculate the minimum and maximum edge lengths in the STL geometry.
- #### Aspect Ratios Calculation
Compute the minimum and maximum aspect ratios of the facets in the STL geometry.
- #### Inside and Outside Points Calculation 
Identify points inside and outside the STL geometry.
- #### Generate blockMeshDict 
Generate a blockMeshDict file for OpenFOAM based on the STL geometry.
- #### Generate snappyHexMeshDict
Generate a snappyHexMeshDict file for OpenFOAM based on the STL geometry.

## License
This project is licensed under the terms of the GNU Lesser General Public License (LGPL).
