"""
-------------------------------------------------------------------------------
    VTK is free software: you can redistribute it and/or modify it
    under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 2.1 of the License, or
    (at your option) any later version.

    VTK is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
    for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with VTK.  If not, see <http://www.gnu.org/licenses/>.

Application
    stlAnalysis

Group
    grpGeometryAnalysis

Description
    Script to analyze STL geometry files using VTK. This script computes
    various geometric properties such as volume, surface area, curvature,
    and more. It also identifies points inside and outside the geometry.
    
Author    
    Amr Emad
    Nima Samkhaniani
Version
    v1.1

Dependencies
    - VTK (pip install vtk)
    - NumPy (pip install numpy)
    - JSON (Standard library)

Usage
    Run the script from the command line as follows:

    python stl2foam.py <path_to_stl_file> [options]

    Example:
    python stl2foam.py examples/CRM-HL_Coarse.stl --volume --surface_area --center_of_mass --bounding_box --curvature --surface_normals --facet_areas --edge_lengths --aspect_ratios --inside_outside_points --generate_blockMeshDict

    or add it to bashscript e.g:
    alias stl2foam="python /home/nsamkhaniani/foam/ianus/stltoopenfoam/stl2foam.py"
    
    then run it in any directory.
    to generate blockMeshDict run it as:
    stl2foam constant/triSurface/CAD.stl  --bounding_box --generate_blockMeshDict -minCellNumber 5


    Options:
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

-------------------------------------------------------------------------------
"""

from stl2foamlib import *

def main():
    """
    Main function to read the STL file, compute metrics, and write a JSON report.
    """
    args = parse_arguments()

    # Read the STL file
    mesh = read_stl_file(args.filename)
    
    # Compute and export data based on flags
    volume, surface_area, center_of_mass = None, None, None
    if args.volume or args.surface_area:
        mass_properties = vtk.vtkMassProperties()
        mass_properties.SetInputData(mesh)
        if args.volume:
            volume = mass_properties.GetVolume()
        if args.surface_area:
            surface_area = mass_properties.GetSurfaceArea()
    
    if args.center_of_mass:
        center_of_mass_filter = vtk.vtkCenterOfMass()
        center_of_mass_filter.SetInputData(mesh)
        center_of_mass_filter.Update()
        center_of_mass = center_of_mass_filter.GetCenter()
    
    min_bounds, max_bounds = None, None
    if args.bounding_box:
        min_bounds, max_bounds = compute_bounding_box(mesh)
    
    curvature_values = None
    if args.curvature:
        curved_mesh = compute_curvature(mesh, curvature_type='mean')
        curvature_values = extract_curvature_data(curved_mesh)
    
    surface_normals = None
    if args.surface_normals:
        surface_normals = compute_surface_normals(mesh)
    
    min_area, max_area = None, None
    if args.facet_areas:
        min_area, max_area = compute_facet_areas(mesh)
    
    min_edge_length, max_edge_length = None, None
    if args.edge_lengths:
        min_edge_length, max_edge_length = compute_edge_lengths(mesh)
    
    min_aspect_ratio, max_aspect_ratio = None, None
    if args.aspect_ratios:
        min_aspect_ratio, max_aspect_ratio = compute_aspect_ratios(mesh)
    
    inside_point, outside_point = None, None
    if args.inside_outside_points and center_of_mass is not None and min_bounds is not None and max_bounds is not None:
        outside_point = find_outside_point(mesh, center_of_mass, min_bounds, max_bounds, buffer=args.buffer, max_buffer=args.max_buffer)  # Ensure within buffer range
    
    if args.inside_outside_points:
        inside_point = find_inside_point(mesh)
        
    # Write JSON report
    write_json_report(args.filename, volume, surface_area, center_of_mass, min_bounds, max_bounds, curvature_values, surface_normals, min_area, max_area, min_edge_length, max_edge_length, min_aspect_ratio, max_aspect_ratio, inside_point, outside_point)
    print(f"JSON report written based on {args.filename}")

    # Generate blockMeshDict if requested
    if args.generate_blockMeshDict and min_bounds and max_bounds:
        compute_edges(mesh, feature_angle=args.feature_angle)
        generate_blockMeshDict(min_bounds, max_bounds, minCellNumber=args.minCellNumber,buffer=args.buffer)
        print("blockMeshDict generated successfully.")

    # Generate snappyHexMeshDict if requested
    if args.generate_snappyHexMeshDict and inside_point:
        if args.feature_angle:
            compute_edges(mesh, feature_angle=args.feature_angle)
        else:
            compute_edges(mesh)
        generate_snappyHexMeshDict(inside_point)
        print("snappyHexMeshDict generated successfully.")
    
    # Generate snappyHexMeshDict if requested
    if args.generate_snappyHexMeshDict and inside_point:
        if args.feature_angle:
            compute_edges(mesh, feature_angle=args.feature_angle)
        else:
            compute_edges(mesh)
        generate_snappyHexMeshDict(inside_point)
        print("snappyHexMeshDict generated successfully.")

    # Generate only feature edges    
    if args.feature_edge:
        if args.feature_angle:
            compute_edges(mesh, feature_angle=args.feature_angle)
        else:
            compute_edges(mesh)


       
if __name__ == "__main__":
    main()
