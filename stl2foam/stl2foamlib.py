import vtk
import numpy as np
import json
import os
import argparse
import math
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster

def read_stl_file(filename):
    """
    Reads an STL file and returns the mesh data.

    @param filename: The path to the STL file.
    @return: The mesh data as a vtkPolyData object.
    """
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def compute_curvature(mesh, curvature_type='mean'):
    """
    Computes the curvature of the mesh.

    @param mesh: The mesh data as a vtkPolyData object.
    @param curvature_type: The type of curvature to compute ('mean', 'gaussian', 'maximum', 'minimum').
    @return: The mesh data with curvature values as a vtkPolyData object.
    """
    curvature_filter = vtk.vtkCurvatures()
    curvature_filter.SetInputData(mesh)
    
    if curvature_type == 'mean':
        curvature_filter.SetCurvatureTypeToMean()
    elif curvature_type == 'gaussian':
        curvature_filter.SetCurvatureTypeToGaussian()
    elif curvature_type == 'maximum':
        curvature_filter.SetCurvatureTypeToMaximum()
    elif curvature_type == 'minimum':
        curvature_filter.SetCurvatureTypeToMinimum()
    
    curvature_filter.Update()
    return curvature_filter.GetOutput()


def compute_edges(mesh, feature_angle=30.0, tri_dir="constant/triSurface"):
    """
    Extracts feature edges from the mesh and saves them to a VTK file.
    
    @param mesh: The mesh data as a vtkPolyData object.
    @param feature_angle: The angle threshold for detecting sharp edges (in degrees).
    @return: The extracted edges as a vtkPolyData object.
    """
    # Ensure the output directory exists
    if not os.path.exists(tri_dir):
        os.makedirs(tri_dir)

    # Path to save the VTK file
    filename = os.path.join(tri_dir, "featureEdges.vtk")
    

    # Extract feature edges
    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(mesh)  # Use SetInputData for vtkPolyData input

    # Set the criteria for edge extraction
    feature_edges.BoundaryEdgesOn()    # Extract boundary edges (edges at the boundary of the mesh)
    feature_edges.FeatureEdgesOn()     # Extract feature edges (sharp edges)
    feature_edges.NonManifoldEdgesOn() # Extract non-manifold edges
    feature_edges.ManifoldEdgesOn()    # Extract manifold edges (optional, usually turned off)
    
    # Set the angle threshold for detecting sharp edges
    feature_edges.SetFeatureAngle(feature_angle)
    feature_edges.Update()

    # Write the extracted edges to a VTK file
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(filename)  # Ensure the filename is in quotes
    writer.SetInputData(feature_edges.GetOutput())
    writer.Write()

    # Return the extracted edges as vtkPolyData
    return feature_edges.GetOutput()


# def compute_edges(mesh, feature_angle=30.0, tri_dir="constant/triSurface", num_clusters=3):
#     """
#     Extracts feature edges from the mesh, classifies them based on their distances, and saves them to VTK files.
    
#     @param mesh: The mesh data as a vtkPolyData object.
#     @param feature_angle: The angle threshold for detecting sharp edges (in degrees).
#     @param tri_dir: Directory to save the VTK files.
#     @param num_clusters: Number of clusters for edge classification.
#     @return: The extracted and classified edges as a list of vtkPolyData objects.
#     """
#     # Ensure the output directory exists
#     if not os.path.exists(tri_dir):
#         os.makedirs(tri_dir)

#     # Path to save the main VTK file
#     filename = os.path.join(tri_dir, "featureEdges.vtk")
    
#     # Extract feature edges
#     feature_edges = vtk.vtkFeatureEdges()
#     feature_edges.SetInputData(mesh)  # Use SetInputData for vtkPolyData input

#     # Set the criteria for edge extraction
#     feature_edges.BoundaryEdgesOn()    # Extract boundary edges (edges at the boundary of the mesh)
#     feature_edges.FeatureEdgesOn()     # Extract feature edges (sharp edges)
#     feature_edges.NonManifoldEdgesOn() # Extract non-manifold edges
#     feature_edges.ManifoldEdgesOn()    # Extract manifold edges (optional, usually turned off)
    
#     # Set the angle threshold for detecting sharp edges
#     feature_edges.SetFeatureAngle(feature_angle)
#     feature_edges.Update()

#     # Get the points and lines from the feature edges
#     points = feature_edges.GetOutput().GetPoints()
#     lines = feature_edges.GetOutput().GetLines()
    
#     # Store the points representing each edge
#     edge_points = []
#     line_data = vtk.vtkIdList()
#     lines.InitTraversal()
    
#     while lines.GetNextCell(line_data):
#         point_id1 = line_data.GetId(0)
#         point_id2 = line_data.GetId(1)
        
#         p1 = np.array(points.GetPoint(point_id1))
#         p2 = np.array(points.GetPoint(point_id2))
        
#         edge_points.append(p1)
#         edge_points.append(p2)
    
#     edge_points = np.array(edge_points)
    
#     # Compute pairwise distances
#     pairwise_distances = pdist(edge_points, metric='euclidean')
    
#     # Perform hierarchical clustering
#     Z = linkage(pairwise_distances, method='ward')
#     labels = fcluster(Z, num_clusters, criterion='maxclust')
    
#     # Create VTK objects for each cluster
#     classified_edges = []
#     for cluster_index in range(1, num_clusters + 1):
#         edge_points_vtk = vtk.vtkPoints()
#         edge_lines = vtk.vtkCellArray()
        
#         lines.InitTraversal()
#         point_counter = 0
#         while lines.GetNextCell(line_data):
#             point_id1 = line_data.GetId(0)
#             point_id2 = line_data.GetId(1)
            
#             if labels[point_counter] == cluster_index:
#                 p1 = points.GetPoint(point_id1)
#                 p2 = points.GetPoint(point_id2)
                
#                 edge_id1 = edge_points_vtk.InsertNextPoint(p1)
#                 edge_id2 = edge_points_vtk.InsertNextPoint(p2)
                
#                 edge_line = vtk.vtkLine()
#                 edge_line.GetPointIds().SetId(0, edge_id1)
#                 edge_line.GetPointIds().SetId(1, edge_id2)
#                 edge_lines.InsertNextCell(edge_line)
            
#             point_counter += 1
        
#         polydata = vtk.vtkPolyData()
#         polydata.SetPoints(edge_points_vtk)
#         polydata.SetLines(edge_lines)
        
#         classified_edges.append(polydata)
        
#         # Save each cluster to a separate VTK file
#         cluster_filename = os.path.join(tri_dir, f"featureEdges_cluster_{cluster_index}.vtk")
#         writer = vtk.vtkPolyDataWriter()
#         writer.SetFileName(cluster_filename)
#         writer.SetInputData(polydata)
#         writer.Write()
    
#     return classified_edges


def extract_curvature_data(curved_mesh):
    """
    Extracts curvature data from the mesh.

    @param curved_mesh: The mesh data with curvature values as a vtkPolyData object.
    @return: A list of curvature values.
    """
    curvature_data = curved_mesh.GetPointData().GetScalars()
    num_points = curved_mesh.GetNumberOfPoints()
    curvature_values = []
    for i in range(num_points):
        curvature_values.append(curvature_data.GetValue(i))
    return curvature_values

def compute_bounding_box(mesh):
    """
    Computes the bounding box of the mesh.

    @param mesh: The mesh data as a vtkPolyData object.
    @return: A tuple containing the minimum and maximum bounds of the mesh.
    """
    bounds = mesh.GetBounds()
    min_bounds = [bounds[0], bounds[2], bounds[4]]
    max_bounds = [bounds[1], bounds[3], bounds[5]]
    return min_bounds, max_bounds

def compute_surface_normals(mesh):
    """
    Computes the surface normals of the mesh.

    @param mesh: The mesh data as a vtkPolyData object.
    @return: A tuple containing the number of outward-facing and inward-facing normals.
    """
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(mesh)
    normals_filter.ComputePointNormalsOff()
    normals_filter.ComputeCellNormalsOn()
    normals_filter.Update()
    
    normals = normals_filter.GetOutput().GetCellData().GetNormals()
    num_normals = normals.GetNumberOfTuples()
    
    outward_facing = 0
    inward_facing = 0
    
    for i in range(num_normals):
        normal = normals.GetTuple(i)
        if np.dot(normal, normal) > 0:  # Simple check assuming normals are unit vectors
            outward_facing += 1
        else:
            inward_facing += 1
    
    return outward_facing, inward_facing

def compute_facet_areas(mesh):
    """
    Computes the minimum and maximum facet areas of the mesh.

    @param mesh: The mesh data as a vtkPolyData object.
    @return: A tuple containing the minimum and maximum facet areas.
    """
    areas = []
    for i in range(mesh.GetNumberOfCells()):
        cell = mesh.GetCell(i)
        points = cell.GetPoints()
        if points.GetNumberOfPoints() == 3:
            pt1 = np.array(points.GetPoint(0))
            pt2 = np.array(points.GetPoint(1))
            pt3 = np.array(points.GetPoint(2))
            edge1 = pt2 - pt1
            edge2 = pt3 - pt1
            area = np.linalg.norm(np.cross(edge1, edge2)) / 2
            areas.append(area)
    return min(areas), max(areas)

def compute_edge_lengths(mesh):
    """
    Computes the minimum and maximum edge lengths of the mesh.

    @param mesh: The mesh data as a vtkPolyData object.
    @return: A tuple containing the minimum and maximum edge lengths.
    """
    edge_lengths = []
    for i in range(mesh.GetNumberOfCells()):
        cell = mesh.GetCell(i)
        points = cell.GetPoints()
        if points.GetNumberOfPoints() == 3:
            pt1 = np.array(points.GetPoint(0))
            pt2 = np.array(points.GetPoint(1))
            pt3 = np.array(points.GetPoint(2))
            edge_lengths.append(np.linalg.norm(pt2 - pt1))
            edge_lengths.append(np.linalg.norm(pt3 - pt1))
            edge_lengths.append(np.linalg.norm(pt3 - pt2))
    return min(edge_lengths), max(edge_lengths)

def compute_aspect_ratios(mesh):
    """
    Computes the minimum and maximum aspect ratios of the mesh.

    @param mesh: The mesh data as a vtkPolyData object.
    @return: A tuple containing the minimum and maximum aspect ratios.
    """
    aspect_ratios = []
    for i in range(mesh.GetNumberOfCells()):
        cell = mesh.GetCell(i)
        points = cell.GetPoints()
        if points.GetNumberOfPoints() == 3:
            pt1 = np.array(points.GetPoint(0))
            pt2 = np.array(points.GetPoint(1))
            pt3 = np.array(points.GetPoint(2))
            edge1 = np.linalg.norm(pt2 - pt1)
            edge2 = np.linalg.norm(pt3 - pt1)
            edge3 = np.linalg.norm(pt3 - pt2)
            edges = [edge1, edge2, edge3]
            aspect_ratio = max(edges) / min(edges)
            aspect_ratios.append(aspect_ratio)
    return min(aspect_ratios), max(aspect_ratios)

def is_point_inside(mesh, point):
    """
    Checks if a point is inside the geometry using vtkSelectEnclosedPoints.

    @param mesh: The mesh data as a vtkPolyData object.
    @param point: The point coordinates to check.
    @return: True if the point is inside the geometry, False otherwise.
    """
    enclosed_points = vtk.vtkSelectEnclosedPoints()
    enclosed_points.SetSurfaceData(mesh)
    
    points = vtk.vtkPoints()
    points.InsertNextPoint(point)
    
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    enclosed_points.SetInputData(polydata)
    enclosed_points.Update()
    
    return enclosed_points.IsInside(0)

# def find_inside_point(mesh, center_of_mass, min_bounds, max_bounds, initial_distance=0.1, step_factor=0.5):
#     """
#     Finds a point inside the geometry by moving inward from the center of mass along a deterministic direction.

#     @param mesh: The mesh data as a vtkPolyData object.
#     @param center_of_mass: The center of mass of the geometry.
#     @param min_bounds: The minimum bounds of the bounding box.
#     @param max_bounds: The maximum bounds of the bounding box.
#     @param initial_distance: The initial distance to move inward from the center of mass.
#     @param step_factor: The factor by which to reduce the distance at each step.
#     @return: A list representing the coordinates of the inside point.
#     """
#     direction = np.array([1.0, 1.0, 1.0], dtype=np.float64)  # Fixed direction for deterministic approach
#     direction /= np.linalg.norm(direction)
#     distance = initial_distance
#     inside_point = np.array(center_of_mass, dtype=np.float64) - direction * distance
    
#     while not is_point_inside(mesh, inside_point):
#         distance *= step_factor
#         inside_point = np.array(center_of_mass, dtype=np.float64) - direction * distance
#         if distance < 1e-6:  # Break if the distance becomes too small to avoid infinite loop
#             break

#     return inside_point.tolist()

def find_inside_point(mesh, coefDistance=1e-01):
    """
    Finds a point inside the geometry by moving inward from the center of a face along the normal direction 
    @param mesh: The mesh data as a vtkPolyData object.
    @return: A list representing the coordinates of the inside point.
    """
    # Get the points and polygons from the polydata
    points = mesh.GetPoints()
    polygons = mesh.GetPolys()

    # Initialize cell array iterator
    cell_iterator = polygons.NewIterator()
    cell_iterator.GoToFirstCell()
    cell_id_list = vtk.vtkIdList()

    # Get the first polygon (for simplicity, we take the first face)
    cell_iterator.GetCurrentCell(cell_id_list)

    # Get the points of the polygon
    point_ids = [cell_id_list.GetId(i) for i in range(cell_id_list.GetNumberOfIds())]
    face_points = vtk.vtkPoints()
    for pid in point_ids:
        face_points.InsertNextPoint(points.GetPoint(pid))

    # Compute the normal of the face using vtkPolygon
    polygon = vtk.vtkPolygon()
    polygon.GetPoints().DeepCopy(face_points)
    
    normal = [0.0, 0.0, 0.0]
    polygon.ComputeNormal(polygon.GetPoints(), normal)

    p0 = face_points.GetPoint(0)  # We just take the first point for simplicity
    p1 = face_points.GetPoint(1)
    p2 = face_points.GetPoint(2)
    pf =[ (p0[i]+p1[i]+p2[i])/3.0 for i in range(3)]
    distance = [p0[i] - pf[i] for i in range(3)]
    magDistance = math.sqrt(sum(d ** 2 for d in distance))

    # Function to move inward from the selected point along the normal
    inward_point = [pf[i] - coefDistance * magDistance * normal[i] for i in range(3)]
    return inward_point

def compute_principal_axes(mesh):
    """
    Computes the principal axes of the mesh by calculating the inertia tensor manually.

    @param mesh: The mesh data as a vtkPolyData object.
    @return: The principal axes as an array of eigenvectors.
    """
    points = mesh.GetPoints()
    num_points = points.GetNumberOfPoints()
    center_of_mass = np.zeros(3)

    for i in range(num_points):
        point = np.array(points.GetPoint(i))
        center_of_mass += point
    center_of_mass /= num_points

    inertia_tensor = np.zeros((3, 3))

    for i in range(mesh.GetNumberOfCells()):
        cell = mesh.GetCell(i)
        p0 = np.array(points.GetPoint(cell.GetPointId(0)))
        p1 = np.array(points.GetPoint(cell.GetPointId(1)))
        p2 = np.array(points.GetPoint(cell.GetPointId(2)))

        # Move points to center of mass coordinate system
        p0 -= center_of_mass
        p1 -= center_of_mass
        p2 -= center_of_mass

        # Compute the inertia tensor for this triangle
        for p in [p0, p1, p2]:
            for j in range(3):
                for k in range(3):
                    inertia_tensor[j, k] += (p[j] * p[k])

    _, eigenvectors = np.linalg.eigh(inertia_tensor)
    return eigenvectors

def find_outside_point(mesh, center_of_mass, min_bounds, max_bounds, initial_distance=0.1, buffer=10, max_buffer=20):
    """
    Finds a point outside the geometry by moving outward from the center of mass along a principal axis direction.

    @param mesh: The mesh data as a vtkPolyData object.
    @param center_of_mass: The center of mass of the geometry.
    @param min_bounds: The minimum bounds of the bounding box.
    @param max_bounds: The maximum bounds of the bounding box.
    @param initial_distance: The initial distance to move outward from the center of mass.
    @param buffer: The minimum buffer size added to the bounding box for the blockMeshDict.
    @param max_buffer: The maximum buffer size allowed for the bounding box.
    @return: A list representing the coordinates of the outside point.
    """
    eigenvectors = compute_principal_axes(mesh)
    direction = eigenvectors[:, 0]  # Using the first principal axis direction
    direction /= np.linalg.norm(direction)

    bbox_center = (np.array(min_bounds) + np.array(max_bounds)) / 2.0
    bbox_size = np.array(max_bounds) - np.array(min_bounds)
    max_distance = np.linalg.norm(bbox_size) + max_buffer  # Max distance based on the bounding box and max buffer

    distance = initial_distance
    outside_point = np.array(center_of_mass, dtype=np.float64) + direction * distance

    # Loop to find a point outside the geometry but within max distance
    while distance <= max_distance:
        if not is_point_inside(mesh, outside_point):
            break
        distance += initial_distance
        outside_point = np.array(center_of_mass, dtype=np.float64) + direction * distance

    # Ensure the outside point is within 70% to 90% of the extended bounding box
    extended_min_bounds, extended_max_bounds = create_extended_bounding_box(min_bounds, max_bounds, buffer)
    for i in range(3):
        extended_range = extended_max_bounds[i] - extended_min_bounds[i]
        lower_bound = extended_min_bounds[i] + 0.7 * extended_range
        upper_bound = extended_max_bounds[i] - 0.1 * extended_range
        outside_point[i] = np.clip(outside_point[i], lower_bound, upper_bound)

    # Debug information
    print(f"Outside point distance: {distance}, max_distance: {max_distance}, outside_point: {outside_point}")

    return outside_point.tolist()

def create_extended_bounding_box(min_bounds, max_bounds, buffer):
    """
    Creates an extended bounding box based on the original bounding box and buffer.

    @param min_bounds: The minimum bounds of the original bounding box.
    @param max_bounds: The maximum bounds of the original bounding box.
    @param buffer: The buffer size to extend the bounding box.
    @return: The extended minimum and maximum bounds.
    """
    extended_min_bounds = [min_bounds[i] - buffer for i in range(3)]
    extended_max_bounds = [max_bounds[i] + buffer for i in range(3)]
    return extended_min_bounds, extended_max_bounds

def compute_geometry_center_and_lengths(min_bounds, max_bounds):
    """
    Computes the center of the geometry and the length in each direction.

    @param min_bounds: The minimum bounds of the bounding box.
    @param max_bounds: The maximum bounds of the bounding box.
    @return: The center of the geometry and the lengths in each direction.
    """
    center = [(min_bounds[i] + max_bounds[i]) / 2 for i in range(3)]
    lengths = [(max_bounds[i] - min_bounds[i]) for i in range(3)]
    return center, lengths

def write_json_report(filename, volume=None, surface_area=None, center_of_mass=None, 
                      min_bounds=None, max_bounds=None, curvature_values=None, 
                      surface_normals=None, min_area=None, max_area=None,
                      min_edge_length=None, max_edge_length=None,
                      min_aspect_ratio=None, max_aspect_ratio=None,
                      inside_point=None, outside_point=None):
    """
    Writes a JSON report with the computed metrics.

    @param filename: The path to the STL file.
    @param volume: The volume of the geometry.
    @param surface_area: The surface area of the geometry.
    @param center_of_mass: The center of mass of the geometry.
    @param min_bounds: The minimum bounds of the geometry.
    @param max_bounds: The maximum bounds of the geometry.
    @param curvature_values: The curvature values of the geometry.
    @param surface_normals: The number of outward-facing and inward-facing normals.
    @param min_area: The minimum facet area of the geometry.
    @param max_area: The maximum facet area of the geometry.
    @param min_edge_length: The minimum edge length of the geometry.
    @param max_edge_length: The maximum edge length of the geometry.
    @param min_aspect_ratio: The minimum aspect ratio of the geometry.
    @param max_aspect_ratio: The maximum aspect ratio of the geometry.
    @param inside_point: The coordinates of a point inside the geometry.
    @param outside_point: The coordinates of a point outside the geometry.
    """
    base_name = os.path.splitext(os.path.basename(filename))[0]
    json_output_file = f"{base_name}_report.json"
    
    report = {"filename": filename}
    
    if volume is not None:
        report["volume"] = volume
    if surface_area is not None:
        report["surface_area"] = surface_area
    if center_of_mass is not None:
        report["center_of_mass"] = center_of_mass
    if min_bounds is not None and max_bounds is not None:
        center, lengths = compute_geometry_center_and_lengths(min_bounds, max_bounds)
        report["bounding_box"] = {
            "min_bounds": min_bounds,
            "max_bounds": max_bounds,
            "center": center,
            "lengths": lengths
        }
    if curvature_values is not None:
        report["curvature_values"] = curvature_values
    if surface_normals is not None:
        report["surface_normals"] = {
            "outward_facing": surface_normals[0],
            "inward_facing": surface_normals[1]
        }
    if min_area is not None and max_area is not None:
        report["min_area"] = {"value": min_area, "units": "square meters"}
        report["max_area"] = {"value": max_area, "units": "square meters"}
    if min_edge_length is not None and max_edge_length is not None:
        report["min_edge_length"] = {"value": min_edge_length, "units": "meters"}
        report["max_edge_length"] = {"value": max_edge_length, "units": "meters"}
    if min_aspect_ratio is not None and max_aspect_ratio is not None:
        report["min_aspect_ratio"] = min_aspect_ratio
        report["max_aspect_ratio"] = max_aspect_ratio
    if inside_point is not None:
        report["inside_point"] = {"coordinates": inside_point, "units": "meters"}
    if outside_point is not None:
        report["outside_point"] = {"coordinates": outside_point, "units": "meters"}

    with open(json_output_file, 'w') as outfile:
        json.dump(report, outfile, indent=4)

def generate_blockMeshDict(min_bounds, max_bounds, minCellNumber=10, buffer=-1):
    """
    Generates the blockMeshDict for OpenFOAM based on bounding box values.

    @param min_bounds: The minimum bounds of the geometry.
    @param max_bounds: The maximum bounds of the geometry.
    @cellRefinement: The scaler which can refine or coarse the base grid
    @param buffer: Buffer size to add around the bounding box. Any negative value means it is calculate automaically.
    """
    xmin, ymin, zmin = min_bounds
    xmax, ymax, zmax = max_bounds

    # Calculate bounding box dimensions
    dx = xmax - xmin
    dy = ymax - ymin
    dz = zmax - zmin

    dmin = min(dx,dy,dz)

    # Calculate buffer length automatically based on bounding box
    if(buffer < 0):
        buffer = 0.01*dmin
    
    #calculate grid cell numbers in each direction
    xnCells = int(dx/dmin*minCellNumber)
    ynCells = int(dy/dmin*minCellNumber)
    znCells = int(dz/dmin*minCellNumber) 

    vertices = [
        (xmin - buffer, ymin - buffer, zmin - buffer),
        (xmax + buffer, ymin - buffer, zmin - buffer),
        (xmax + buffer, ymax + buffer, zmin - buffer),
        (xmin - buffer, ymax + buffer, zmin - buffer),
        (xmin - buffer, ymin - buffer, zmax + buffer),
        (xmax + buffer, ymin - buffer, zmax + buffer),
        (xmax + buffer, ymax + buffer, zmax + buffer),
        (xmin - buffer, ymax + buffer, zmax + buffer),
    ]

    blockMeshDict = """/*--------------------------------*- C++ -*----------------------------------*\\
Web:         https://ianus-simulation.de/en/         
\*---------------------------------------------------------------------------*/
    FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
//*---------------------------------------------------------------------------*/

convertToMeters 1.0;

vertices
(
    {}
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({} {} {}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    enclosure
    {{
        type wall;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
            (0 1 5 4)
            (2 3 7 6)
            (1 2 6 5)
            (3 0 4 7)
        );
    }}
);

mergePatchPairs
(
);
""".format(
        "\n    ".join([f"({v[0]} {v[1]} {v[2]})" for v in vertices]),
        xnCells, ynCells, znCells
    )

    os.makedirs('system', exist_ok=True)
    with open('system/blockMeshDict', 'w') as f:
        f.write(blockMeshDict)


def generate_snappyHexMeshDict(inside_point):
    """
    Generates the snappyHexMeshDict for OpenFOAM based on inside point.

    @param inside_point: The point inside the geometry.
    """
    xin, yin, zin = inside_point

    snappyHexMeshDict = """/*--------------------------------*- C++ -*----------------------------------*\\
Web:         https://ianus-simulation.de/en/         
//*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
// Which of the steps to run
castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{
    /*inlet.stl
    {{
        type triSurfaceMesh;
        name inlet;
    }}*/
    walls.stl
    {{
        type triSurfaceMesh;
        name walls;
    }}
    /*features.stl
    {{
        type triSurfaceMesh;
        name features;
    }}*/
}};

castellatedMeshControls
{{
    maxLocalCells   200000000;
    maxGlobalCells  300000000;
    minRefinementCells 10;
    nCellsBetweenLevels 4;
    maxLoadUnbalance 0.1;
    allowFreeStandingZoneFaces true;
    resolveFeatureAngle      30;
    //gapLevelIncrement 2;
    features
    (
        /*{{
            file "walls.eMesh";
            level 1;
        }}
        {{
            file "inlet.eMesh";
            level 1;
        }}*/
        {{
            file "featureEdges.vtk";
            level 3;
        }}
    );

    refinementSurfaces
    {{
        walls
        {{
            level (3 3);
            curvatureLevel (5 0 3 -1);
            patchInfo
            {{
                type walls;
                inGroups (walls);
            }}
        }}
        inlet
        {{
            level (3 3);
            patchInfo
            {{
                type patch;
                inGroups (inlet);
            }}
        }}
        /*features
        {{
            level (1 4);
        }}*/
    }}

    refinementRegions
    {{
        walls
        {{
            mode    distance;
            levels  ((1 1));
            gapLevel (6 0 2);		//refine gaps to have a min of 5 cells, refine up to LV4 to reach 5 cells
            gapMode     mixed;
        }}
    }}
    
    locationInMesh ({} {} {});
}}

snapControls
{{
    nSmoothPatch 5;
    tolerance 2.0;
    nSolveIter 100; //The higher the value the better the body fitted mesh. The default value is 30. If you are having problems with the mesh quality (related to the snapping step), try to increase this value to 300.
    nRelaxIter 5; //Increase this value to improve the quality of the body fitted mesh.
    nFeatureSnapIter 10; //Increase this value to improve the quality of the edge features.
    multiRegionFeatureSnap false;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
}}

addLayersControls
{{
    relativeSizes true;
    layers{{}}
    expansionRatio 1.2;
    firstLayerThickness 0.3;
    minThickness 1e-09;
    nGrow 0;
    featureAngle 180;
    slipFeatureAngle 30;
    nRelaxIter 5;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 130;
    minMedialAxisAngle 130;
    nBufferCellsNoExtrude 0;
    nLayerIter 25;
}}

meshQualityControls
{{
    maxNonOrtho          60;
    maxBoundarySkewness  3.5;
    maxInternalSkewness  3.5;
    maxConcave           80;
    minFlatness          0.5;
    minVol               1e-30;
    minArea              -1;
    minTwist             0.02;
    minDeterminant       0.001;
    minFaceWeight        0.05;
    minVolRatio          0.01;
    minTriangleTwist     -1;
    minTetQuality        1e-30;
    nSmoothScale         4;
    errorReduction       0.75;
}}

// Advanced
debug 0;

// Merge tolerance. Is fraction of overall bounding box of initial mesh.
// Note: the write tolerance needs to be higher than this.
mergeTolerance 1E-6;
""".format(xin, yin, zin)

    os.makedirs('system', exist_ok=True)
    with open('system/snappyHexMeshDict', 'w') as f:
        f.write(snappyHexMeshDict)

def parse_arguments():
    """
    Parse command-line arguments.

    @return: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="STL Geometry Analysis using VTK")
    parser.add_argument("filename", type=str, help="Path to the STL file")
    parser.add_argument("--volume", action="store_true", help="Compute volume")
    parser.add_argument("--surface_area", action="store_true", help="Compute surface area")
    parser.add_argument("--center_of_mass", action="store_true", help="Compute center of mass")
    parser.add_argument("--bounding_box", action="store_true", help="Compute bounding box")
    parser.add_argument("--curvature", action="store_true", help="Compute curvature")
    parser.add_argument("--surface_normals", action="store_true", help="Compute surface normals")
    parser.add_argument("--facet_areas", action="store_true", help="Compute facet areas")
    parser.add_argument("--edge_lengths", action="store_true", help="Compute edge lengths")
    parser.add_argument("--aspect_ratios", action="store_true", help="Compute aspect ratios")
    parser.add_argument("--inside_outside_points", action="store_true", help="Compute inside and outside points")
    parser.add_argument("--generate_blockMeshDict", action="store_true", help="Generate blockMeshDict for OpenFOAM")
    parser.add_argument("--buffer", type=float, default=-1, help="Buffer size to add around the bounding box")
    parser.add_argument("--minCellNumber", type=float, default=5, help="represent the cell numbers in the shortest edge")
    parser.add_argument("--max_buffer", type=float, default=20, help="Maximum buffer size allowed for the bounding box")
    parser.add_argument("--generate_snappyHexMeshDict", action="store_true", help="Generate snappyHexMeshDict for OpenFOAM")
    parser.add_argument("--feature_angle", type=float, default=30, help="Angle to extract edges")
    parser.add_argument("--feature_edge", action="store_true", help="Compute feature edges")

    return parser.parse_args()