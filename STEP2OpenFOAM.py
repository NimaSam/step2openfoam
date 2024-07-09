# -*- coding: utf-8 -*-
# author: Lukas Kilian
# Step 2 OpenFOAM Pipeline
# Date of creation: 09.07.2024



import bpy
import numpy as np
import mathutils
import random



def IterRayCast(obj, ray_origin, ray_direction, face_index = 0, delta = 1e-6, deltamax= 1e-4, iter = 0, itermax = 100,):
    '''Iterative core function to find the inside point of a given object <obj> with defined 
    ray origin <ray_origin> and ray direction <ray_direction>. Handels exceptions when no point is found
    or ray cast fails due to other reasons.'''

    #shift the origin of the ray cast away from the center of the face (which is equal to ray_origin)
    ray_origin_shifted = ray_origin + ray_direction * delta
    result, ray_hit_location, _, index = obj.ray_cast(ray_origin_shifted, ray_direction)

    if result and index == face_index: # if the raycast hits a face but it hits the same face ...
        if iter < itermax: # and if we didnt exceed max. no. iterations ...
            if delta < deltamax: # and our delta is within the threshold...

                iter += 1 # ... then increase our iteration counter
                delta *= 2 # ... double the threshold
                
                print('IterRayCast: The ray case hit the same face it was cast from.' \
                      'Increasing delta and trying again, iteration = ' \
                      + str(iter) + ', delta = ' + str(delta) + '...')
                
                # ... and iteratively repeat this function until a result is found or our conditions are violated.
                ray_hit_location, iter, ray_origin_shifted, result, index = \
                    IterRayCast(obj, ray_origin, ray_direction, face_index, delta, deltamax, iter, itermax, )
                
            else: 
                Set3DCursorToLocation(obj,ray_origin)
                print('Face idx = ' + str(face_index) + ', 3D Cursor set to face where error occurred.')
                raise Exception('IterRayCast: The raycast reached the maximum delta threshold. ' \
                                'Try increasing the maximum threshold but validate that the point is actually inside the geometry.')
        else: 
            Set3DCursorToLocation(obj,ray_origin)
            print('Face idx = ' + str(face_index) + ', 3D Cursor set to face where error occurred.')
            raise Exception('IterRayCast: Maximum number of iterations reached (itermax = ' + str(itermax) + '). ' \
                            'Try increasing maximum number of iterations.')
    elif not result: 
        Set3DCursorToLocation(obj,ray_origin)
        print('Face idx = ' + str(face_index) + ', 3D Cursor set to face where error occurred.')
        raise Exception('IterRayCast: The raycast did not hit a face. ' \
                        'This can be because the mesh is not manifold or because the face normals are inverted.')

    return ray_hit_location, iter, ray_origin_shifted, result, index



def FindInsidePoint(obj, face_index = 0, delta = 1e-6, deltamax= 1e-4):
    '''Finds a point inside the mesh by casting a ray from the <face_index>-th face of the Object <obj>,
    and returning the midway point between ray cast origin and ray cast hit.'''

    mesh = obj.data

    # check whether specified face_index is out of bounds for given obj
    try: face = mesh.polygons[face_index]
    except: raise Exception('FindInsidePoint: Face Index out of bounds for given object. ' \
                            'Try decreasing face index. (face_index = ' + str(face_index) + \
                            ', No. polygons = ' + str(len(mesh.polygons)) + ')')
    
    face_center_local = face.center
    face_normal_local = face.normal

    ray_origin = face_center_local
    ray_direction = - face_normal_local

    ray_hit_location, iter, ray_origin_shifted, _, _ = IterRayCast(obj, ray_origin, ray_direction, face_index, delta, deltamax)

    # print('FindInsidePoint: Raycast succeeded after ' + str(iter) + ' iterations.')

    inside_pos = ( ray_origin_shifted + ray_hit_location ) / 2 

    return inside_pos



def GetInsidePointsForFaceIndices(obj, face_indices, delta = 1e-6, deltamax=1e-4):
    '''Finds the Inside Point for all Faces with indices <face_indices> of a given Object <obj>.'''
    points = [[] for i in range(len(face_indices))]
    for i, face_idx in enumerate(face_indices):
        points[i] = FindInsidePoint(obj, face_idx, delta, deltamax)
    return points



def Set3DCursorToLocation(obj, local_pos):
    '''Places the Blender 3D cursor at the global position of the 
    local position <local_pos> in Object <obj> space. '''
    world_matrix = obj.matrix_world
    bpy.context.scene.cursor.location = world_matrix @ local_pos



def Reset3DCursor():
    '''Resets Location of 3D cursor to (0,0,0)'''
    bpy.context.scene.cursor.location = mathutils.Vector((0,0,0))


def GetRandomSphericalPoints(npoints, ndim=3):
    '''Returns a distribution of <npoints> random, uniformly distributed, 
    normalized vectors in <ndim> space. '''
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec



def GetMinDist(obj, ray_origin, no_rays):
    '''
    Casts <no_rays> random, uniformly distributed rays originating from <ray_origin> 
    out in 3D space within <obj> geometry to find the minium distance to the nearest surface. 
    Also determines whether a  ray doesn't hit a face in no_hit_flag, which would 
    indicate a non-manifold mesh or incorrect face normals.
    '''

    no_hit_flag = False

    xi, yi, zi = GetRandomSphericalPoints(no_rays)
    results = [[] for i in range(no_rays)]
    ray_hits = [[] for i in range(no_rays)]

    for i in range(no_rays):
        ray_direction = mathutils.Vector((xi[i], yi[i], zi[i]))
        results[i], ray_hits[i], _, _ = obj.ray_cast(ray_origin, ray_direction)

    no_hit_flag =  False in results # Set this flag to true if there is a single ray which didn't hit a face
    distances = [(ray_hits[i] - ray_origin).length for i in range(no_rays)]
    mindist = min(distances)

    return mindist, no_hit_flag



def FindOptimalPoint(obj, points, no_rays):
    '''Determines the point of <points> with the most space to surrounding <obj> geometry. '''

    N = len(points)
    no_hit_flags = [False for i in range(N)]
    mindists = [0 for i in range(N)]

    for i, point in enumerate(points):
        mindists[i], no_hit_flags[i] = GetMinDist(obj, point, no_rays)

    mindist = min(mindists)
    mindist_idx = mindists.index(mindist)

    return points[mindist_idx], mindist



def GetRandomFaceIndices(obj, n_faces):
    '''Returns <n_faces> uniform, unique, random indices within the amount of polygons of <obj>'''

    N = len(obj.data.polygons)

    # clamp n_faces to not exceed actual no. of faces of object
    if n_faces > N: n_faces = N

    # get random, but unique sampling of indices within range of face indices
    indices = random.sample(range(0,N), n_faces)
    
    return indices


def SearchForPointWithThreshold(obj, delta = 1e-5, maxiter = 1000, no_rays = 50, no_rays_secondary = 1000):
    '''
    Searches randomly through inner points within the geometry until 
    a point with <delta> space in all directions is found 
    or until <maxiter> iterations are surpassed. 
    The intial check will be with <no_rays> rays. This number should be low for efficiency 
    If no value surpases <delta>, a secondary check is done with <no_rays_secondary> rays. 
    '''

    # as written above, this script will check faces randomly throughout the mesh and check
    # with a low ray number raycast whether any easily detectable object is close that already
    # violates the delta requirement, i.e. a face that is closer to the inner point than required
    # by delta. If that is not the case, a secondary, much more thorough check is done with a higher number
    # of rays to (probabilistically) validate that there actually is nothing closer than delta.
    # This script will keep running until maxiter iterations are surpassed or a point is found.


    face_indices = GetRandomFaceIndices(obj, maxiter)

    i = 0

    # points = GetInsidePointsForFaceIndices(obj, face_indices)
    while i < maxiter:
        point = FindInsidePoint(obj, face_indices[i])
        mindist, _ = GetMinDist(obj, point, no_rays)
        if mindist > delta:
            mindist, _ = GetMinDist(obj, point, no_rays_secondary)
            if mindist > delta:
                print('Optimal point found after ' + str(i) + ' iterations, mindist = ' + str(mindist))
                return point
        i+=1
    
    raise Exception('SearchForPointWithThreshold: No points found within ' + str(maxiter) + ' iterations. ' \
                    'Try increasing maximum interatios or decreasing delta threshold.')


def LoadSTEP(filepath, file, detail_level = 100):
    Reset3DCursor()
    try: 
        bpy.ops.import_scene.occ_import_step(filepath=filepath, 
                                             override_file=file, 
                                             detail_level = detail_level,
                                             hierarchy_types = 'EMPTIES')
    except:
        raise Exception('Blender STEPper addon not installed!')


if __name__ == "__main__":

    # Define whether the random distribution is deterministic
    deterministic = 0
    seed = 42
    
    if deterministic:
        np.random.seed(seed)
        random.seed(seed)

    filepath = "C:/Users/Luke/Desktop/stp_test/"
    file = "a.stp"
    detail_level = 1000

    exportpath = "C:/Users/Luke/Desktop/stp_test/export/"

    LoadSTEP(filepath, file, detail_level=detail_level)

    obj = bpy.context.active_object

    optpoint = SearchForPointWithThreshold(obj, delta=0.001, maxiter=50, no_rays=50, no_rays_secondary=5000)

    Set3DCursorToLocation(obj, optpoint)    


    # filepath="C:\\Users\\Luke\\Desktop\\stp_test\\"













    # This is a place to park potentially useful code snippets:


    # face_indices = [i for i in range(20)]
    # face_indices = GetRandomFaceIndices(obj, 50)
    # print(face_indices)
    # points = GetInsidePointsForFaceIndices(obj, face_indices)
    # optpoint, mindist = FindOptimalPoint(obj, points, no_rays = 1000)
    # print('Mindist = ' + str(mindist))



    # world_matrix = obj.matrix_world
    # obj_loc = obj.location
    # face_center = world_matrix @ face_center_local
    # face_normal = world_matrix.to_3x3() @ face_normal_local
    # shift the raycast origin away slightly so it doesnt hit the same face it is cast from
    # face_center_local_shifted = face_center_local - face_normal_local*delta
    # ray_origin = face_center_local_shifted