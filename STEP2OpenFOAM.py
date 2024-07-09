


import bpy
import numpy as np
import mathutils


def IterRayCast(obj, ray_origin, ray_direction, face_index = 0, delta = 1e-6, deltamax= 1e-4, iter = 0, itermax = 100,):

    #shift the origin of the ray cast away from the center of the face (which is equal to ray_origin)
    ray_origin_shifted = ray_origin + ray_direction * delta
    result, ray_hit_location, _, index = obj.ray_cast(ray_origin_shifted, ray_direction)

    if result and index == face_index: # if the raycast hits a face but it hits the same face ...
        if iter < itermax: # and if we didnt exceed max. no. iterations ...
            if delta < deltamax: # and our delta is within the threshold...

                iter += 1 # ... then increase our iteration counter
                delta *= 2 # ... double the threshold
                
                print('IterRayCast: The ray case hit the same face it was cast from. \
                      Increasing delta and trying again, iteration = ' \
                      + str(iter) + ', delta = ' + str(delta) + '...')
                
                # ... and iteratively repeat this function until a result is found or our conditions are violated.
                ray_hit_location, iter, ray_origin_shifted, result, index = \
                    IterRayCast(obj, ray_origin, ray_direction, face_index, delta, deltamax, iter, itermax, )
                
            else: 
                raise Exception('IterRayCast: The raycast reached the maximum delta threshold.' \
                                'Try increasing the maximum threshold but validate that the point is actually inside the geometry.')
        else: 
            raise Exception('IterRayCast: Maximum number of iterations reached (itermax = ' + str(itermax) + ').' \
                            'Try increasing maximum number of iterations.')
    elif not result: 
        raise Exception('IterRayCast: The raycast did not hit a face.' \
                        'This can be because the mesh is not manifold or because the face normals are inverted.')

    return ray_hit_location, iter, ray_origin_shifted, result, index


def FindInsidePoint(obj, face_index = 0, delta = 1e-6, deltamax= 1e-4):

    mesh = obj.data

    try: face = mesh.polygons[face_index]
    except: raise Exception('FindInsidePoint: Face Index out of bounds for given object.' \
                            'Try decreasing face index. (face_index = ' + str(face_index) + \
                            ', No. polygons = ' + str(len(mesh.polygons)) + ')')
    
    face_center_local = face.center
    face_normal_local = face.normal

    ray_origin = face_center_local
    # ray_shift = - face_normal_local
    ray_direction = - face_normal_local

    ray_hit_location, iter, ray_origin_shifted, _, _ = IterRayCast(obj, ray_origin, ray_direction, 0, delta, deltamax)

    # print('FindInsidePoint: Raycast succeeded after ' + str(iter) + ' iterations.')

    inside_pos = ( ray_origin_shifted + ray_hit_location ) / 2 

    return inside_pos


def GetInsidePointsForFaceIndices(obj, face_indices, delta = 1e-6, deltamax=1e-4):
    points = [[] for i in range(len(face_indices))]
    for i, face_idx in enumerate(face_indices):
        points[i] = FindInsidePoint(obj, face_idx, delta, deltamax)
    return points


def Set3DCursorToInsideGeometry(obj, pos):

    # delta = 1e-6
    # deltamax = 1e-4
    # face_idx = 0
    world_matrix = obj.matrix_world

    # inside_pos_local = FindInsidePoint(obj, face_idx, delta, deltamax)

    bpy.context.scene.cursor.location = world_matrix @ pos

    # return inside_pos_local



def GetRandomSphericalPoints(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def CastRandomRays(obj, ray_origin, no_rays):

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

    return no_hit_flag, mindist


def FindOptimalPoint(obj, points, no_rays):
    N = len(points)
    no_hit_flags = [False for i in range(N)]
    mindists = [0 for i in range(N)]

    for i, point in enumerate(points):
        no_hit_flags[i], mindists[i] = CastRandomRays(obj, point, no_rays)

    mindist = min(mindists)
    mindist_idx = mindists.index(mindist)

    return points[mindist_idx], mindist


if __name__ == "__main__":

    deterministic = True
    seed = 42
    
    if deterministic:
        np.random.seed(seed)

    obj = bpy.context.active_object
    # loc = Set3DCursorToInsideGeometry(obj)


    face_indices = [i for i in range(20)]
    points = GetInsidePointsForFaceIndices(obj, face_indices)
    
    optpoint, mindist = FindOptimalPoint(obj, points, 1000)

    print(mindist)
    Set3DCursorToInsideGeometry(obj, optpoint)    












    # This is a place to park potentially useful code snippets:

    # world_matrix = obj.matrix_world
    # obj_loc = obj.location
    # face_center = world_matrix @ face_center_local
    # face_normal = world_matrix.to_3x3() @ face_normal_local
    # shift the raycast origin away slightly so it doesnt hit the same face it is cast from
    # face_center_local_shifted = face_center_local - face_normal_local*delta
    # ray_origin = face_center_local_shifted