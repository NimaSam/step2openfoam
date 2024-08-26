# -*- coding: utf-8 -*-
# author: Lukas Kilian
# STL 2 OpenFOAM Pipeline
# Date of creation: 23.08.2024


bl_info = {
    "name" : "STL2OpenFOAM",
    "description" : "A pipeline for creating OpenFOAM data from .STL files",
    "author" : "Lukas Kilian",
    "version" : (0, 0, 2),
    "blender" : (4, 0, 0),
    # "location" : "View3D",
    # "warning" : "",
    "support" : "COMMUNITY",
    # "doc_url" : "",
    # "category" : "3D View"
}


openfoam_files = {}
openfoam_files['blockmesh'] = './sample_project/system/blockMeshDict'


progressbar_prefix_len = 30 # the length of the prefix string before printing the progress bar

version_string = 'v' + str(bl_info['version'][0]) + \
                 '.' + str(bl_info['version'][1]) + \
                 '.' + str(bl_info['version'][2])

# dependencies = ['snappyhexmesh_gui-master']

import os
import bpy
import sys
import json
import random
import argparse
import mathutils
# import addon_utils
import numpy as np

config_args = {}



# ███    ███ ██ ███████  ██████ 
# ████  ████ ██ ██      ██      
# ██ ████ ██ ██ ███████ ██      
# ██  ██  ██ ██      ██ ██      
# ██      ██ ██ ███████  ██████ 
                            
# Utility and helper scripts

def InfoMsg(msg, newline=False):
    '''Utility script to print unified info messages in the console.'''
    if newline: print('')
    print('## STL2OpenFoam Info: ' +msg)
    # print('')


def PrintNameWithVersion():
    '''Prints a fancy Ascii text with version number into the console'''

    n = len(version_string)
    nmax = 32
    print(r'''   
  ___ _____ _    ___ ___                 ___ ___   _   __  __ 
 / __|_   _| |  |_  ) _ \ _ __  ___ _ _ | __/ _ \ /_\ |  \/  |
 \__ \ | | | |__ / / (_) | '_ \/ -_) ' \| _| (_) / _ \| |\/| |
 |___/ |_| |____/___\___/| .__/\___|_||_|_| \___/_/ \_\_|  |_|
                         |_|  ''' 
          + (nmax-n)*' ' + version_string + '\n')



def LoadConfig(configpath):
    '''Loads the config file and returns the data.'''
    if not os.path.exists(configpath):
        raise Exception('STL2OpenFOAM Error: Config could not be found! Check configpath in main function!')

    with open(configpath, 'r') as f:
        config = json.load(f)
    
    return config




# stolen with courtesy from https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
# old fill symbol -> █
def PrintProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#', printEnd = "\r", fixedWidth = True):
    '''
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    '''
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    if fixedWidth:
        prefix += (progressbar_prefix_len - len(prefix)) * '.'
        if len(prefix)>progressbar_prefix_len: prefix = prefix[:progressbar_prefix_len]
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def _path_arg(configpath):
    if not os.path.exists(configpath):
        raise argparse.ArgumentTypeError('Invalid path, config file could not be found')
    return configpath


def SetupArgparser():
    parser = argparse.ArgumentParser(description='STL 2 OpenFOAM pipeline ' + version_string, prog='STL2OpenFOAM')
    parser.add_argument('config_path', help = 'Path to the config file', type = _path_arg)
    args = parser.parse_args()
    config_args['config'] = args.config_path





# ██████  ██      ███████ ███    ██ ██████  ███████ ██████  
# ██   ██ ██      ██      ████   ██ ██   ██ ██      ██   ██ 
# ██████  ██      █████   ██ ██  ██ ██   ██ █████   ██████  
# ██   ██ ██      ██      ██  ██ ██ ██   ██ ██      ██   ██ 
# ██████  ███████ ███████ ██   ████ ██████  ███████ ██   ██ 
                                                          
# All purely blender specific scripts

def Set3DCursorToLocation(obj, local_pos):
    '''Places the Blender 3D cursor at the global position of the 
    local position <local_pos> in Object <obj> space. '''
    world_matrix = obj.matrix_world
    bpy.context.scene.cursor.location = world_matrix @ local_pos



def SetActive(obj):
    '''Sets the current object as active and selected in Blender.'''
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)



def Reset3DCursor():
    '''Resets Location of 3D cursor to (0,0,0)'''
    bpy.context.scene.cursor.location = mathutils.Vector((0,0,0))



def GetFirstMeshInScene():
    '''
    Returns the first mesh object of the active scene. 
    This should be the first and only mesh after cleaning the blender scene and importing the STEP file. 
    '''
    
    objs = bpy.context.scene.objects
    meshes = [obj for obj in objs if obj.type == 'MESH']

    if len(meshes) == 0:
        print('STL2OpenFoam Warning: No mesh found in Scene!')
        return None
    elif len(meshes) > 1:
        print('STL2OpenFoam Warning: More than 1 object found after import, returning first object!')
    
    obj = meshes[0]
    return obj


def BlenderApplyRotScale():
    '''Apply Rotation and Scale of all objects in Blender file'''
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)


def BlenderPurgeOrphans():
    '''Purges all Orphans in the Blender file'''
    if bpy.app.version >= (3, 0, 0):
        bpy.ops.outliner.orphans_purge(
            do_local_ids=True, do_linked_ids=True, do_recursive=True
        )
    else:
        # call purge_orphans() recursively until there are no more orphan data blocks to purge
        result = bpy.ops.outliner.orphans_purge()
        if result.pop() != "CANCELLED":
            BlenderPurgeOrphans()


def DeleteAllBlenderData():
    '''Cleans all data in the current Blender file'''
    if bpy.context.active_object and bpy.context.active_object.mode == "EDIT":
        bpy.ops.object.editmode_toggle()
    for obj in bpy.data.objects:
        obj.hide_set(False)
        obj.hide_select = False
        obj.hide_viewport = False
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    collection_names = [col.name for col in bpy.data.collections]
    for name in collection_names:
        bpy.data.collections.remove(bpy.data.collections[name])
    BlenderPurgeOrphans()



def JoinAllMeshesAndAssignMaterialSlots():
    '''
    Joins all objects into one mesh and assigns materials
    based on the object names for the purpose of mesh
    quality assurance and later separation via the material slots
    '''

    objs = bpy.context.scene.objects

    if len(objs) <= 1: #if there is one or no objects, do nothing
        return
    
    
    
    for obj in objs:
        mat = bpy.data.materials.get(obj.name)
        if not mat:
            mat = bpy.data.materials.new(name = obj.name)
        if obj.data.materials:
            raise Warning('Object already has a material slot. This should not happen. Something is wrong with import or material assignment.')
        obj.data.materials.append(mat)

    target = objs[0] #join all objects into the first one

    target.name = 'joined_geometry' #setting names just for clarity inside blender
    target.data.name = 'data_joined_geometry'

    for obj in objs:
        obj.select_set(True)

    bpy.context.view_layer.objects.active = target

    bpy.ops.object.join()

    return target




def CleanMesh(obj, delete_non_manifold = False):
    '''
    Cleans the mesh of an obj. Specifically, remove doubles and merge tiny gaps, 
    recalculate normals, delete inside faces, delete loose vertices, edges and faces.
    '''

    print('\nCleaning mesh...')
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=1e-7)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.mesh.delete_loose(use_faces=True)
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_interior_faces()
    bpy.ops.mesh.delete(type='FACE')

    if delete_non_manifold:
        bpy.ops.mesh.select_mode(type='VERT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.mesh.select_non_manifold()
        bpy.ops.mesh.delete(type='VERT')

    bpy.ops.object.mode_set(mode = 'OBJECT')




def CheckMesh(obj):
    '''
    Checks the quality of a given mesh and outputs parameters into the console.
    '''
    print('\nChecking mesh...')
    me = obj.data
    no_v = len(me.vertices)
    no_f = len(me.polygons)

    def checkMeshPrint(msg, param):
        n = len(msg)
        m = 20
        if n>m:
            msg = msg[:m]
            n = m
        out = '>>> ' + msg + (m-n)*'.' + ' ' + str(param)
        print(out)

    face_lens = [len(x.vertices) for x in [p for p in me.polygons]]
    no_tris = face_lens.count(3)
    no_quads = face_lens.count(4)
    no_ngons = no_f - no_quads - no_tris

    checkMeshPrint('No. Vertices', no_v)
    checkMeshPrint('No. Faces', no_f)
    checkMeshPrint('... of that tris', no_tris)
    checkMeshPrint('... of that quads', no_quads)
    checkMeshPrint('... of that ngons', no_ngons)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='VERT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.mesh.select_non_manifold()
    bpy.ops.object.mode_set(mode = 'OBJECT')
    non_manifold_verts = [v for v in me.vertices if v.select]
    no_non_manifold = len(non_manifold_verts)

    checkMeshPrint('Non manifold', no_non_manifold)  
    


def SeparateGeometryByMaterialGroups(obj):
    '''
    Separates an obj by material slots and renames the resulting objs according to the latter.
    '''
    if not obj:
        raise Exception('SeparateGeometryByMaterialGroups: No obj given or obj is empty.')
    
    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.separate(type='MATERIAL')

    objs = bpy.data.objects
    for obj in objs:
        obj.name = obj.material_slots[0].name
        obj.data.name = 'mesh_' + obj.name



# may get back to this in the future for joining objects without ops. 
# unused for now.
# def appendMeshToMesh (passedHostOB, passedAppendOB, passedUpdate = True):
#     result = False
#     if passedHostOB != None:
#         if passedHostOB.type == 'MESH':
#             if passedAppendOB != None:
#                 if passedAppendOB.type == 'MESH':
#                     me_host = passedHostOB.data         # Fetch the host mesh.
#                     me_append = passedAppendOB.data     # Fetch the append mesh.
                    
#                     # Transfer the vertices from the append to the host.
#                     num_v = len(me_host.vertices)
#                     num_append_v = len(me_append.vertices)
#                     me_host.vertices.add(num_append_v)
#                     for v in range(num_append_v):
#                         # However, we need to offset the index by the number of verts in the host mesh we are appending to.
#                         me_host.vertices[num_v + v].co = passedHostOB.matrix_world.inverted() @ passedAppendOB.matrix_world @ me_append.vertices[v].co
#                         if me_append.vertices[v].select:
#                             me_host.vertices[num_v + v].select = True

#                     num_e = len(me_host.edges)
#                     num_append_e = len(me_append.edges)
#                     me_host.edges.add(num_append_e)
#                     for e in range(num_append_e):
#                         x_e = me_append.edges[e].vertices
#                         o_e = [i + num_e for i in x_e]
#                         me_host.edges[num_e + e].vertices = o_e

#                     # Now append the faces...
#                     num_f = len(me_host.polygons)
#                     num_append_f = len(me_append.polygons)
#                     me_host.polygons.add(num_append_f)
#                     for f in range(num_append_f):
#                         x_fv = me_append.polygons[f].vertices
#                         o_fv = [i + num_v for i in x_fv]
#                         # However, we need to offset the index by the number of polygons in the host mesh we are appending to.
#                         if len(x_fv) == 4: 
#                             me_host.polygons[num_f + f].vertices_raw = o_fv
#                         else: 
#                             me_host.polygons[num_f + f].vertices = o_fv
                    
#     #                 if passedUpdate == True:
#     #                     me_host.update(calc_edges=True)
#     #                 result = True
#     # return result




# ██ ███    ██ ███████ ██ ██████  ███████     ██████   ██████  ██ ███    ██ ████████ 
# ██ ████   ██ ██      ██ ██   ██ ██          ██   ██ ██    ██ ██ ████   ██    ██    
# ██ ██ ██  ██ ███████ ██ ██   ██ █████       ██████  ██    ██ ██ ██ ██  ██    ██    
# ██ ██  ██ ██      ██ ██ ██   ██ ██          ██      ██    ██ ██ ██  ██ ██    ██    
# ██ ██   ████ ███████ ██ ██████  ███████     ██       ██████  ██ ██   ████    ██    
                                                                                   
# All scripts which handle the localization of a point inside the mesh

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

    InfoMsg(' Attempting to find point inside mesh with delta = %s...'%(delta), True)

    prog_msg = 'Running raycast...'
    PrintProgressBar(0,maxiter, prefix = prog_msg, length = 40, fixedWidth=False)
    
    face_indices = GetRandomFaceIndices(obj, maxiter)

    i = 0

    # points = GetInsidePointsForFaceIndices(obj, face_indices)
    while i < maxiter:
        point = FindInsidePoint(obj, face_indices[i])
        mindist, _ = GetMinDist(obj, point, no_rays)
        PrintProgressBar(i+1,maxiter, prefix = prog_msg, length = 40, fixedWidth=False)
        if mindist > delta:
            mindist, _ = GetMinDist(obj, point, no_rays_secondary)
            if mindist > delta:
                PrintProgressBar(maxiter,maxiter, prefix = prog_msg, length = 40, fixedWidth=False)
                mindiststr = "{:.6f}".format(mindist)
                pointstr = ["{:.3f}".format(x) for x in point]
                InfoMsg('Optimal point found at (%s, %s, %s) after %s iterations, min dist = %s'
                        %(pointstr[0], pointstr[1], pointstr[2], i,mindiststr))
                return point
        i+=1

    raise Exception('STL2OpenFOAM Error: No points found within %s iterations. ' \
                    'Try increasing maximum interatios or decreasing delta threshold.'%(maxiter))



def FindInsidePoint(obj, face_index = 0, delta = 1e-6, deltamax= 1e-4):
    '''Finds a point inside the mesh by casting a ray from the <face_index>-th face of the Object <obj>,
    and returning the midway point between ray cast origin and ray cast hit.'''

    mesh = obj.data

    # check whether specified face_index is out of bounds for given obj
    try: face = mesh.polygons[face_index]
    except: raise Exception('STL2OpenFOAM Error: Face Index out of bounds for given object. ' \
                            'Try decreasing face index. (face_index = %s, No. polygons = %s)'
                            %(face_index,len(mesh.polygons)))
    
    face_center_local = face.center
    face_normal_local = face.normal

    ray_origin = face_center_local
    ray_direction = - face_normal_local

    ray_hit_location, iter, ray_origin_shifted, _, _ = IterRayCast(obj, ray_origin, ray_direction, face_index, delta, deltamax)

    # print('FindInsidePoint: Raycast succeeded after ' + str(iter) + ' iterations.')

    inside_pos = ( ray_origin_shifted + ray_hit_location ) / 2 

    return inside_pos




def IterRayCast(obj, ray_origin, ray_direction, face_index = 0, delta = 1e-6, deltamax= 1e-4, iter = 0, itermax = 100,):
    '''Iterative core function to find the inside point of a given object <obj> with defined 
    ray origin <ray_origin> and ray direction <ray_direction>. Throws exceptions when no point is found
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
                raise Exception('IterRayCast Error: The raycast reached the maximum delta threshold. ' \
                                'Try increasing the maximum threshold but validate that the point is actually inside the geometry.')
        else: 
            Set3DCursorToLocation(obj,ray_origin)
            print('Face idx = ' + str(face_index) + ', 3D Cursor set to face where error occurred.')
            raise Exception('IterRayCast Error: Maximum number of iterations reached (itermax = ' + str(itermax) + '). ' \
                            'Try increasing maximum number of iterations.')
    elif not result: 
        Set3DCursorToLocation(obj,ray_origin)
        print('Face idx = ' + str(face_index) + ', 3D Cursor set to face where error occurred.')
        raise Exception('IterRayCast Error: The raycast did not hit a face. ' \
                        'This can be because the mesh is not manifold or because the face normals are inverted.')

    return ray_hit_location, iter, ray_origin_shifted, result, index



def GetInsidePointsForFaceIndices(obj, face_indices, delta = 1e-6, deltamax=1e-4):
    '''Finds the Inside Point for all Faces with indices <face_indices> of a given Object <obj>.'''
    points = [[] for i in range(len(face_indices))]
    for i, face_idx in enumerate(face_indices):
        points[i] = FindInsidePoint(obj, face_idx, delta, deltamax)
    return points



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



def GetRandomSphericalPoints(npoints, ndim=3):
    '''Returns a distribution of <npoints> random, uniformly distributed, 
    normalized vectors in <ndim> space. '''
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec



def GetRandomFaceIndices(obj, n_faces):
    '''Returns <n_faces> uniform, unique, random indices within the amount of polygons of <obj>'''

    N = len(obj.data.polygons)

    # clamp n_faces to not exceed actual no. of faces of object
    if n_faces > N: n_faces = N

    # get random, but unique sampling of indices within range of face indices
    indices = random.sample(range(0,N), n_faces)
    
    return indices





# ██████  ██       ██████   ██████ ██   ██ ███    ███ ███████ ███████ ██   ██ 
# ██   ██ ██      ██    ██ ██      ██  ██  ████  ████ ██      ██      ██   ██ 
# ██████  ██      ██    ██ ██      █████   ██ ████ ██ █████   ███████ ███████ 
# ██   ██ ██      ██    ██ ██      ██  ██  ██  ██  ██ ██           ██ ██   ██ 
# ██████  ███████  ██████   ██████ ██   ██ ██      ██ ███████ ███████ ██   ██ 
                                                                            
                                                                            

def WriteBlockMesh(obj, ndim, buffer = 0.1):
    '''
    Writes a BlockMeshDict of a given blender <obj>, 
    creates a bounding box enlarged by <buffer> percent 
    (e.g. 0.1 means the bounding box is 10% bigger, aka 110% of the actual size),
    divides the largest dimension into <ndim> blocks and calculates the remaining 
    block divisions accordingly.
    '''
    bbdata = GetBoundingBox(obj)

    outputfile = './outputfile.txt'
    blockmeshinput = openfoam_files['blockmesh'] 

    dx = bbdata['dx']
    dy = bbdata['dy']
    dz = bbdata['dz']

    dmax = max(dx,dy,dz)

    delta = dmax/ndim
    delta *= 1 + buffer # expand the block width by our buffer

    O_delta = Magnitude(delta)
    O_calc = O_delta -2 # this is the rounding accuracy for bockMesh creation, 2 dimensions finer than the block size.

    # this is a glorified rounding function, but with ceil.
    # computes floating point accuracy ceil of x with dimension n
    MagnitudeCeil = lambda x, n : int(np.ceil(x/10**(n)))*10**(n)

    # here we round our random float value to 2 dimensions smaller than the dimension of itself.
    # e.g., delta = 0.034651 => O_delta = -2 => O_delta - 2 = -4 => delta_calc = 0.0347
    delta_calc = MagnitudeCeil(delta, O_calc)

    bm_xmax = MagnitudeCeil(bbdata['xmax'] + dx*buffer/2, O_calc)
    bm_ymax = MagnitudeCeil(bbdata['ymax'] + dy*buffer/2, O_calc)
    bm_zmax = MagnitudeCeil(bbdata['zmax'] + dz*buffer/2, O_calc)

    div_x = int(np.ceil(dx*(1+buffer) / delta_calc))
    div_y = int(np.ceil(dy*(1+buffer) / delta_calc))
    div_z = int(np.ceil(dz*(1+buffer) / delta_calc))

    bm_xmin = bm_xmax - div_x * delta_calc
    bm_ymin = bm_ymax - div_y * delta_calc
    bm_zmin = bm_zmax - div_z * delta_calc

    with open(blockmeshinput, 'r') as f:
        data = f.read()

        # calling another round operation here to get rid of floating point errors (0.12340000000000001 -> 0.1234)
        # also round is OK here because the value has already bein ceiled towards its target...
        data = data.replace('ofkey_xmin', str(round(bm_xmax, -O_calc)))
        data = data.replace('ofkey_ymin', str(round(bm_ymax, -O_calc)))
        data = data.replace('ofkey_zmin', str(round(bm_zmax, -O_calc)))

        data = data.replace('ofkey_xmax', str(round(bm_xmin, -O_calc)))
        data = data.replace('ofkey_ymax', str(round(bm_ymin, -O_calc)))
        data = data.replace('ofkey_zmax', str(round(bm_zmin, -O_calc)))

        data = data.replace('ofkey_blocksx', str(div_x))
        data = data.replace('ofkey_blocksy', str(div_y))
        data = data.replace('ofkey_blocksz', str(div_z))

    with open(outputfile, 'w') as f:
        f.write(data)





def Magnitude(x):
    '''Returns the magnitude of a value <x>'''
    if x == 0:
        return None
    return int(np.floor(np.log10(np.abs(x))))



    


def GetBoundingBox(obj):
    '''
    Returns bounding box vertices, and min, max and mean x, y & z values of a given <obj>.
    '''

    bboxVerts = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]

    xx = [a[0] for a in bboxVerts]
    yy = [a[1] for a in bboxVerts]
    zz = [a[2] for a in bboxVerts]

    bbdata = {}

    bbdata['xmin'] = min(xx)
    bbdata['xmax'] = max(xx)
    bbdata['ymin'] = min(yy)
    bbdata['ymax'] = max(yy)
    bbdata['zmin'] = min(zz)
    bbdata['zmax'] = max(zz)

    bbdata['dx'] = max(xx) - min(xx)
    bbdata['dy'] = max(yy) - min(yy)
    bbdata['dz'] = max(zz) - min(zz)
    
    bbdata['xmean'] = (min(xx) + max(xx)) / 2
    bbdata['ymean'] = (min(yy) + max(yy)) / 2
    bbdata['zmean'] = (min(zz) + max(zz)) / 2

    return bbdata








# ██ ███    ███ ██████   ██████  ██████  ████████ 
# ██ ████  ████ ██   ██ ██    ██ ██   ██    ██    
# ██ ██ ████ ██ ██████  ██    ██ ██████     ██    
# ██ ██  ██  ██ ██      ██    ██ ██   ██    ██    
# ██ ██      ██ ██       ██████  ██   ██    ██    
                                                
# FILE LOADING: Import all STL files 


def ImportSTLFiles(directory):
    '''Imports all stl files which are immediate children in directory <directory> into the scene'''
    if not os.path.exists(directory):
        raise Exception('Directory could not be found, check in json and try again! (%s)'%(directory))
    stlfiles = [file for file in os.listdir(directory) if file.lower().endswith('.stl')]
    if not stlfiles:
        raise Exception('No STL Files found in directory \'%s\', check directory and try again!'%(directory))

    for stlfile in stlfiles:
        stlfilepath = os.path.join(directory, stlfile)
        bpy.ops.wm.stl_import(filepath=stlfilepath) 




# ███████ ██   ██ ██████   ██████  ██████  ████████ 
# ██       ██ ██  ██   ██ ██    ██ ██   ██    ██    
# █████     ███   ██████  ██    ██ ██████     ██    
# ██       ██ ██  ██      ██    ██ ██   ██    ██    
# ███████ ██   ██ ██       ██████  ██   ██    ██    
                                                  
                                                  
# FILE EXPORT: Handled by SnappyhexmeshGUI Blender addon

def ExportSnappyhexmeshGUI(exportpath, 
                           obj, 
                           clear_directory=True,
                           no_cpus=1, 
                           cell_length=0.1,
                           surface_refinement_min=0,
                           surface_refinement_max=0,
                           feature_edge_level=0,
                           cleanup_distance=1e-5):
    '''
    Handles export of <obj> to <exportpath> via the SnappyhexmeshGUI addon. 
    '''

    # This Exception handle should be obsolete as the check is done on script startup:
    # try: # check if plugin is installed
    #     bpy.context.scene.snappyhexmeshgui.bl_rna
    # except:
    #     raise Exception("STL2OpenFOAM Error: SnappyHexMeshGUI Blender addon is not installed!")
    
    InfoMsg('Exporting mesh to SnappyHexMesh...', newline=True)
    SetActive(obj) 
    
    if clear_directory: bpy.ops.object.snappyhexmeshgui_clean_case_dir()

    bpy.ops.object.snappyhexmeshgui_apply_locrotscale()
    bpy.context.scene.snappyhexmeshgui.export_path = exportpath
    bpy.context.scene.snappyhexmeshgui.number_of_cpus = no_cpus
    bpy.context.scene.snappyhexmeshgui.cell_side_length = cell_length

    SetActive(obj)
    bpy.context.object.shmg_surface_min_level = surface_refinement_min
    bpy.context.object.shmg_surface_max_level = surface_refinement_max
    bpy.context.object.shmg_feature_edge_level = feature_edge_level
    # TODO for some reason, the following property has to be passed as a string. 
    # There also is the property snappyhexmeshgui.merge_distance, but it gets ignored or
    # overwritten by merge_distance_string. Potentially buggy, investigate in future. 
    bpy.context.scene.snappyhexmeshgui.merge_distance_string = str(cleanup_distance) 
    bpy.ops.object.snappyhexmeshgui_cleanup_meshes()

    bpy.ops.object.snappyhexmeshgui_add_location_in_mesh_object()
    SetActive(obj) # obj needs to be set active after adding empty in scene for loc in mesh
    bpy.ops.object.snappyhexmeshgui_export()

    InfoMsg('Export succesful.')



# Note: This function has become obsolete due to the 
# built in function bpy.ops.object.snappyhexmeshgui_clean_case_dir
def ClearExportDirectory(exportpath):
    '''A function to handle cleaning of the export directory. For now it deletes only .stl files'''
    n = 0
    for root, dir, files in os.walk(exportpath):
        for file in files:
            if file.lower().endswith('.stl'):
                n+=1
                os.remove(os.path.join(root,file))
    InfoMsg('Deleted %s .stl files from export directory.'%(n))







# ███    ███  █████  ██ ███    ██ 
# ████  ████ ██   ██ ██ ████   ██ 
# ██ ████ ██ ███████ ██ ██ ██  ██ 
# ██  ██  ██ ██   ██ ██ ██  ██ ██ 
# ██      ██ ██   ██ ██ ██   ████ 

if __name__ == "__main__":

    SetupArgparser()

    PrintNameWithVersion()

    # CheckAndEnableAddonDependencies()
    
    DeleteAllBlenderData()


    configpath = './config_stl.json'
    configpath = 'D:/DATA2/GIT/IANUS/step2openfoam/config_stl.json'
    config = LoadConfig(configpath)

    # Import .stl files
    filepath = config['stl_import_filepath'] 
    filepath = 'D:/DATA2/GIT/IANUS/step2openfoam/stl_geometries'
    
    
    ImportSTLFiles(filepath)

    obj = JoinAllMeshesAndAssignMaterialSlots()
    CleanMesh(obj, delete_non_manifold=True)
    CheckMesh(obj)

    # BlenderApplyRotScale() # apply the custom scale to objs

    # Find point inside mesh
    deterministic = config['pointInMesh_deterministic']
    seed = config['pointInMesh_seed']
    if deterministic:
        np.random.seed(seed)
        random.seed(seed)



    # obj = GetFirstMeshInScene()

    # Find the point inside the mesh
    delta = config['pointInMesh_delta']
    maxiter = config['pointInMesh_maxiter']
    no_rays = config['pointInMesh_rays_primary']
    no_rays_secondary = config['pointInMesh_rays_secondary']
    optpoint = SearchForPointWithThreshold(
        obj, 
        delta=delta, 
        maxiter=maxiter, 
        no_rays=no_rays, 
        no_rays_secondary=no_rays_secondary
        )

    # The "location in mesh" for snappyhexmesh is set to where the 3D cursor is
    Set3DCursorToLocation(obj, optpoint)    


    WriteBlockMesh(obj,65)

    SeparateGeometryByMaterialGroups(obj)

    # # Export via snappyhexmeshgui
    # exportpath = config['snappyhex_export_filepath'] 
    # no_cpus = config['snappyhex_no_cpus']
    # surface_refinement_max = config['snappyhex_surface_refinement_max']
    # surface_refinement_min = config['snappyhex_surface_refinement_min']
    # cell_length = config['snappyhex_cell_length']
    # feature_edge_level = config['snappyhex_feature_edge_level']
    # cleanup_distance = config['snappyhex_cleanup_distance']
    # ExportSnappyhexmeshGUI(exportpath, 
    #                        obj, 
    #                        clear_directory=True,
    #                        no_cpus=no_cpus,
    #                        cell_length=cell_length,
    #                        surface_refinement_max=surface_refinement_max,
    #                        surface_refinement_min=surface_refinement_min,
    #                        feature_edge_level=feature_edge_level,
    #                        cleanup_distance=cleanup_distance,
    #                        )

    # InfoMsg("Ending execution.", newline=True)

