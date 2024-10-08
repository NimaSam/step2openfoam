# -*- coding: utf-8 -*-
# author: Lukas Kilian
# Step 2 OpenFOAM Pipeline
# Date of creation: 09.07.2024


bl_info = {
    "name" : "Step2OpenFOAM",
    "description" : "A pipeline for creating OpenFOAM data from .step files",
    "author" : "Lukas Kilian",
    "version" : (0, 1, 2),
    "blender" : (4, 0, 0),
    # "location" : "View3D",
    # "warning" : "",
    "support" : "COMMUNITY",
    # "doc_url" : "",
    # "category" : "3D View"
}

progressbar_prefix_len = 30 # the length of the prefix string before printing the progress bar

dependencies = ['snappyhexmesh_gui-master', 'STEPper']

import os
import bpy
import json
import random
import argparse
import mathutils
import addon_utils
import numpy as np




# ███    ███ ██ ███████  ██████ 
# ████  ████ ██ ██      ██      
# ██ ████ ██ ██ ███████ ██      
# ██  ██  ██ ██      ██ ██      
# ██      ██ ██ ███████  ██████ 
                            
# Utility and helper scripts

def InfoMsg(msg, newline=False):
    '''Utility script to print unified info messages in the console.'''
    if newline: print('')
    print('## Step2OpenFoam Info: ' +msg)
    # print('')


def PrintNameWithVersion():
    '''Prints a fancy Ascii text with version number into the console'''
    version_string = 'v' + str(bl_info['version'][0]) + \
                     '.' + str(bl_info['version'][1]) + \
                     '.' + str(bl_info['version'][2])
    n = len(version_string)
    nmax = 32
    print(r'''   
  ___ _            ___ ___                 ___ ___   _   __  __ 
 / __| |_ ___ _ __|_  ) _ \ _ __  ___ _ _ | __/ _ \ /_\ |  \/  |
 \__ \  _/ -_) '_ \/ / (_) | '_ \/ -_) ' \| _| (_) / _ \| |\/| |
 |___/\__\___| .__/___\___/| .__/\___|_||_|_| \___/_/ \_\_|  |_|
             |_|           |_|  ''' 
          + (nmax-n)*' ' + version_string + '\n')



def LoadConfig():
    '''Loads the config file and returns the data.'''
    configpath = './config.json'
    if not os.path.exists(configpath):
        raise Exception('Step2OpenFOAM Error: Config could not be found! Check configpath in main function!')

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



def SetupArgparser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
    parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

    args = parser.parse_args()

    print(args.accumulate(args.integers))




# ██████  ██      ███████ ███    ██ ██████  ███████ ██████  
# ██   ██ ██      ██      ████   ██ ██   ██ ██      ██   ██ 
# ██████  ██      █████   ██ ██  ██ ██   ██ █████   ██████  
# ██   ██ ██      ██      ██  ██ ██ ██   ██ ██      ██   ██ 
# ██████  ███████ ███████ ██   ████ ██████  ███████ ██   ██ 
                                                          
# All purely blender specific scripts

def GetBlenderAddons():
    '''
    Returns the list of enabled addons..
    '''
    paths_list = addon_utils.paths()
    addon_list = []
    for path in paths_list:
        for mod_name, mod_path in bpy.path.module_names(path):
            is_enabled, is_loaded = addon_utils.check(mod_name)
            addon_list.append(mod_name)
    return addon_list



def CheckAndEnableAddonDependencies():
    '''
    Checks whether addon dependencies are fulfilled. 
    If an addon is installed but not enabled, enable it.
    '''
    available_addons = GetBlenderAddons()
    for dependency in dependencies:
        if dependency in available_addons:
            is_enabled, is_loaded = addon_utils.check(dependency)
            if not is_enabled:
                addon_utils.enable(dependency)
        else:
            raise Exception('Step2OpenFOAM Error: Missing Addon : %s!'%(dependency))
        
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
        print('Step2OpenFoam Warning: No mesh found in Scene!')
        return None
    elif len(meshes) > 1:
        print('Step2OpenFoam Warning: More than 1 object found after import, returning first object!')
    
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

    raise Exception('Step2OpenFOAM Error: No points found within %s iterations. ' \
                    'Try increasing maximum interatios or decreasing delta threshold.'%(maxiter))



def FindInsidePoint(obj, face_index = 0, delta = 1e-6, deltamax= 1e-4):
    '''Finds a point inside the mesh by casting a ray from the <face_index>-th face of the Object <obj>,
    and returning the midway point between ray cast origin and ray cast hit.'''

    mesh = obj.data

    # check whether specified face_index is out of bounds for given obj
    try: face = mesh.polygons[face_index]
    except: raise Exception('Step2OpenFOAM Error: Face Index out of bounds for given object. ' \
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















# ██ ███    ███ ██████   ██████  ██████  ████████ 
# ██ ████  ████ ██   ██ ██    ██ ██   ██    ██    
# ██ ██ ████ ██ ██████  ██    ██ ██████     ██    
# ██ ██  ██  ██ ██      ██    ██ ██   ██    ██    
# ██ ██      ██ ██       ██████  ██   ██    ██    
                                                
# FILE LOADING: Handled by the STEPper Blender addon

def CheckFilepath(filepath, file):
    '''
    Checks whether the filepath and to-be-imported file exists. 
    Also handles misspelling of the extension .step and .stp.
    '''

    path = os.path.join(filepath, file)
    if os.path.exists(path):
        return filepath, file
    else: #check if just the file ending is misspelled
        if file.lower().endswith('.step'):
            file = file[:-5] + '.stp'
        elif file.lower().endswith('.stp'):
            file = file[:-4] + '.step'
        path = os.path.join(filepath, file)
        if os.path.exists(path):
            return filepath, file
        
    raise Exception('Step2OpenFOAM Error: The specified file could not be found!')



def ImportSTEP(filepath, file, detail_level = 100):
    '''Loads a STEP file into Blender using the STEPper addon'''
    Reset3DCursor()
    
    filepath, file = CheckFilepath(filepath, file)
    # if not CheckFilepath(filepath, file):

    InfoMsg('Importing STEP file %s, detail level = %s...'%(file,detail_level), True)

    bpy.ops.import_scene.occ_import_step(filepath=filepath, 
                                            override_file=file, 
                                            detail_level = detail_level,
                                            hierarchy_types = 'EMPTIES')
    obj = GetFirstMeshInScene()
    verts = len(obj.data.vertices)
    InfoMsg('Successfully imported one mesh with %s vertices.'%(verts))













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
    #     raise Exception("Step2OpenFOAM Error: SnappyHexMeshGUI Blender addon is not installed!")
    
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

    PrintNameWithVersion()

    CheckAndEnableAddonDependencies()
    
    DeleteAllBlenderData()

    # SetupArgparser()

    
    config = LoadConfig()

    # Import .step file via STEPper
    filepath = config['stepper_import_filepath'] 
    file = config['stepper_import_file'] 
    detail_level = config['stepper_import_detail_level'] 
    ImportSTEP(filepath, file, detail_level=detail_level)
    BlenderApplyRotScale() # apply the custom scale given during the STEPper import

    # Find point inside mesh
    deterministic = config['pointInMesh_deterministic']
    seed = config['pointInMesh_seed']
    if deterministic:
        np.random.seed(seed)
        random.seed(seed)

    # Get the first Mesh object in the scene. There should only be one mesh after the STEPper import
    obj = GetFirstMeshInScene()

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

    # Export via snappyhexmeshgui
    exportpath = config['snappyhex_export_filepath'] 
    no_cpus = config['snappyhex_no_cpus']
    surface_refinement_max = config['snappyhex_surface_refinement_max']
    surface_refinement_min = config['snappyhex_surface_refinement_min']
    cell_length = config['snappyhex_cell_length']
    feature_edge_level = config['snappyhex_feature_edge_level']
    cleanup_distance = config['snappyhex_cleanup_distance']
    ExportSnappyhexmeshGUI(exportpath, 
                           obj, 
                           clear_directory=True,
                           no_cpus=no_cpus,
                           cell_length=cell_length,
                           surface_refinement_max=surface_refinement_max,
                           surface_refinement_min=surface_refinement_min,
                           feature_edge_level=feature_edge_level,
                           cleanup_distance=cleanup_distance,
                           )

    InfoMsg("Ending execution.", newline=True)