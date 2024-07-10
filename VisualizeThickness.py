import bpy
import mathutils
import matplotlib
from numpy import pi 

# cmap = matplotlib.cm.get_cmap('jet')

cmap = matplotlib.colormaps.get_cmap('jet')


def Set3DCursorToLocation(pos):
    bpy.context.scene.cursor.location = pos

def SetActive(obj):
    '''Sets the current object as active and selected in Blender.'''
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)


def GetDepth(obj, poly, delta = 1e-6):
    center = poly.center
    normal = poly.normal

    ray_dir = -normal
    ray_origin = center + ray_dir * delta 
    result, hit_loc, _, _ = obj.ray_cast(ray_origin, ray_dir)
    
    dist = 0

    if result:
        # print("hit loc " + str(hit_loc) + " center " + str(center))
        dist = (-hit_loc+center).length
        # print(dist)
    else:
        Set3DCursorToLocation(center)
        return
    
    if dist <= 0:
        Set3DCursorToLocation(center)
        return
    

    return dist



def GetColorFromValue(val, invert_colorbar = False):
    if val < 0:
        rgba = (0,0,0,1)
    elif val > 1:
        rgba = (1,1,1,1)
    else:  
        rgba = cmap(1 - val if invert_colorbar else val)
    grayscale = (val, val, val, 1.0) 
    return rgba


'''
dists = [ 0.4 .... 8 ]

scale_colors = -1:      dists_scaled = [ 0.05 .... 1 ]

dmin = 0.4, dmax = 8:   dists_scaled = [ 0 .... 1 ] TODO: add CLAMPING!

        1            _________
                    /
                   /
                  /
        0   ...../

            0   min  max    x
            
            f(x) = (x-min) / (max-min)
'''

def SetFaceColorToDepth(obj, 
                        scale_colors=-1, 
                        dmin=-1, dmax=-1, 
                        delta=1e-5,
                        invert_colorbar = False,  
                        absolute = False):
    mesh = obj.data

    if not mesh.vertex_colors:
        mesh.vertex_colors.new()
    col_layer = mesh.vertex_colors.active

    dists = [0 for i in range(len(mesh.polygons))]
    for i, poly in enumerate(mesh.polygons):
        dists[i] = GetDepth(obj, poly, delta)

    distmax = max(dists)

    if distmax <= 0:
        raise Exception("Maximum distance equals to zero, check face normals!")

    # if scale_colors <= 0:
    #     scale_colors = 1/distmax

    # dist_scaled = [x * scale_colors for x in dists]

    if dmin > 0 and dmax > 0 and dmax > dmin:
        dist_scaled = [(x - dmin)/(dmax - dmin) for x in dists]

    # dist_scaled = dists
    # for x in dist_scaled:
    #     if x<0:
    #         raise Exception('???')
    

    # print("maxdist = " + str(distmax))

    # print("scaled dists " + str(dist_scaled[0]))

    for i, poly in enumerate(mesh.polygons):
        dist = dist_scaled[i]
        col = GetColorFromValue(dist_scaled[i], invert_colorbar)
        # print(str(i) + ", dist=" + str(dist_scaled[i]))
        for idx in poly.loop_indices:
            col_layer.data[idx].color = col



def SingleCastDebug(obj, j=0, dmin=-1, dmax=-1, delta=1e-6, absolute = False):

    mesh = obj.data

    poly = mesh.polygons[j]

    if not mesh.vertex_colors:
        mesh.vertex_colors.new()
    col_layer = mesh.vertex_colors.active

    dist = GetDepth(obj, poly, delta)

    Set3DCursorToLocation(poly.center)

    dist = (dist - dmin) / (dmax - dmin)
    col = GetColorFromValue(dist,)

    # print(col)
    for idx in poly.loop_indices:
        col_layer.data[idx].color = col






def CreateCamera(obj, rot, orthographic, camera_safezone): 

    rendercam = bpy.context.scene.render

    if not bpy.data.cameras:
        cam_data = bpy.data.cameras.new('Camera') 
    else:
        cam_data = bpy.data.cameras[0]
    
    if not 'Camera' in bpy.data.objects:
        cam_obj = bpy.data.objects.new('Camera', cam_data) 
        bpy.context.scene.collection.objects.link(cam_obj) 
    
    else:
        cam_obj = bpy.context.scene.objects['Camera']

    
    bpy.context.scene.camera = cam_obj 
    
    SetActive(cam_obj)

    res_x = rendercam.resolution_x 
    res_y = rendercam.resolution_y 

    bpy.context.object.data.type = ('ORTHO' if orthographic else 'PERSP')
    cam_obj.rotation_euler = mathutils.Euler(rot, 'XYZ')
    
    
    SetActive(obj)
    if rendercam.resolution_y < rendercam.resolution_x:
        rendercam.resolution_y = res_y - camera_safezone 
    else: 
        rendercam.resolution_x = res_x - camera_safezone
    # print(rendercam.resolution_y)
    bpy.ops.view3d.camera_to_view_selected()
    rendercam.resolution_y = res_y 
    rendercam.resolution_x = res_x 
    # print(rendercam.resolution_y)






if __name__ == "__main__":
    obj = bpy.context.scene.objects["testgeom"]

    dmin = 0.1
    dmax = 0.8
    SetFaceColorToDepth(obj, 1, dmin=dmin, dmax=dmax, delta = 1e-2 )

    # SingleCastDebug(obj, j=50, dmin=0.1, dmax=2)
    rot = mathutils.Vector((pi/4, 0, pi/4))
    CreateCamera(obj, rot, orthographic=True, camera_safezone=60)
    bpy.ops.object.mode_set(mode='VERTEX_PAINT')