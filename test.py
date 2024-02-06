import math
Dist = 28.70908
Angle = 42.2736916 
y_ofset = 34.86103
x_ofset = 27.59055
llpython = [Dist,Angle,0,0,0,0,0,0]

def convert_to_mid_of_robot(llpython, x_offset, y_offset):
    distance = llpython[0]
    angle = llpython[1]
    angle_rad = math.radians(angle) if isinstance(angle, (int, float)) else angle
    mol = distance * math.tan(angle_rad)
    mol = abs(y_offset - mol)
    distance += x_offset
    angle_rad = math.atan2(mol, distance)
    llpython = [distance, math.degrees(angle_rad)] + [0,  0,  0,  0,  0,  0]
    return llpython
def convert_to_x_y_coordinates(llpython):
    distance = llpython[0]
    angle = llpython[1]
    angle_rad = math.radians(angle) if isinstance(angle, (int, float)) else angle
    x = distance * math.cos(angle_rad)
    y = distance * math.sin(angle_rad)
    llpython = [distance, angle] + [x, y] + [0,  0,  0,  0]
    return llpython

llpython = convert_to_mid_of_robot(llpython, x_ofset, y_ofset)
print (llpython)
llpython = convert_to_x_y_coordinates(llpython)
print (llpython)
