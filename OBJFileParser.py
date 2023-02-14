import numpy as np
import math
from BasicFunctions import *
    

def file_read(file):
    vertices = []
    normais = []
    faces = []
    with open(file, "r") as opened_file:
        for line in opened_file:
            if line[0] == "v" and line[1] == " ":
                v = line.split(" ")
                vertices.append([float(v[1]), float(v[2]), float(v[3].rstrip("\n"))])
            if line[0] == "v" and line[1] == "n":
                vn = line.split(" ")
                normais.append([float(vn[1]), float(vn[2]), float(vn[3].rstrip("\n"))])
            if line[0] == "f" and line[1] == " ":
                f = line.split(" ")
                faces.append([f[1], f[2], f[3].rstrip("\n")])
    return (vertices, normais, faces)


def face_to_line(results):
    list = []
    for face in results[2]:
        info = []
        normal = 0
        for coord in face:
            index = coord.split("/")
            vert = results[0][int(index[0])-1]
            info.append(vert)
            if results[0][int(index[2])-1] != normal:
                normal = results[1][int(index[2])-1]
        info.append(normal)
        list.append(info)
    return list

def parse(file):
    resultados = file_read(file)
    final = face_to_line(resultados)
    return final
