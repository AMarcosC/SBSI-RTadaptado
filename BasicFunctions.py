"""
Este arquivo condensa funções e classes básicas para o funcionamento do programa, sendo funções que não
se relacionam diretamente com o problema do ray-tracing
"""

import os
import math
import numpy as np
import sys
import pickle
from PIL import Image
from colour import Color
from PIL import ImageDraw
from PIL import ImageFont

"""Classes"""

class vec3():  #classe que define o funcionamento de um vetor e suas operações
    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)
    def __mul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)
    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
    def __abs__(self):
        return self.dot(self)
    def norm(self):
        mag = np.sqrt(abs(self))
        return self * (1.0 / np.where(mag == 0, 1, mag))
    def components(self):
        return (self.x, self.y, self.z)
    def extract(self, cond):
        return vec3(extract(cond, self.x),
                    extract(cond, self.y),
                    extract(cond, self.z))
    def place(self, cond):
        r = vec3(np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r

class Sphere:  #classse que define as propriedades de uma esfera
    def __init__(self, center, r, color):
        self.c = center
        self.r = r
        self.color = color

class Objeto:  #classse que define as propriedades de um objeto genérico qualquer
    def __init__(self, file, color):
        self.f = file
        self.color = color

class Triangle:  #classe que define as propriedades de um triângulo
    def __init__(self, v1, v2, v3, color, normal = vec3(0,0,1)):
        self.v = [v1, v2, v3]
        self.color = color
        self.normal = normal

class Point:  #classe que define um ponto (não utilizada pq a vec3 é mais completa)
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


"""Funções Básicas"""

def vetor_escalar(vetor,escalar): #multiplicação de um vetor por um escalar
    comp_x = vetor.x * escalar
    comp_y = vetor.y * escalar
    comp_z = vetor.z * escalar
    return(vec3(comp_x,comp_y,comp_z))  #retorna o valor como vec3 (classe)


def vetor_escalar_noclass(vetor,escalar): #multiplicação de um vetor por um escalar, sem usar a classe vec3
    comp_x = vetor[0] * escalar
    comp_y = vetor[1] * escalar
    comp_z = vetor[2] * escalar
    return([comp_x,comp_y,comp_z])  #retorna o valor como vec3 (classe)


def menor(x1,x2):  #retorna o menor valor entre dois valores
    if x1 <= x2:
        return x1
    else:
        return x2

def maior(x1,x2):  #retorna o menor valor entre dois valores
    if x1 <= x2:
        return x2
    else:
        return x1

def menor_absoluto(x1,x2):  #retorna o menor valor absoluto entre dois valores (com sinal)
    if abs(x1) <= abs(x2):
        return x1
    else:
        return x2

def mais_proximo(x1,x2,dir):  #retorna a interseção mais próxima em um objeto (dir em vec3)
    if dir.z > 0:
        if x1 >= x2:
            return x1
        else:
            return x2
    else:
        if x1 >= x2:
            return x2
        else:
            return x1


def azimuth(angle):  #converte o ângulo no sistema trigonométrico para o de azimute
    if angle >= 0 and angle < 90:
        return (90 - angle)
    elif angle >= 90 and angle < 180:   #ajeitar depois
        return (360 - (angle - 90))
    elif angle >= 180 and angle < 270:  #ajeitar depois
        return (270 - angle + 180)
    elif angle >= 270 and angle < 360:
        return (360 - angle + 90)

def polar_to_vector(el,az):  #converte as coordenadas polares do sol em um vetor unitário (não utilizada, corrigida na função seguinte)
    x = math.cos(math.radians(el))*math.cos(math.radians(az))
    y = math.cos(math.radians(el))*math.sin(math.radians(az))
    z = math.sin(math.radians(el))
    return vec3(x, y, z)

def polar_to_vector_ajustado(el,az,offset):  #converte as coordenadas polares do sol em um vetor unitário, considerando azimute com 0° no norte e sentido horário
    az_ajustado = azimuth(az) - offset
    x = math.cos(math.radians(el))*math.cos(math.radians(az_ajustado))
    y = math.cos(math.radians(el))*math.sin(math.radians(az_ajustado))
    z = math.sin(math.radians(el))
    return vec3(x, y, z)

def polar_to_vector_ajustado_array(el,az,offset):  #converte as coordenadas polares do sol em um vetor unitário, considerando azimute com 0° no norte e sentido horário
    az_ajustado = azimuth(az) - offset
    x = math.cos(math.radians(el))*math.cos(math.radians(az_ajustado))
    y = math.cos(math.radians(el))*math.sin(math.radians(az_ajustado))
    z = math.sin(math.radians(el))
    return [x, y, z]


def change_to_current_dir():  #muda a pasta de trabalho atual (apenas para debug)
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

def vec3_array_to_python_array(array):
    new_array=np.full_like(array, None)
    for i in range (0, len(array), 1):
        for j in range (0, len(array[0]), 1):
            if array[i][j] != None:
                new_array[i][j] == [array[i][j].x, array[i][j].y, array[i][j].z]
    return new_array

def python_array_to_pickle(array, filename):
    file = open(filename, 'wb')
    pickle.dump(array, file)
    file.close()

def random_color():
    color = list(np.random.choice(range(256), size=3))
    return ([color[0], color[1], color[2], 255])

def random_bright_color():
    color = list(np.random.choice(range(80,256), size=3))
    return ([color[0], color[1], color[2], 255])

def overlay_images(front_image, back_image, final_image):
    img1 = Image.open(front_image)
    img1 = img1.convert("RGBA")
    img2 = Image.open(back_image)
    img2 = img2.convert("RGBA")
    new_img = Image.blend(img2, img1, 0.5)
    new_img.save(final_image,"PNG")

def impar(x):
    if x%2 != 0:
        return True
    else:
        return False

def highest_value_in_array(array):
    temp_highest = 0
    for line in array:
        for number in line:
            if number > temp_highest:
                temp_highest = number
    return temp_highest

def ordered_values_until(number):
    lista = []
    for k in range (0,number,1):
        lista.append(k)
    return lista

def all_combinations(lines, columns):
    comb = []
    for i in range (0, lines, 1):
        for j in range(0, columns, 1):
            comb.append([i,j])
    return comb

def all_combinations_range(i0, i1, j0, j1):
    comb = []
    for i in range(i0, i1, 1):
        for j in range(j0, j1, 1):
            comb.append([i, j])
    return comb

def all_combinations_placa(i0, i1, j0, j1):
    comb = []
    for i in range(i0, i1, -1):
        for j in range(j0, j1, 1):
            comb.append([i, j])
    return comb

def results_to_list(results):
    output = []
    for res in results:
        print("Tem resultado!")
        output.append(res)
    return output


def list_to_array_reshape(lista, x, y):
    array = []
    counter = 0
    line = []
    for i in lista:
        if counter < x-1:
            line.append(i)
            counter += 1
        else:
            line.append(i)
            array.append(line)
            counter = 0
            line = []
    return array

def distribute_for_three(a):
    if a % 3 == 0:
        return [a//3, a//3, a//3]
    elif a % 3 == 1:
        return [a//3, a//3, (a//3)+1]
    elif a % 3 == 2:
        return [a//3, (a//3)+1, (a//3)+1]

def distribute_for_two(a):
    if a % 2 == 0:
        return [a//2, a//2]
    elif a % 2 == 1:
        return [(a//2)+1, a//2]

def color_range_image(list_colors):
    spacing = 50
    width = 3*spacing
    height = len(list_colors)*spacing
    image_m = np.full((height, width, 4), [255,255,255,255])
    cont = 0
    cont_n = 0
    for c in list_colors:
        for i in range(cont*spacing, (cont+1)*spacing, 1):
            for j in range(0, spacing, 1):
                image_m[i][j] = [c.red*255, c.green*255, c.blue*255, 255]
        cont = cont + 1
    img_color = Image.fromarray(np.uint8(image_m)).convert('RGBA')  #Transformando a matriz em uma imagem .png
    img0 = ImageDraw.Draw(img_color)
    myFont = ImageFont.truetype('utilities/Roboto-Black.ttf', 30)
    for c_n in list_colors:
        img0.text([60, (50*cont_n) + 10], "{}".format(cont_n), font=myFont, fill=(0,0,0))
        cont_n = cont_n + 1
    img_color.save('output/ColorDict.png')

def triangle_normal(p1, p2, p3):
    A = p2 - p1
    B = p3 - p1
    Nx = (A.y * B.z) - (A.z * B.y)
    Ny = (A.z * B.x) - (A.x * B.z)
    Nz = (A.x * B.y) - (A.y * B.x)
    Nmod = ((Nx**2) + (Ny**2) + (Nz**2))**(1/2)
    Nxx = Nx / Nmod
    Nyy = Ny / Nmod
    Nzz = Nz / Nmod
    N = [Nxx, Nyy, Nzz]
    return N

def panel_to_list(p_list):
    v = 1
    n = 1
    v_list = []
    n_list = []
    f_list = []
    for panel in p_list:
        p0 = panel.coord[0]
        p1 = panel.coord[1]
        p2 = panel.coord[2]
        p3 = panel.coord[3]
        v_list.append([p0.x,p0.y,p0.z])
        v_list.append([p1.x,p1.y,p1.z])
        v_list.append([p2.x,p2.y,p2.z])
        v_list.append([p3.x,p3.y,p3.z])
        n_list.append(triangle_normal(p0, p2, p3))
        f_list.append(r'{}/{} {}/{} {}/{}'.format(v,n,v+2,n,v+3,n))
        f_list.append(r'{}/{} {}/{} {}/{}'.format(v,n,v+3,n,v+1,n))
        v = v + 4
        n = n + 1
    return (v_list, n_list, f_list)


def panel_to_list_new(p_list,esp):
    v = 1
    n = 1
    v_list = []
    n_list = []
    f_list = []
    o_list = []
    for panel in p_list:
        o_temp = []
        p0 = panel.coord[0]
        p1 = panel.coord[1]
        p2 = panel.coord[2]
        p3 = panel.coord[3]
        p4 = vec3(p0.x, p0.y, p0.z + esp)
        p5 = vec3(p1.x, p1.y, p1.z + esp)
        p6 = vec3(p2.x, p2.y, p2.z + esp)
        p7 = vec3(p3.x, p3.y, p3.z + esp)
        #vértices
        v_list.append([p0.x,p0.y,p0.z])
        v_list.append([p1.x,p1.y,p1.z])
        v_list.append([p2.x,p2.y,p2.z])
        v_list.append([p3.x,p3.y,p3.z])
        v_list.append([p4.x,p4.y,p4.z])
        v_list.append([p5.x,p5.y,p5.z])
        v_list.append([p6.x,p6.y,p6.z])
        v_list.append([p7.x,p7.y,p7.z])
        #normais
        n_list.append(triangle_normal(p0, p1, p2))
        n_list.append(triangle_normal(p4, p0, p2))
        n_list.append(triangle_normal(p5, p3, p1))
        n_list.append(triangle_normal(p4, p5, p0))
        n_list.append(triangle_normal(p2, p7, p6))
        n_list.append(triangle_normal(p4, p7, p5))
        #faces
        f_list.append(r'{}/{} {}/{} {}/{}'.format(v,n,v+1,n,v+2,n))
        f_list.append(r'{}/{} {}/{} {}/{}'.format(v+1,n,v+3,n,v+2,n))
        f_list.append(r'{}/{} {}/{} {}/{}'.format(v+4,n+1,v+0,n+1,v+2,n+1))
        f_list.append(r'{}/{} {}/{} {}/{}'.format(v+4,n+1,v+2,n+1,v+6,n+1))
        f_list.append(r'{}/{} {}/{} {}/{}'.format(v+5,n+2,v+3,n+2,v+1,n+2))
        f_list.append(r'{}/{} {}/{} {}/{}'.format(v+5,n+2,v+7,n+2,v+3,n+2))
        f_list.append(r'{}/{} {}/{} {}/{}'.format(v+4,n+3,v+5,n+3,v+0,n+3))
        f_list.append(r'{}/{} {}/{} {}/{}'.format(v+5,n+3,v+1,n+3,v+0,n+3))
        f_list.append(r'{}/{} {}/{} {}/{}'.format(v+2,n+4,v+7,n+4,v+6,n+4))
        f_list.append(r'{}/{} {}/{} {}/{}'.format(v+2,n+4,v+3,n+4,v+7,n+4))
        f_list.append(r'{}/{} {}/{} {}/{}'.format(v+4,n+5,v+7,n+5,v+5,n+5))
        f_list.append(r'{}/{} {}/{} {}/{}'.format(v+4,n+5,v+6,n+5,v+7,n+5))
        #indices
        o_temp.append(v_list)
        o_temp.append(n_list)
        o_temp.append(f_list)
        o_list.append(o_temp)
        v_list = []
        n_list = []
        f_list = []
        v = v + 8
        n = n + 6
    return (o_list)


def list_to_obj_file(p_list):
    lists = panel_to_list(p_list)
    file = open("Panels-3D.obj","w+")
    file.write("#Modelo 3d das placas - Antonio Marcos Cruz da Paz\n")
    file.write("o Placas\n")
    for v in lists[0]:
        file.write("v {} {} {}\n".format(v[0],v[1],v[2]))
    for n in lists[1]:
        file.write("vn {:.20f} {:.20f} {:.20f}\n".format(n[0], n[1], n[2]))
    file.write("usemtl None\n")
    file.write("s 1\n")
    for f in lists[2]:
        file.write("f {}\n".format(f))
    file.close()

def list_to_obj_file_new(p_list, esp, name):
    lists = panel_to_list_new(p_list, esp)
    file = open(name,"w+")
    file.write("#Modelo 3d das placas - Antonio Marcos Cruz da Paz\n")
    pc = 1
    for o in lists:
        file.write("o Placa {}\n".format(pc))
        for v in o[0]:
            file.write("v {} {} {}\n".format(v[0],v[1],v[2]))
        for n in o[1]:
            file.write("vn {:.20f} {:.20f} {:.20f}\n".format(n[0], n[1], n[2]))
        file.write("usemtl None\n")
        file.write("s 1\n")
        for f in o[2]:
            file.write("f {}\n".format(f))
        pc += 1
    file.close()
