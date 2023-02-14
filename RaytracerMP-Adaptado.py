"""RaytracerMP - Adaptada"""
"""
Autor: Antônio Marcos Cruz da Paz - antonio.marcos@aluno.ufca.edu.br
"""

"""Bibliotecas"""

import os
import sys
from OBJFileParser import parse
from BasicFunctions import *
import math
import numpy as np
from PIL import Image
from numpy import linalg
from colour import Color
import pickle
import multiprocessing
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm
import yaml
import time as timer

"""Funções Voltadas ao problema"""


def ray_p(t,e,dire):  #fórmula que define o ponto no espaço atingido por um raio
    p_0 = e[0] + (dire[0]*t)
    p_1 = e[1] + (dire[1]*t)
    p_2 = e[2] + (dire[2]*t)
    return [p_0, p_1, p_2]


def pixel_pos(i,j):  #transforma um pixel na tela em um ponto no espaço
    u = l + ((r-l)*(i+0.5))/n_x  #0,5 para centralizar o ponto no pixel
    v = top + ((bot-top)*(j+0.5))/n_y  #0,5 para centralizar o ponto no pixel
    return([u, v])


def screen_size_forr(list_triangles):  #determina o tamanho da tela
    global l, r, top, bot, depth, n_x, n_y, pixel_por_metro, forramento
    x_menor = FARAWAY
    x_maior = - FARAWAY
    y_menor = FARAWAY
    y_maior = - FARAWAY
    z_maior = - FARAWAY
    for triangle in list_triangles:
        for vertex in triangle:
            if vertex[0] > x_maior:
                x_maior = vertex[0]
            if vertex[0] < x_menor:
                x_menor = vertex[0]
            if vertex[1] > y_maior:
                y_maior = vertex[1]
            if vertex[1] < y_menor:
                y_menor = vertex[1]
            if vertex[2] > z_maior:
                z_maior = vertex[2]
    l = math.floor(int((x_menor)-(forramento))) - 1
    r = math.ceil(int((x_maior)+(forramento))) + 1
    top = math.ceil(int((y_maior)+(forramento))) + 1
    bot = math.floor(int((y_menor)-(forramento))) - 1
    if z_maior <= 0:
        depth = 1
    else:
        depth = int(z_maior) + 1
    n_x = abs(r-l)*pixel_por_metro
    n_y = abs(top-bot)*pixel_por_metro



def intercept_tri(tri, e, d, cl):  #função que define a interceptação de um triângulo por um raio
    v1 = tri[0]  #vértice de um triângulo
    v2 = tri[1]  #vértice de um triângulo
    v3 = tri[2]  #vértice de um triângulo
    a = [
    [v1[0] - v2[0], v1[0] - v3[0], d[0]],
    [v1[1] - v2[1], v1[1] - v3[1], d[1]],
    [v1[2] - v2[2], v1[2] - v3[2], d[2]],
    ]
    b = [(v1[0] - e[0]), (v1[1] - e[1]), (v1[2] - e[2])]
    if linalg.det(a) != 0:
        x = linalg.solve(a, b)
        if x[0] > 0 and x[1] > 0 and (x[0]+x[1] < 1) and x[2] > 0 and x[2] < FARAWAY:
            return (cl, x[2])
        else:
            return ([0,0,0,0], FARAWAY)
    else:
        return ([0,0,0,0], FARAWAY)


def intercept_tri_bool(tri, e, d): #função que define a interceptação de um triângulo por um raio (retorna um booleano)
    v1 = tri[0]  #vértice de um triângulo
    v2 = tri[1]  #vértice de um triângulo
    v3 = tri[2]  #vértice de um triângulo
    a = [
    [v1[0] - v2[0], v1[0] - v3[0], d[0]],
    [v1[1] - v2[1], v1[1] - v3[1], d[1]],
    [v1[2] - v2[2], v1[2] - v3[2], d[2]],
    ]
    b = [(v1[0] - e[0]), (v1[1] - e[1]), (v1[2] - e[2])]
    if linalg.det(a) != 0:
        x = linalg.solve(a, b)
        if x[0] > 0 and x[1] > 0 and (x[0]+x[1] < 1) and x[2] > 0 and x[2] < FARAWAY:
            return True
        else:
            return False
    else:
        return False



def intercept_tri_cached_a(tri, e, cl, a, adet):  #função que define a interceptação de um triângulo por um raio
    v1 = tri[0]  #vértice de um triângulo
    v2 = tri[1]  #vértice de um triângulo
    v3 = tri[2]  #vértice de um triângulo
    if adet != 0:
        b = [(v1[0] - e[0]), (v1[1] - e[1]), (v1[2] - e[2])]
        x = linalg.solve(a, b)
        if x[0] > 0 and x[1] > 0 and (x[0]+x[1] < 1) and x[2] > 0 and x[2] < FARAWAY:
            return (cl, x[2])
        else:
            return ([0,0,0,0], FARAWAY)
    else:
        return ([0,0,0,0], FARAWAY)


def intercept_tri_bool_cached_a(tri, e, a, adet):  #função que define a interceptação de um triângulo por um raio
    v1 = tri[0]  #vértice de um triângulo
    v2 = tri[1]  #vértice de um triângulo
    v3 = tri[2]  #vértice de um triângulo
    if adet != 0:
        b = [(v1[0] - e[0]), (v1[1] - e[1]), (v1[2] - e[2])]
        x = linalg.solve(a, b)
        if x[0] > 0 and x[1] > 0 and (x[0]+x[1] < 1) and x[2] > 0 and x[2] < FARAWAY:
            return True
        else:
            return False
    else:
        return False


def find_a(v1, v2, v3, d):
    a = [
    [v1[0] - v2[0], v1[0] - v3[0], d[0]],
    [v1[1] - v2[1], v1[1] - v3[1], d[1]],
    [v1[2] - v2[2], v1[2] - v3[2], d[2]],
    ]
    return a



def obj_to_triangles(list, color):  #traz o arquivo obj e coloca os triângulos no formato da classe utilizada pelo programa
    list_triangles = []
    for face in list:
        v1 = vec3(face[0][0],face[0][1], face[0][2])
        v2 = vec3(face[1][0],face[1][1], face[1][2])
        v3 = vec3(face[2][0],face[2][1], face[2][2])
        normal = vec3(face[3][0], face[3][1], face[3][2])
        face_atual = Triangle(v1, v2, v3, color, normal)
        list_triangles.append(face_atual)
    return list_triangles



def telhado_parsing(list):
    global direct
    for i in range (0, len(list), 1):
        face = list[i]
        ver1 = face[0]
        ver2 = face[1]
        ver3 = face[2]
        f_telhado[i][0][0] = face[0][0]
        f_telhado[i][0][1] = face[0][1]
        f_telhado[i][0][2] = face[0][2]
        f_telhado[i][1][0] = face[1][0]
        f_telhado[i][1][1] = face[1][1]
        f_telhado[i][1][2] = face[1][2]
        f_telhado[i][2][0] = face[2][0]
        f_telhado[i][2][1] = face[2][1]
        f_telhado[i][2][2] = face[2][2]
        n_telhado[i][0] = face[3][0]
        n_telhado[i][1] = face[3][1]
        n_telhado[i][2] = face[3][2]
        c_telhado[i][0] = cor_telhado[0]
        c_telhado[i][1] = cor_telhado[1]
        c_telhado[i][2] = cor_telhado[2]
        c_telhado[i][3] = cor_telhado[3]
        a = find_a(ver1, ver2, ver3, direct)
        a_telhado[i] = a
        adet_telhado[i] = linalg.det(a)


def model_parsing(list):
    global direct
    for i in range (0, len(list), 1):
        face = list[i]
        ver1 = face[0]
        ver2 = face[1]
        ver3 = face[2]
        f_model[i][0][0] = face[0][0]
        f_model[i][0][1] = face[0][1]
        f_model[i][0][2] = face[0][2]
        f_model[i][1][0] = face[1][0]
        f_model[i][1][1] = face[1][1]
        f_model[i][1][2] = face[1][2]
        f_model[i][2][0] = face[2][0]
        f_model[i][2][1] = face[2][1]
        f_model[i][2][2] = face[2][2]
        n_model[i][0] = face[3][0]
        n_model[i][1] = face[3][1]
        n_model[i][2] = face[3][2]
        c_model[i][0] = cor_model[0]
        c_model[i][1] = cor_model[1]
        c_model[i][2] = cor_model[2]
        c_model[i][3] = cor_model[3]
        a = find_a(ver1, ver2, ver3, direct)
        a_model[i] = a
        adet_model[i] = linalg.det(a)


def a_current_dir_parsing():
    global luz_dir, a_model_dir, adet_model_dir, adet_model, f_model
    for i in range(0, len(adet_model), 1):
        ver1 = f_model[i][0]
        ver2 = f_model[i][1]
        ver3 = f_model[i][2]
        a = find_a(ver1, ver2, ver3, luz_dir)
        a_model_dir[i] = a
        adet_model_dir[i] = linalg.det(a)


def obj_face_count(list):
	return len(list)


def add_triangles_to_cena(list_triangles):  #adiciona os triângulos na cena
    for triangle in list_triangles:
        cena.append(triangle)



def colorize():
    global pre_mapped_colors, shadow_table, ka, kd, pre_mapped_f
    merge = pre_mapped_colors*shadow_table
    image = (merge*ka) + (merge*pre_mapped_f*kd)
    return image


def return_dot_product_f_matrix():
    global luz_dir, pre_mapped_normals
    fs = np.full((n_y, n_x, 4), 0, dtype=np.double)
    for i in range(0, n_y, 1):
        for j in range(0, n_x, 1):
            n = pre_mapped_normals[i][j]
            f = np.inner(n,luz_dir)
            if f >= 0:
                fs[i][j] = [f,f,f,1]
    return fs


def trace_tri(): #função que emite os raios para um conjunto de triângulos
    coord_list = all_combinations(n_y, n_x)
    resultados = trace_tri_results()
    print("---Passando Resultados---")
    res_ar = list_to_array_reshape(resultados, n_x, n_y)
    return res_ar


def trace_tri_results(): #função que emite os raios para um conjunto de triângulos
    coord_list = all_combinations(n_y, n_x)
    pbar_trace = tqdm(total=len(coord_list))
    resultados_list = []
    pool = Pool(processes=core_count-1)  #aumentar ou diminuir depois
    for result in pool.imap(light_bounce, coord_list, chunksize=n_x):
        resultados_list.append(result)
        pbar_trace.update(1)
    pool.close() # No more work
    pool.join() # Wait for completion
    return resultados_list  #retorna a lista com as cores, precisa ser colocada em shape


def light_bounce(c):  #determina a cor de um certo pixel, para o caso do pré-mapeamento de objetos
    cor = pre_mapped_colors[c[0]][c[1]]
    objeto = pre_mapped[c[0]][c[1]][1]
    if np.array_equal(cor, [0,0,0,0]):
        return [0,0,0,0]
    else:
        intercept_point = pre_mapped_intercept[c[0]][c[1]]
        for i in range (0, len(f_model), 1):
            outro_obj = f_model[i]
            a_atual = a_model_dir[i]
            adet_atual = adet_model_dir[i]
            if (not np.array_equal(outro_obj, objeto)) and intercept_tri_bool_cached_a(outro_obj, intercept_point, a_atual, adet_atual): #se estivermos no telhado, se o triângulo não for ele mesmo e interceptar outro triângulo
                return [0,0,0,1]
        return [1,1,1,1]


def shadow_to_heatmap(tabela):  #transforma os pontos sombreados em uma matriz cujos valores determinam a intensidade do sombreamento
    tabela_return = []
    for i in range (0, len(tabela)):
        linha_return = []
        for j in range (0, len(tabela[0])):
            if area_de_interesse[i][j] != None:
                if np.array_equal(tabela[i][j], [0,0,0,255]):
                    linha_return.append(1)
                else:
                    linha_return.append(0)
            else:
                linha_return.append(-1)
        tabela_return.append(linha_return)
    return tabela_return


def heatmap_to_img(heatmap):  #transforma a matriz dos valores de sombreamento em uma imagem (campo escalar)
    numero_cores = len(heatmap)+1
    colors = color_range(numero_cores)
    soma = np.zeros((n_y,n_x))
    img = []
    for time in heatmap:
        soma = soma + time
    for i in soma:
        line = []
        for j in i:
            if j >= 0:
                line.append([colors[int(j)].red*255,colors[int(j)].green*255,colors[int(j)].blue*255,255])
            else:
                line.append([0,0,0,0])
        img.append(line)
    img1 = Image.fromarray(np.uint8(img)).convert('RGBA')  #Transformando a matriz em uma imagem .png
    img1.save('output/Heatmap.png')
    color_range_image(colors)
    return soma

def pixel_coordinates(n, m):  #determina as coordenadas reais dos pixels da tela
    tabela_pc = []
    for i in range (0,n,1):
        linha_pc = []
        for j in range (0,m,1):
            x_y = pixel_pos(j,i)
            z = depth
            linha_pc.append([x_y[0], x_y[1], z])
        tabela_pc.append(linha_pc)
    return tabela_pc


def object_pre_mapping():   #retorna uma matriz com os objetos tocados pelo primeiro raio emitido
    coord_list = all_combinations(n_y, n_x)
    resultados = object_pre_mapping_results()
    print("---Passando Resultados---")
    res_ar = list_to_array_reshape(resultados, n_x, n_y)
    return res_ar


def object_pre_mapping_results():   #cria um pool de processos para procurar os objetos tocados pelo primeiro raio emitido
    coord_list = all_combinations(n_y, n_x)
    resultados_list = []
    print("---Pré Mapeando Objetos---")
    pbar = tqdm(total=len(coord_list))
    pool = Pool(processes=core_count-1)  #aumentar ou diminuir depois
    results = pool.imap(object_pre_mapping_check_simple, coord_list, chunksize=n_x)
    for result in results:
        resultados_list.append(result)
        pbar.update(1)
    pool.close() # No more work
    pool.join() # Wait for completion
    return resultados_list  #analisar necessidade


def object_pre_mapping_check_simple(c):  #verifica qual é o objeto tocado pelo raio emitido em um certo ponto da tela
    global direct
    intercept_point = None
    pos_ini = coordenadas_pixels[c[0]][c[1]]
    dist_atual = FARAWAY
    obj_atual = [[None]]
    obj_index = None
    for i in range (0, len(f_cena), 1):
        objeto = f_cena[i]
        a_atual = a_cena[i]
        adet_atual = adet_cena[i]
        temp = intercept_tri_cached_a(objeto, pos_ini, c_cena[i], a_atual, adet_atual)
        if temp[1] < dist_atual:  #se a distância da interceptação temp[1] for menor que a distância atual e estiver no telhado
            dist_atual = temp[1]
            intercept_point = ray_p(dist_atual,pos_ini,direct) #descobrimos o ponto dessa interceptação no espaço
            obj_atual = objeto
            obj_index = i
    return [intercept_point, obj_atual, obj_index]

def in_telhado(coisa, lista):
    for i in range (0, len(lista), 1):
        if np.array_equal(coisa, lista[i]):
            return True
    return False

def pre_mapping_vectorize():
    global pre_mapped, pre_mapped_colors, pre_mapped_normals, pre_mapped_intercept, pre_mapped_area_di
    for i in range(0, n_y, 1):
        for j in range(0, n_x, 1):
            a = pre_mapped[i][j]
            if a[1][0][0] != None:
                pre_mapped_colors[i][j] = c_cena[a[2]]
                pre_mapped_normals[i][j] = n_cena[a[2]]
                pre_mapped_intercept[i][j] = a[0]
                if in_telhado(a[1], f_telhado):
                    pre_mapped_area_di[i][j] = 1


def color_range(n_colors):  #cria a faixa de cores usada na imagem do mapa de calor
    if n_colors == 1:
        return [Color("green")]
    elif n_colors == 2:
        return [Color("green"), Color("red")]
    elif n_colors == 3:
        return [Color("white"), Color("green"), Color("red")]
    elif n_colors == 4:
        return [Color("white"), Color("green"), Color("red"), Color("#581845")]
    elif n_colors >= 5:
        final_range = []
        color_division = distribute_for_two(n_colors)
        color0 = Color("blue")
        color1 = Color("yellow")
        range1 = list(color0.range_to(color1,color_division[0]+2))
        del range1[color_division[0]+1]
        del range1[color_division[0]]
        color2 = Color("red")
        range2 = list(color1.range_to(color2,color_division[1]+2))
        del range2[color_division[1]+1]
        del range2[color_division[1]]
        for c1 in range1:
            final_range.append(c1)
        for c2 in range2:
            final_range.append(c2)
        print(final_range)
        return final_range


def area_of_interest_from_pre_mapping_simple():  #obtém a aŕea de interesse a partir do pré-mapeamento de objetos (a forma antiga foi descontinuada)
    tt = np.full((n_y, n_x), None)
    for i in range (0, len(pre_mapped), 1):
        for j in range (0, len(pre_mapped[0]), 1):
            if pre_mapped_area_di[i][j] == 1:
                pp = pre_mapped_intercept[i][j]
                tt[i][j] = vec3(pp[0], pp[1], pp[2])
    return tt


"""Variáveis Globais e Locais"""
time_start = timer.perf_counter()

core_count=multiprocessing.cpu_count()
print("Número de núcleos da CPU: {}".format(core_count))


with open('Raytracer-Config.yaml', "r") as c_file:
    cf = yaml.safe_load(c_file)


offset = cf['MODELO']['OFFSET']
pixel_por_metro = cf['GERAL']['DENSIDADE_PIXEL']
forramento = cf['MODELO']['FORRAMENTO']
FARAWAY = 1.0e39
depth = 10  #profundidade da tela em relação à origem

kd = cf['GERAL']['K_D']  #coeficiente de difusão
ka = cf['GERAL']['K_A']  #coeficiente de ambiente

n_x = 200  #(tamaho da tela em x)
n_y = 150  #tamanho da tela em y)
l = -4    #coordenada x da esquerda da tela
r = 4    #coordenada x da direita da tela
top = 3  #coordenada y do topo da tela
bot = -3  #coordenada y do fim da tela

direct = [0,0,-1]  #direção dos raios de luz

change_to_current_dir()


telhado_obj = parse(cf['MODELO']['AREA_DE_INTERESSE_OBJ'])
modelagem_obj = parse(cf['MODELO']['MODELAGEM_OBJ'])



f_telhado = np.full((obj_face_count(telhado_obj),3, 3), 0, dtype=np.double)
n_telhado = np.full((obj_face_count(telhado_obj), 3), 0, dtype=np.double)
cor_telhado = cf['MODELO']['AREA_DE_INTERESSE_COR']
c_telhado = np.full((obj_face_count(telhado_obj),4), 0, dtype=np.intc)
a_telhado = np.full((obj_face_count(telhado_obj),3, 3), 0, dtype=np.double)
adet_telhado = np.full((obj_face_count(telhado_obj)), 0, dtype=np.double)



f_model = np.full((obj_face_count(modelagem_obj),3, 3), 0, dtype=np.double)
n_model = np.full((obj_face_count(modelagem_obj), 3), 0, dtype=np.double)
cor_model = cf['MODELO']['MODELAGEM_COR']
c_model = np.full((obj_face_count(modelagem_obj),4), 0, dtype=np.intc)
a_model = np.full((obj_face_count(modelagem_obj),3, 3), 0, dtype=np.double)
adet_model = np.full((obj_face_count(modelagem_obj)), 0, dtype=np.double)


telhado_parsing(telhado_obj)
model_parsing(modelagem_obj)

f_cena = np.concatenate((f_telhado, f_model))
n_cena = np.concatenate((n_telhado, n_model))
c_cena = np.concatenate((c_telhado, c_model))
a_cena = np.concatenate((a_telhado, a_model))
adet_cena = np.concatenate((adet_telhado, adet_model))


print("Modelos: {} e {}".format(cf['MODELO']['AREA_DE_INTERESSE_OBJ'], cf['MODELO']['MODELAGEM_OBJ']))
print("Número de Faces: {}".format(obj_face_count(f_cena)))

screen_size_forr(f_cena)

print(n_x, n_y)


# a direção "d" do raio é sempre -w [0,0,-1] (vetor unitário)

luz_dir = vec3(-0.6247,0.6247,0.46852)  #direção da origem até a luz, unitário


#fonte: https://www.sunearthtools.com/dp/tools/pos_sun.php?lang=pt
sunpath = cf['TRAJETORIA']

"""Inicialização"""
cont = 0
heatmap = []

coordenadas_pixels = pixel_coordinates(n_y,n_x)
coordenadas_intercept = []


"""Vetorização do código"""
pre_mapped_colors = np.full((n_y, n_x, 4), 0, dtype=np.intc)
pre_mapped_normals = np.full((n_y, n_x, 3), 0, dtype=np.double)
pre_mapped_intercept = np.full((n_y, n_x, 3), np.inf, dtype=np.double)
pre_mapped_area_di = np.full((n_y, n_x), 0, dtype=np.intc)


#tab_area_of_interest_base = multiprocessing.Array()   #estudar retirar essa linha

pre_mapped = object_pre_mapping()
pre_mapping_vectorize()
#area_de_interesse = area_of_interest()  #area_de_interesse: matriz de coordenadas em vec3
area_de_interesse = area_of_interest_from_pre_mapping_simple()

for time in sunpath:
    cont = cont+1
    print("----- Etapa {} de {} ------".format(cont,len(sunpath)))
    luz_dir = polar_to_vector_ajustado_array(time[0], time[1], offset)
    a_model_dir = np.full_like(a_cena, 0, dtype=np.double)
    adet_model_dir = np.full_like(adet_cena, 0, dtype=np.double)
    a_current_dir_parsing()
    shadow_table = trace_tri()
    pre_mapped_f = return_dot_product_f_matrix()
    tabela_cor = colorize()
    img1 = Image.fromarray(np.uint8(tabela_cor)).convert('RGBA')  #Transformando a matriz em uma imagem .png
    img1.save('output/{}-{}.png'.format(cf['DADOS']['NOME_IMAGENS'],cont))


print("---------------Terminou-------------------")

time_elapsed = (timer.perf_counter() - time_start)
print("Tempo de Execução: %5.1f secs" % (time_elapsed))

#python3 RaytracerMP-VECT4.py
