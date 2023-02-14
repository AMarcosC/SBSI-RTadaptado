# SBSI-RaytracerMP

## Instalação

O programa foi produzido utilizando o sistema operacional Ubuntu 20.04.4 LTS, com a versão do 3.8.10 do Python. Esta versão e versões superiores do Ubuntu devem permitir a execução do programa de forma nativa. Além disso, qualquer sistema operacional com uma distribuição do Python na versão 3.8.2 ou superior devem rodar o programa. Versões do Python 3.x inferiores também devem rodar o programa, mas talvez não suportem algumas das bibliotecas utilizadas. Para execução em sistemas operacionais Windows, é recomendado o uso de uma distribuição como a [Anaconda](https://www.anaconda.com/products/distribution), ou o uso do terminal [Ubuntu on Windows](https://apps.microsoft.com/store/detail/ubuntu-on-windows/9NBLGGH4MSV6?hl=pt-br&gl=br), que reproduz as funcionalidades do terminal do Ubuntu no Windows, e já possui Python instalado.

Recomenda-se que o repositório seja utilizado dentro de um ambiente virtual (virtual enviroment). As bibliotecas instaladas dentro de um ambiente virtual não interferem nas bibliotecas instaladas globalmente, permitindo que vários programas utilizem as versões apropriadas destas bibliotecas.

- Tutorial para criar ambiente virtual no [Anaconda Navigator](https://docs.anaconda.com/navigator/getting-started/#navigator-managing-environments)
- Tutorial para criar ambiente virtual no [Anaconda Prompt](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)
- Tutorial para criar ambiente virtual no [Terminal do Ubuntu](https://www.arubacloud.com/tutorial/how-to-create-a-python-virtual-environment-on-ubuntu.aspx)

Antes de executar o código, é preciso instalar as dependências do programa. Isto pode ser feito de forma simples, a partir do arquivo `requirements.txt` na raiz do repositório. Estando na raiz do repositório, basta que se execute no terminal o seguinte comando:

```
pip install -r requirements.txt
```

Em alguns casos, pode ser necessário rodar o comando da seguinte forma:

```
pip3 install -r requirements.txt
```

As dependências do programa são:

```
colour==0.1.5  #para trabalhar com cores em vários sistemas
numpy==1.22.3  #principal biblioteca para operações matemáticas básicas com matrizes
Pillow==9.3.0  #sintetização de imagens
PyYAML==6.0    #uso de arquivos e dicionários no formato .yaml
tqdm==4.64.1   #barras de progresso (só são mostradas no prompt)
```

## Raytracer

O programa desenvolvido na etapa (i) do trabalho de Paz (2022), sendo o código original da implementação, relacionado ao traçado de raios e produção do campo escalar de sombreamento, pode ser executado da seguinte maneira, estando-se na raiz do repositório:

```
python RaytracerMP-Original.py
```

Em alguns casos, pode ser necessária a execução da seguinte forma

```
python3 RaytracerMP-Original.py
```

Já as adaptações apresentadas no artigo podem ser executadas da seguinte maneira:

```
python RaytracerMP-Adaptado.py
```

Em alguns casos, pode ser necessária a execução da seguinte forma

```
python3 RaytracerMP-Adaptado.py
```

Antes de se executar o programa, porém, é necessário que se definam algumas propriedades no arquivo `Raytracer-Config.yaml`. Este arquivo pode ser aberto com o editor de texto padrão do Ubuntu, como também pelo bloco de notas no Windows. As propriedades possuem um comentário explicando de forma resumida a sua destinação.

O usuário deverá produzir dois arquivos, os modelos do telhado e da modelagem, em formato `.obj`. É possível fazer a modelagem no *Autodesk Revit*, exportar em `.fbx`, e converter este modelo para `.obj` no *Blender*, como explicado no trabalho. Caso o usuário queira apenas performar um teste, os modelos apresentados no artigo estão presentes na pasta `assets`. Para que se execute o modelo A1 como foi executado no artigo, é necessário que se utilizem os seguintes valores no arquivo `Raytracer-Config.yaml`:

```
DENSIDADE_PIXEL: 10
AREA_DE_INTERESSE_OBJ: 'assets/A1-TEL.obj'  #referente ao telhado do modelo
MODELAGEM_OBJ: 'assets/A16-PAR.obj'         #referente às paredes do modelo
```

É importante salientar que por ser uma versão de testes, a inserção de valores fora do alcance do programa, ou incompreensíveis para ele, pode gerar erros na execução. O plano é inserir estas limitações nas variáveis em uma futura versão que possua uma interface gráfica.

O arquivo `BasicFunctions.py` traz algumas das funções básicas utilizadas pelo programa, que não estão nos códigos principais para manter um padrão de organização.

O arquivo `OBJFileParser.py` traz funções que dizem respeito à leitura e conversão de arquivos `.obj` em listas de triângulos.

Os códigos presentes em `python RaytracerMP-Original.py` e `python RaytracerMP-Adaptado.py` são autorais, mas receberam modificações para que seja executada apenas a etapa do traçado de raios, sem retornar o mapa de sombreamento resultante, visto que existem interesses comerciais em relação ao código original. A visualização e execução do código presente neste repositório, como também a reprodução dos seus resultados, é permitida a qualquer usuário, mas não o seu uso comercial.

