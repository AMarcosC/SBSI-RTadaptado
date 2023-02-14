# TCC-Raytracer e TCC-PanelPlacer

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

O programa desenvolvido na etapa (i) do trabalho, relacionado ao traçado de raios e produção do campo escalar de sombreamento, pode ser executado da seguinte maneira, estando-se na raiz do repositório:

```
python RaytracerMP.py
```

Em alguns casos, pode ser necessária a execução da seguinte forma

```
python3 RaytracerMP.py
```

O código `RaytracerMP.py` corresponde à versão que possui multiprocessamento. Não se recomenda a execução da versão em `Raytracer.py` por ser uma versão mais lenta e com funções defasadas, já tendo sido descontinuada. Ela está presente no repositório apenas para casos de incompatibilidade com a versão com multiprocessamento.

Antes de se executar o programa, porém, é necessário que se definam algumas propriedades no arquivo `Raytracer-Config.yaml`. Este arquivo pode ser aberto com o editor de texto padrão do Ubuntu, como também pelo bloco de notas no Windows. As propriedades possuem um comentário explicando de forma resumida a sua destinação.

O usuário deverá produzir dois arquivos, os modelos do telhado e da modelagem, em formato `.obj`. É possível fazer a modelagem no *Autodesk Revit*, exportar em `.fbx`, e converter este modelo para `.obj` no *Blender*, como explicado no trabalho. Caso o usuário queira apenas performar um teste, alguns modelos estão presentes na pasta `assets` na raiz do repositório, e também existem exemplos prontos, como será explicado adiante.

É importante salientar que por ser uma versão de testes, a inserção de valores fora do alcance do programa, ou incompreensíveis para ele, pode gerar erros na execução. O plano é inserir estas limitações nas variáveis em uma futura versão que possua uma interface gráfica.

## PanelPlacer

O programa desenvolvido na etapa (ii) do trabalho, relacionado à inserção de placas de forma a promover o menor sombreamento possível, pode ser executado da seguinte maneira, estando-se na raiz do repositório:

```
python PanelPlacer.py
```

Em alguns casos, pode ser necessária a execução da seguinte forma

```
python3 PanelPlacer.py
```

Antes de se executar o programa, porém, é necessário que se definam algumas propriedades no arquivo `PanelPlacer-Config.yaml`. Este arquivo pode ser aberto com o editor de texto padrão do Ubuntu, como também pelo bloco de notas no Windows. As propriedades possuem um comentário explicando de forma resumida a sua destinação.

O usuário deverá fornecer os dois arquivos binários produzidos pelo Raytracer: a área de interesse (`area`) e o campo escalar (`heatmap`). Para a sobreposição da imagem da placa na imagem do campo escalar, é necessário fornecer também a imagem da última.

É importante salientar que por ser uma versão de testes, a inserção de valores fora do alcance do programa, ou incompreensíveis para ele, pode gerar erros na execução. O plano é inserir estas limitações nas variáveis em uma futura versão que possua uma interface gráfica.

## Auxiliares

O arquivo `BasicFunctions.py` traz algumas das funções básicas utilizadas pelo programa, que não estão nos códigos principais para manter um padrão de organização.

O arquivo `OBJFileParser.py` traz funções que dizem respeito à leitura e conversão de arquivos `.obj` em listas de triângulos.

Os códigos presentes em ambos os arquivos também são autorais, com o auxílio de algumas bibliotecas.

## Utilidades

Alguns outros programas foram produzidos para auxiliar ou tratar os resultados. Na pasta `utilities`, temos:

- `JoinHeatmaps.py`, destinado a juntar os binários de etapas diferentes da execução de um mesmo modelo em um único binário e uma única imagem. Quando o trabalho fala em "etapas de execução" na Metodologia, quer dizer que a execução foi feita em partes (produzindo um binário para cada conjunto de períodos na trajetória solar), que foram somados utilizando este código;
- `PanelNumber.py`, para recompor o número das placas na imagem gerada. Em alguns casos, a numeração das placas fica muito grande ou muito pequena, e este programa pode ser usado para fazer um ajuste fino no tamanho da fonte da numeração, inclusive utilizando a fonte `Roboto-Black.ttf` presente na pasta `utilities`.

Na pasta raiz, há também o programa `RT-Retest.py`, que serve para aplicar a modelagem das placas inseridas com o PanelPlacer novamente no algoritmo de traçado de raios. Trata-se de um código muito semelhante ao presente em `RaytracerMP.py`, porém com uma terceira entrada para o arquivo `.obj` das placas locadas.

## Exemplos

Exemplos para demonstração do código estão presentes na pasta "legacy-examples". A execução do exemplo necessita da instalação das bibliotecas presentes em **requirements.txt**. Para instalar as dependências, basta executar *pip install -r requirements.txt* (o uso de um *virtual environment* é recomendado).

Executando-se **legacy-examples/raytracer - modelo 01/Raytracer.py**, serão obtidos os resultados presentes
no TCC 01:

~~~
- Dados do exemplo:
    - Modelo: telhado simples, separado entre o modelo tridimensional do telhado (assets/Telhado-Telhado.obj)
    e o modelo tridimensional do resto da casa, nesse caso, basicamente, as paredes (assets/Telhado-Parede.obj)
    - Localidade: Juazeiro do Norte-CE
    - Data: 04/07/2022, de 07h às 14h
    - Dados de azimute e elevação obtidos do site sunearthtools.com (comentar as entradas da lista no fim
    do código para diminuir o tempo de execução)

- Output do exemplo (em output):
    - 8 fotos correspondentes à renderização utilizando ray-tracing, para cada hora do dia
    - 1 foto correspondente ao campo escelar de sombreamento.
~~~

## Versão atual

Os arquivos soltos na pasta raiz do repositório, como também as pastas *teste*, *assets* e *output*, são as versões que estão sendo constantemente modificadas. Portanto, a execução dos códigos presentes na raiz pode resultar em erros devido às constantes modificações no código.


## Recursos adicionados recentemente

- Uso de arquivos `.yaml` para as variáveis do programa, não sendo mais necessário acessar o código principal para modificá-las.

## Melhorias Pendentes

- Aplicar uso da GPU no processamento do traçado de raios;
- Processar o ray-tracing apenas na área de interesse, agilizando o código;
- Correção das dimensões reais da placa ao mudar a orientação durante a locação;
- Produzir arquivo com as coordenadas das placas para inserir no AutoCAD, por exemplo; A utilização da biblioteca `pyautocad` se mostrou uma ótima alternativa, mas a sua inconstância no funcionamento para atuais versões do *AutoCAD* a torna uma solução pouco desejável. Seu repositório não é mais atualizado desde 2016. Um arquivo que retorne as coordenadas das placas pode ser utilizado por um outro programa produzido em outra linguagem, ou inclusive uma rotina `AutoLisp`, que o *AutoCAD* rode nativamente. O arquivo `lista_placas` possui estas coordenadas, mas em binário, e dentro da classe criada pelo programa.
- Classificar o resultado por um grau de compacidade, relativo a quantidade de áreas livres e seus valores, nas modalidades *livre* e *livre alternando orientação*;
- Uso de arquivos `.csv` ou planilhas similares, para tornar a inserção da trajetória solar mais prática e menos propensa a erros. Esta medida não foi adotada devido ao desejo de se utilizar uma interface gráfica que torne toda a inserção de dados pelo usuário mais cômoda.
