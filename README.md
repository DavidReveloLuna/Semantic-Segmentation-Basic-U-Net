# Segmentación Semántica Modelo Básico U-Net
Vamos a entrenar un modelo U-Net básico para realizar segmentación semántica usando Keras y el dataset Electron Microscopy Datset

[Paper original](https://arxiv.org/pdf/1505.04597.pdf)

![Modelo U-Net](https://github.com/DavidReveloLuna/Semantic-Segmentation-Basic-U-Net/blob/master/assets/ArquitecturaUnet.png)

## 1. Preparación del Entorno

    $ conda create -n Unet anaconda python=3.7
    $ conda activate Unet
    $ conda install tensorflow-gpu==2.1.0 cudatoolkit=10.1
    $ pip install tensorflow==2.1.0
    $ pip install jupyter
    $ pip install keras==2.3.1
    $ pip install numpy scipy Pillow cython matplotlib scikit-image opencv-python h5py imgaug IPython[all] tqdm
    $ conda install ipykernel
    $ python -m ipykernel install --user --name Unet --display-name "Unet"
    
## 2. Descargar el dataset Electron Microscopy Dataset

    Descargar y descomprimir la carpeta, copiar las carpetas stage1_train y stage1_test para el entrenamiento y pruebas

[Download Dataset](https://www.epfl.ch/labs/cvlab/data/data-em/)

## 3. Entrenamiento del Modelo

    Ejecutar el documento de Jupyter Notebook
[Basic U-Net](https://github.com/DavidReveloLuna/Semantic-Segmentation-Basic-U-Net/blob/master/BasicUnet.ipynb)

## 3. Resultados

![Imagen](https://github.com/DavidReveloLuna/Semantic-Segmentation-Basic-U-Net/blob/master/assets/image.png)
![Segmentación](https://github.com/DavidReveloLuna/Semantic-Segmentation-Basic-U-Net/blob/master/assets/mask.png)
    

