# Segmentación Semántica Modelo Básico U-Net
Vamos a entrenar un modelo U-Net básico para realizar segmentación semántica usando Keras y el dataset Data Science Bowl 2018 para la detección de nucleos celulares

[Tutorial de Youtube](https://www.youtube.com/watch?v=3v7sYqigeSc&list=PLsjK_a5MFguLIBZQqxDvgUTp2SegKmMFH&index=18)

[Paper original](https://arxiv.org/pdf/1505.04597.pdf)

![Modelo U-Net](https://github.com/DavidReveloLuna/Semantic-Segmentation-Basic-U-Net/blob/master/assets/ArquitecturaUnet.png)

## 1. Preparación del Entorno

    $ conda create -n Unet anaconda python=3.7
    $ conda activate Unet
    $ conda install tensorflow-gpu==2.1.0 cudatoolkit=10.1 tensorboard==2.1.0 tensorflow-estimator==2.1.0
    $ pip install tensorflow==2.1.0
    $ pip install jupyter
    $ pip install keras==2.3.1
    $ pip install numpy scipy Pillow cython matplotlib scikit-image opencv-python h5py imgaug IPython[all] tqdm
    $ conda install ipykernel
    $ python -m ipykernel install --user --name Unet --display-name "Unet"
    
## 2. Descargar el dataset Data Science Bowl

    Descargar y descomprimir la carpeta, copiar las carpetas stage1_train y stage1_test para el entrenamiento y pruebas

[Download Dataset](https://www.kaggle.com/c/data-science-bowl-2018/data)

## 3. Entrenamiento del Modelo

    Ejecutar el documento de Jupyter Notebook
[Basic U-Net](https://github.com/DavidReveloLuna/Semantic-Segmentation-Basic-U-Net/blob/master/BasicUnet.ipynb)

## 3. Resultados

![Imagen](https://github.com/DavidReveloLuna/Semantic-Segmentation-Basic-U-Net/blob/master/assets/image.png)
![Segmentación](https://github.com/DavidReveloLuna/Semantic-Segmentation-Basic-U-Net/blob/master/assets/mask.png)
    

