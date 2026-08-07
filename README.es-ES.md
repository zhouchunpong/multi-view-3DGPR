

# Actualización de 2024
Proporcionamos un enlace de Google Drive para descargar nuestro conjunto de datos.

# multi-view-3DGPR
Código de referencia del artículo **Multi-View Fusion and Distillation for Subgrade Distresses Detection based on 3D-GPR**
![image](https://github.com/zhouchunpong/multi-view-3DGPR/assets/6890539/71a8494a-c189-45e1-a458-681ea5661a30)
Si desea conocer más sobre este conjunto de datos y el nuevo método, lea nuestro artículo [enlace](https://arxiv.org/abs/2308.04779).

## Resumen
El radar de penetración terrestre 3D (3D-GPR) se ha utilizado ampliamente en la detección de deterioros en la subrasante. Para mejorar la eficiencia y precisión de la detección, estudios pioneros han intentado adoptar técnicas de detección automática como el aprendizaje profundo. Sin embargo, los trabajos existentes suelen depender de datos GPR tradicionales 1D A-scan, 2D B-scan o 3D C-scan, lo que resulta en una información espacial insuficiente o una alta complejidad computacional. Para abordar estos desafíos, presentamos una nueva metodología para la tarea de detección de deterioros en la subrasante, aprovechando la información multivista de los datos GPR 3D estándar. En consecuencia, construimos un conjunto de datos de imágenes multivista real derivado de los datos GPR 3D estándar para la tarea de detección, proporcionando información espacial más rica en comparación con los datos A-scan y B-scan, y reduciendo la complejidad computacional de los datos en comparación con los datos C-scan. Posteriormente, desarrollamos un nuevo marco de Fusión y Destilación Multivista, llamado GPR-MVFD, diseñado específicamente para utilizar de manera óptima el conjunto de datos GPR multivista. Este marco incorpora ingeniosamente la fusión basada en atención y la destilación multivista para facilitar la extracción de características significativas para los deterioros en la subrasante. Además, se adopta un mecanismo de aprendizaje autoadaptativo para estabilizar el entrenamiento del modelo y prevenir la degradación del rendimiento en cada rama. Los experimentos exhaustivos demuestran el valor del nuevo conjunto de datos GPR construido y muestran la efectividad y eficiencia de nuestro GPR-MVFD propuesto. Cabe destacar que nuestro marco no solo supera los métodos base GPR existentes, sino también los métodos del estado del arte en los campos del aprendizaje multivista, aprendizaje multimodal y destilación de conocimiento.

# Conjunto de Datos multi-view-3DGPR



<img src="https://github.com/zhouchunpong/multi-view-3DGPR/assets/6890539/efecdad8-08b3-48f1-b845-077b9f7c08c9"  width="50%" />


El conjunto de datos está disponible ahora: 

Enlace de Baidu Drive: https://pan.baidu.com/s/14uZ6F0NbQxgwfaWTERQX2Q 
Contraseña: 2023

Enlace de Google Drive: https://drive.google.com/drive/folders/1TbZCAUq7GEWRk7dUo3CmuhyE1DUnKoen?usp=sharing



# Requisitos
* python 3.7
* pytorch 1.11.0
* cuda 11.3


## Utilización

Para entrenar el modelo descrito en el artículo, ejecute el siguiente comando:

```
python ./code/train_DenseNet121_MVFD.py
```

Para evaluar el modelo entrenado, ejecute el siguiente comando:

```
python ./code/test_DenseNet121_MVFD.py
```




## Cita
Si considera útil nuestro artículo o conjunto de datos, por favor citearnosnos.
```bash
@article{zhou2023multi,
  title={Multi-View Fusion and Distillation for Subgrade Distresses Detection based on 3D-GPR},
  author={Zhou, Chunpeng and Ning, Kangjie and Wang, Haishuai and Yu, Zhi and Zhou, Sheng and Bu, Jiajun},
  journal={arXiv preprint arXiv:2308.04779},
  year={2023}
}
```
