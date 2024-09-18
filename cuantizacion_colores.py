#==================================
#Cuantización de colores usando K-Means
#==================================

#Realiza una cuantización de vectores (VQ) por píxel de una imagen del Palacio de Verano
#(China), reduciendo el número de colores necesarios para mostrar la imagen de 96,615
#colores únicos a 64, preservando la calidad visual general.

#En este ejemplo, los píxeles se representan en un espacio 3D y se usa K-means para
#encontrar 64 agrupaciones de colores. En la literatura de procesamiento de imágenes,
#el libro de códigos obtenido de K-means (los centros de las agrupaciones) se llama
#la paleta de colores. Usando un solo byte, se pueden direccionar hasta 256 colores,
#mientras que una codificación RGB requiere 3 bytes por píxel. El formato de archivo
#GIF, por ejemplo, usa una paleta de este tipo.

#Para la comparación, también se muestra una imagen cuantizada utilizando un libro de
#códigos aleatorio (colores seleccionados al azar).


# Autores: Robert Layton <robertlayton@gmail.com>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#
# Licencia: BSD 3 cláusulas

from time import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle

n_colors = 64  # Número de colores para la cuantización

# Cargar la foto del Palacio de Verano
china = load_sample_image("china.jpg")

# Convertir a floats en lugar de la codificación predeterminada de enteros de 8 bits. Dividir por
# 255 es importante para que plt.imshow funcione bien con datos flotantes (deben estar
# en el rango [0-1])
china = np.array(china, dtype=np.float64) / 255

# Cargar la imagen y transformarla en un arreglo numpy 2D.
w, h, d = original_shape = tuple(china.shape)
assert d == 3  # Asegura que la imagen tiene 3 canales (RGB)
image_array = np.reshape(china, (w * h, d))  # Reestructura la imagen en una matriz 2D

print("Ajustando el modelo con una pequeña submuestra de los datos")
t0 = time()  # Marca el tiempo de inicio
# Toma una muestra aleatoria de 1,000 píxeles de la imagen
image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
# Ajusta el modelo K-Means en la submuestra
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
print(f"terminado en {time() - t0:0.3f}s.")  # Muestra el tiempo que tardó

# Obtener etiquetas para todos los puntos
print("Prediciendo índices de color en toda la imagen (k-means)")
t0 = time()
# Predice las etiquetas de todos los píxeles con base en el modelo K-Means ajustado
labels = kmeans.predict(image_array)
print(f"terminado en {time() - t0:0.3f}s.")  # Tiempo de ejecución

# Crear un libro de códigos aleatorio para comparar
codebook_random = shuffle(image_array, random_state=0, n_samples=n_colors)
print("Prediciendo índices de color en toda la imagen (aleatorio)")
t0 = time()
# Asigna a cada píxel un color del libro de códigos aleatorio
labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)
print(f"terminado en {time() - t0:0.3f}s.")

def recrear_imagen(codebook, labels, w, h):
    """Recrear la imagen (comprimida) a partir del libro de códigos y las etiquetas"""
    return codebook[labels].reshape(w, h, -1)

# Mostrar todos los resultados, junto con la imagen original
plt.figure(1)
plt.clf()
plt.axis("off")
plt.title("Imagen original (96,615 colores)")
plt.imshow(china)

plt.figure(2)
plt.clf()
plt.axis("off")
plt.title(f"Imagen cuantizada ({n_colors} colores, K-Means)")
plt.imshow(recrear_imagen(kmeans.cluster_centers_, labels, w, h))

plt.figure(3)
plt.clf()
plt.axis("off")
plt.title(f"Imagen cuantizada ({n_colors} colores, Aleatorio)")
plt.imshow(recrear_imagen(codebook_random, labels_random, w, h))
plt.show()

