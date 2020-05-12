import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PATH = "/home/cyber/GitHub/pibic/pibic/database"

def show_array_as_img(ndarray, colormap):
    plt.imshow(ndarray, cmap=plt.get_cmap(colormap))
    plt.show()
    # exit()

def store_array_and_img(action, ndarray, counter):
    n = np.zeros((ndarray.shape[-1],), dtype=int)   # Gera uma linha de zeros do tamanho do eixo x de ndarray
    n[0] = action
    array = np.append([n], ndarray, axis=0)         # Concatena o array de actions com a imagem

    path = PATH + "/array/" + str(counter) + ".csv"
    np.savetxt(path, array, fmt="%d", delimiter=',')

    img = Image.fromarray(array)
    img = img.convert("L")

    path = PATH + "/img/" + str(counter) + ".png"
    img.save(path)


def print_all_array(array):
    np.set_printoptions(threshold=np.inf)
    print(array)
    exit()

def arguments():
    # comandos de linha de comando
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--env-name', type=str, default='SpaceInvaders-v0')
    parser.add_argument('--weights', type=str, default=None)
    args = parser.parse_args()
    return args