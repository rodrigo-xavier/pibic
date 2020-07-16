import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PATH = "/home/cyber/GitHub/pibic/pibic/database/pygame"

def show_array_as_img(ndarray, colormap):
    plt.imshow(ndarray, cmap=plt.get_cmap(colormap))
    plt.show()
    # exit()

def store_csv(action, observation, counter):
    n = np.zeros((observation.shape[-1],), dtype=int)   # Gera uma linha de zeros do tamanho do eixo x de observation
    n[0] = action
    array = np.append([n], observation, axis=0)         # Concatena o array de actions com a imagem

    path = PATH + "/array/" + str(counter) + ".csv"
    np.savetxt(path, array, fmt="%d", delimiter=',')

def store_png(action, observation, counter):
    n = np.zeros((observation.shape[-1],), dtype=int)   # Gera uma linha de zeros do tamanho do eixo x de observation
    n[0] = action
    array = np.append([n], observation, axis=0)         # Concatena o array de actions com a imagem

    img = Image.fromarray(array)
    img = img.convert("L")

    path = PATH + "/img/" + str(counter) + ".png"
    img.save(path)

def store_npz(observation_list, action_list):
    path = PATH + "/npz/" + "observation_list.npz"
    np.savez_compressed(path, observation_list)

    path = PATH + "/npz/" + "action_list.npz"
    np.savez_compressed(path, action_list)

def png2npz(img):
    path = PATH + "/npz/" + "img.npz"
    np.savez_compressed(path, img)

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