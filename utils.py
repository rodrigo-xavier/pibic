
def ndarray2img(ndarray, colormap):
    import matplotlib.pyplot as plt
    plt.imshow(ndarray, cmap=plt.get_cmap(colormap))
    plt.show()
    # exit()