import numpy as np, cv2, glob
import matplotlib.pyplot as plt
from variables import savedModelsDir, latent_dim, generatedImagesDir, videoDir, action


def saveimages(epoch, decoder):
    r, c = 3, 3
    z = np.random.normal(size=(r * c, latent_dim))
    gen_imgs = decoder.predict(z)
    gen_imgs = gen_imgs + 0.2
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(generatedImagesDir + '%d.png' % epoch)
    plt.close()


def saveModel(decoder, discriminator):
    json_string = decoder.to_json()
    open(savedModelsDir + 'decoder_model.json', 'w').write(json_string)
    decoder.save_weights(savedModelsDir + 'decoder_model.h5', overwrite=True)

    json_string = discriminator.to_json()
    open(savedModelsDir + 'discriminator_model.json', 'w').write(json_string)
    discriminator.save_weights(savedModelsDir + 'discriminator_model.h5', overwrite=True)


def saveModelBkp(decoder, discriminator):
    json_string = decoder.to_json()
    open(savedModelsDir + 'decoder_model.json.bkp', 'w').write(json_string)
    decoder.save_weights(savedModelsDir + 'decoder_model.h5.bkp', overwrite=True)

    json_string = discriminator.to_json()
    open(savedModelsDir + 'discriminator_model.json.bkp', 'w').write(json_string)
    discriminator.save_weights(savedModelsDir + 'discriminator_model.h5.bkp', overwrite=True)


def generateImages(epoch, decoder):
    r, c = 3, 3
    z = np.random.normal(size=(r * c, latent_dim))
    gen_imgs = decoder.predict(z)
    gen_imgs = gen_imgs + 0.2
    cnt = 0

    for i in range(r):
        for j in range(c):
            plt.imshow(gen_imgs[cnt, :, :, 0])
            plt.savefig(generatedImagesDir + str(i) + '_' + str(j) + '_' + str(epoch) + '.png')
            cnt += 1


def createVideo():
    img_array = []
    for filename in glob.glob(generatedImagesDir + '*.png'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(videoDir + action + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()