img_rows = 120
img_cols = 160
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 9
batch_size = 16
epsilon_std = 1.0
action = 'handwaving'
repository = '/Users/jay/'
videoDir = repository + action + '/video/'
sourceImagesDir = repository + action + '/dataSetImg/'
generatedImagesDir = repository + action + '/generated/'
savedModelsDir = repository + action + '/models/'

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
repository = dir_path + '/'
print repository