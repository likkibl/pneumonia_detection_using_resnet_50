#Importing libraries
from fastai import *
from fastai.vision import *
from fastai.metrics import accuracy
import os
import pandas as pd
import numpy as np

# Setting the data path
x = '/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train'
path = Path(x)
path.ls()

# Setting random seed
np.random.seed(40)

# Data preparation with enhanced augmentations
data = ImageDataBunch.from_folder(
    path, train='.', valid_pct=0.2,
    ds_tfms=get_transforms(
        flip_vert=True, max_rotate=30, max_zoom=1.2, max_warp=0.2
    ),
    size=224, num_workers=4
).normalize(imagenet_stats)


 #Display a batch
data.show_batch(rows=3, figsize=(7, 6), recompute_scale_factor=True)
print(data.classes)
len(data.classes)
data.c

# Defining the learner with ResNet-50
learn = cnn_learner(
    data, models.resnet50, metrics=[accuracy],
    model_dir=Path('../kaggle/working'), path=Path(".")
)


# Finding the optimal learning rate
learn.lr_find()
learn.recorder.plot(suggestions=True)
import logging
logging.getLogger('torch').setLevel(logging.ERROR)



# Training the model (frozen layers)
lr1 = 1e-3
lr2 = 1e-1
learn.fit_one_cycle(5, slice(lr1, lr2))



# Unfreezing the model for fine-tuning
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()




# Fine-tuning with a smaller learning rate
learn.fit_one_cycle(5, slice(1e-4, 1e-3))

# Plotting training and validation loss
learn.recorder.plot_losses()





# Access training losses
train_losses = [float(loss) for loss in learn.recorder.losses]

# Access validation metrics (e.g., accuracy)
valid_metrics = [metric[0] for metric in learn.recorder.metrics]

# Plot training losses
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot validation accuracy (if available)
if valid_metrics:
    plt.figure(figsize=(10, 5))
    plt.plot(valid_metrics, label='Validation Metric')
    plt.title('Validation Metric')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.show()





# Confusion Matrix
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# Testing the model with a new image
img = open_image('/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL/IM-0003-0001.jpeg')
print("Prediction:", learn.predict(img)[0])


# Exporting the model
learn.export(file=Path("/kaggle/working/export.pkl"))
learn.model_dir = "/kaggle/working"
learn.save("stage-1", return_path=True)