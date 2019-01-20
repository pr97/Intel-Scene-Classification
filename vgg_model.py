
# coding: utf-8

# In[1]:


from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.applications.vgg19 import VGG19
from keras.models import load_model
import tensorflow as tf
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


use_log_dir='X:/train-scene classification/logs_vgg19' # Change for every new training session 

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir=use_log_dir, **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


# In[3]:


img_width, img_height = 150, 150
train_data_dir = "X:/train-scene classification/data/train/"
validation_data_dir = "X:/train-scene classification/data/valid/"
test_data_dir = "X:/train-scene classification/data/test/"
nb_train_samples = 16384
nb_validation_samples = 650 
batch_size = 32
epochs = 75
# Save the model according to the conditions  
chkpt_dir = "X:/train-scene classification/chkpts/vgg19.h5" # Change for every new training session 


# In[4]:


model = VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3), classes = 6)


# In[5]:


model.save('vgg19.h5')
# model = load_model('vgg19.h5')


# In[6]:


# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
# for layer in model.layers[:200]:
#     layer.trainable = False

model.summary()
print(len(model.layers))


# In[7]:


#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.3)(x)
predictions = Dense(6, activation="softmax")(x)


# In[8]:


# creating the final model 
model_final = Model(input = model.input, output = predictions)
model_final.summary()


# In[9]:


# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8), metrics=["accuracy"])


# In[10]:


# model_final.load_weights(chkpt_dir)


# In[11]:


# Initiate the train and test generators with data Augumentation 
train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=10)

validation_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=10)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode = "categorical")

validation_generator = validation_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width),
class_mode = "categorical")


# In[12]:


# Save the model according to the conditions  
# chkpt_dir = "X:/train-scene classification/chkpts/densenet_extra_fc.h5" # Change for every new training session 

checkpoint = ModelCheckpoint(chkpt_dir, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=12, verbose=1, mode='auto')
#[TrainValTensorBoard(write_graph=False)]


# In[13]:


import os

# Train the model 
model_final.fit_generator(
train_generator,
samples_per_epoch = nb_train_samples,
epochs = epochs,
validation_data = validation_generator,
nb_val_samples = nb_validation_samples,
callbacks = [checkpoint, early, TrainValTensorBoard(write_graph=False)])


# In[ ]:


#Saving Model
# model_final.save("trained_densenet_v1.h5")


# ## Testing + Predictions

# In[14]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
test_data_dir_2 = "X:/train-scene classification/data/test/final_test"


# In[15]:


labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())
print(labels)
# predictions = [labels[k] for k in predicted_class_indices]


# In[47]:


# train_dir = "X:/train-scene classification/data/train/1_forest"
# img = cv2.imread(os.path.join(train_dir, "466.jpg"))

# valid_dir = "X:/train-scene classification/data/valid/1_forest"
# img = cv2.imread(os.path.join(valid_dir, "2800.jpg"))

img = cv2.imread(os.path.join(test_data_dir_2, "398.jpg"))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
print(img.shape)
img = np.reshape(img, (1, img_width, img_height, 3))
pred = np.argmax(model_final.predict(img), axis=1)
print(pred, labels[pred[0]])


# In[ ]:


#### model = load_model("")


# In[ ]:


# test_datagen = ImageDataGenerator(rescale = 1./255)


# In[ ]:


# test_generator = test_datagen.flow_from_directory(
#     directory="X:/train-scene classification/data/test/",
#     target_size=(img_width, img_height),
#     color_mode="rgb",
#     batch_size=1,
#     class_mode=None,
#     shuffle=False,
#     seed=42
# )


# In[ ]:


# test_generator.reset()
# pred=model_final.predict_generator(test_generator,verbose=1)


# In[ ]:


# import numpy as np
# predicted_class_indices=np.argmax(pred,axis=1)


# In[ ]:


# print(predicted_class_indices[:20])
# print(predicted_class_indices.shape)
# print(pred.shape)


# In[48]:


import pandas as pd
df = pd.read_csv(os.path.join("X:/train-scene classification", 'test_images.csv'))
print(df.head())
test_list = df.image_name.values
print(test_list[:20])


# In[49]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
join_l = []
ans = []
test_data_dir_2 = "X:/train-scene classification/data/test/final_test"
for i, im in enumerate(test_list):
    img = cv2.imread(os.path.join(test_data_dir_2, im))
    if i == 1:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
    img = cv2.resize(img, (img_width, img_height)) 
    img = np.reshape(img, (1, img_width, img_height, 3))
    pred = np.argmax(model_final.predict(img), axis=1)
    ans.append((im, pred))

# x_test = np.vstack(join_l)
# print(x_test.shape)


# In[ ]:


pred = model_final.predict(x_test)


# In[ ]:


print(pred[:10])


# In[ ]:


preds = np.argmax(pred, axis=1)


# In[ ]:


print(preds)


# In[50]:


print(ans[0:10])


# In[51]:


print(ans[0][1].shape)


# In[52]:


li = []
for i in ans:
    pa = i[0]
    val= i[1][0]
    li.append([pa, val])
print(li[:10])


# In[53]:


res_df = pd.DataFrame(li, columns=["image_name", "label"])
res_df.head()


# In[54]:


res_df.to_csv('submission_vgg19_1.csv', index = False)

