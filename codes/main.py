'''
Grayscale Conversion
'''
a=imread(‘image.jpg’);
b=rgb2gray(a);
imshow(b);

'''
Image Enhancement
'''
I=imread('image.jpg');
subplot(2,2,1),imshow(I);
title('ROPStageIII Original');
greenI=I(:,:,2);
subplot(1,1,1);
imshow(greenI);
title(' ROPStageIII  Green channel');
clahed = adapthisteq(greenI);
subplot(1,1,1),imshow(clahed);

'''
Image Feature Extraction
'''
a = imread('image');
dim = ndims(a);
if(dim == 3)

  #convert image into greyscal
  a = rgb2gray(a);
end

#Extracting Blood Vessels
Threshold = 10;
bloodVessels = VesselExtract(a, Threshold);

#Display Blood Vessels image
figure;subplot(121);imshow(a);title('Input Image');
subplot(122);imshow(bloodVessels);title('Extracted Blood Vessels');
function bloodVessels = VesselExtract(a, threshold)

#Kirsch's Templates
h1=[5 -3 -3;     5  0 -3;     5 -3 -3]/15;
h2=[-3 -3 5;     -3  0 5;     -3 -3 5]/15;
h3=[-3 -3 -3;    5  0 -3;      5  5 -3]/15;
h4=[-3  5  5;     -3  0  5;    -3 -3 -3]/15;
h5=[-3 -3 -3;    -3  0 -3;     5  5  5]/15;
h6=[ 5  5  5;    -3  0 -3;    -3 -3 -3]/15;
h7=[-3 -3 -3;    -3  0  5;    -3  5  5]/15;
h8=[ 5  5 -3;     5  0 -3;    -3 -3 -3]/15;

#Spatial Filtering by Kirsch's Templates
t1=filter2(h1,a); t2=filter2(h2,a);t3=filter2(h3,a);
t4=filter2(h4,a);t5=filter2(h5,a);t6=filter2(h6,a);
t7=filter2(h7,a);t8=filter2(h8,a);s=size(a);
bloodVessels=zeros(s(1),s(2));
temp=zeros(1,8);

for i=1:s(1)
    for j=1:s(2)
        temp(1)=t1(i,j);temp(2)=t2(i,j);temp(3)=t3(i,j);temp(4)=t4(i,j);
        temp(5)=t5(i,j);temp(6)=t6(i,j);temp(7)=t7(i,j);temp(8)=t8(i,j);
        if(max(temp)>threshold)
            bloodVessels(i,j)=max(temp);
        end
    end
end

'''
Input Image
'''
# initial data directory
data_dir_nonaug = 'dataRetinopathy'

# labeling and resizing all of data that'll be used
# 0 = ROPStage2 and 1 = ROPStage3
data_nonaug = []
categories = [' ROPStage2', ' ROPStage3']
img_size = 224


for category in categories:
    path = os.path.join(data_dir_nonaug, category1)
    class_num = categories.index(category1);for img in os.listdir(path):
        try:
          img_array = cv2.imread(os.path.join(path,img));
          new_array = cv2.resize(img_array, (img_size, img_size));
          data_nonaug.append([new_array, class_num]); except Exception as e:
            pass;
            print('\n Jumlah image data sebelum augmentasi: {0}'.format(len(data_nonaug)))

a_nonaug = []
b_nonaug = []

if_for features, label in data_nonaug:
    a_nonaug.append(features)
    b_nonaug.append(label)

b_nonaug = np.array(b_nonaug)

'''
Data Argumentation
'''
#os should be imported from tensorflow.keras.utils, along with img_to_array and array_to_img
import load_img from keras.preprocessing.image from tensorflow.keras.utils ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=[0.85, 1.15],
    horizontal_flip = True,
    fill_mode ='nearest'
  )

# ROPStage2 augmentation
Path_ROPStage2 = os.path.join(data_dir, 'ROPStage2')
for image in os.listdir(path_ ROPStage2):
  image = load_img('dataRetinopathy/ROPStage2/0').format(img)
    x = img_to_array(image)
    x = ((1,) + x.shape + x.reshape)

  For batch in datagen.flow(x, batch_size=1, save_to_dir='dataRetinopathyaugmented/ROPStage2', save_prefix='ROPStage2', save_format='jpg'), set i = 0 as follows:
        If i is greater than 26, then i += 1; break

# ROPStage3 augmentation
Path_ROPStage3 = os.path.join(data_dir, 'ROPStage3')
for image in os.listdir(path_ ROPStage3)
    image is same to load_img('dataRETINA/ ROPStage3/0').format(img)
    x = img_to_array(image)
    x = ((1,) + x.shape + x.reshape)


    When i = 0
    for a batch in datagen.flow(x, batch_size=1,
                                save_to_dir='dataRetinopathyaugmented/ROPStage3',
                                save_prefix='ROPStage3', and save_format='jpg'),
                                the following occurs:

                                If i is more than 20, then i += 1:
                                break
                                
'''
Image Resizing
'''
new_array = cv2.resize(img_array, (img_size, img_size)) plt; img_size = 224.imshow(new_array)

'n img size =', 'new_array.shape', plt.show() print img size =  (224, 224, 3)


'''
Image Labelling
'''
# labeling and resizing all of data that'll be used.
# 0 = ROPStage2 and 1 = ROPStage3.
data = []
categories = [' ROPStage2', ' ROPStage3']
img_size = 224

os.path.join(data_dir, category) for category in categories
   classes.index(category) = class_num
   os.listdir(path), for img:
   try:
   img_array = os.path.join(path,img),cv2.imread
   new_array is equal to data.cv2.resize(img_array, (img_size, img_size)).[New_Array, class_num]
   except As an exception:
   pass
   print('\n Number of image data that will be used: 0.format(len(data)))

# Set Aug X and Y.
X_aug = []
y_aug = []

Label for features is added using the formulas X_aug.append(label) and Y_aug.append(label).

= np.array(y_aug)

# Data visualization of augmentation before and after.
A_nonaug, B_nonaug, C_aug, and D_aug are all equal to 0.

If y_nonaug[i] == 0 for i in range(len(y_nonaug)), then a_nonaug += 1; otherwise, b_nonaug += 1
If y_aug[i] == 0 for i in range(len(y_aug)), then c_aug += 1; otherwise, d_aug += 1

data_nonaug = np.array([a_nonaug, b_nonaug])
data_aug = np.array([c_aug, d_aug])
dataplot_aug = [data_nonaug, data_aug]
jumlah_ROPStage3_aug = [a_nonaug, c_aug]
jumlah_ROPStage2_aug = [b_nonaug, d_aug]
split_aug = ['SEBELUM', 'SETELAH']

for i in range(len(dataplot_aug)):
    positions = np.arange(2)
    plt.bar(positions, dataplot_aug[i], 0.8)
    plt.xticks(positions + 0.05, ('ROPStage 3', 'ROPStage2'))
    plt.title('Distribusi Data ROPStage3 dan Ro Pada Dataset {} Augmentasi'.format(split_aug[i]))
    print('---Dataset {0} Augmentasi---'.format(split[i]))
    plt.show()
    print('\n Jumlah data ROPStage3 pada dataset {0} augmentasi: {1}'.format(split_aug[i], jumlah_ROPStage3_aug[i]))
    print('\n Jumlah data ROPStage2 pada dataset {0} augmentasi: {1}'.format(split_aug[i], jumlah_ ROPStage2_aug[i]))
    print('\n')


'''
Data Splitting
'''
train_ratio = 0.80

from sklearn.model_selection import train_test_split

# determining the ratio of train-val-test data
train-val-test data using val_ratio = 0.10 and test_ratio = 0.10.

#split data into:
Train_test_split = X_train, X_test, Y_train, and Y_test(X, y, train_ratio 1, test_size 1, random_state 1)
X_val, X_test, y_val, and y_test are all equal to train_test_split(X_test, y_test, test_size=(test_ratio + val_ratio), random_state=1)

# Visualization of data distribution.
a = 0; b = 0; c = 0; d = 0; e = 0; f = 0

for i in range(len(y_train)):
  if y_train[i] == 0:
    a += 1
  else:
    b += 1

for i in range(len(y_val)):
  if y_val[i] == 0:
    c += 1
  else:
    d += 1

data1 = np.array([a, b])
data2 = np.array([c, d])
data3 = np.array([e, f])
dataplot = [data1, data2, data3]

jumlah_ROPStage3 = [a, c, e]
jumlah_ROPStage2 = [b, d, f]

split = ['Training', 'Validation', 'Test']

for i in range(len(dataplot)):
    positions = np.arange(2)
    plt.bar(positions, dataplot[i], 0.8)
    plt.xticks(positions + 0.05, ('ROPStage3', 'ROPStage2'))
    plt.title('Distribusi Data ROPStage3 dan ROPStage2 Pada {} Dataset'.format(split[i]))
    print('---{0} Dataset---'.format(split[i]))
    plt.show()
    print('\n Jumlah data ROPStage3 pada {0} dataset: {1}'.format(split[i], jumlah_ROPStage3 [i]))
    print('\n Jumlah data ROPStage2 pada {0} dataset: {1}'.format(split[i], jumlah_ROPStage2 [i]))
    print('\n')

'''
Model Training
'''
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam

monitor_val_acc = EarlyStopping(monitor = 'val_accuracy', patience = 3);

# callbacks for model
save_best_only = True,
model_checkpoint = ModelCheckpoint('best_ROPStage3_model.hdf5')

# optimizer.
opt = Adam(learning_rate=2*(10**(-5)))

Model = Sequential();

#Model blocks
Model Block 1 is.Model: add(Conv2D(64, (3, 3), activation = "relu", input_shape = X.shape[1:], padding = "same").Addition: Conv2D(64, (3,3), activation = "relu," padding = "same"
MaxPooling2D (pool_size = (2, 2), strides = 2, padding = "same") model.add
Model Block 2 is.Addition: Conv2D(128, (3,3), activation = "relu," padding = "same"
Conv2D(128, (3,3), activation = "relu," padding = "same"); model.add
model.Pool_size = (2,2), Strides = 2, Padding = "Same," MaxPooling2D, add
Model Block 3 is.Model: add(Conv2D(256, (3,3), activation = "relu," padding = "same").Model: add(Conv2D(256, (3,3), activation = "relu," padding = "same").Model: add(Conv2D(256, (3,3), activation = "relu," padding = "same").Model: add(Conv2D(256, (3,3), activation = "relu," padding = "same").Addition: MaxPooling2D (pool_size = (2,2), strides = 2)
Block 4 model is #.Model: add(Conv2D(512, (3,3), activation = "relu," padding = "same").Addition: Conv2D(512, (3,3), activation = "relu," padding = "same"
Conv2D(512, (3,3), activation = "relu," padding = "same"); model.add
model.Addition: Conv2D(512, (3,3), activation = "relu," padding = "same"
model.Pool_size = (2,2), Strides = 2, Padding = "Same," MaxPooling2D, add
Model Block 5 is.model = add(Conv2D(512, 3, 3, activation = "relu", padding = "same")).model = add(Conv2D(512, 3, 3, activation = "relu", padding = "same")).model = add(Conv2D(512, 3, 3, activation = "relu", padding = "same")).model = add(Conv2D(512, 3, 3, activation = "relu", padding = "same")).Pool_size = (2,2), Strides = 2, Padding = "Same," MaxPooling2D, add
Model with Fully Connected Layers.add(Flatten()) model.Dense(4096, activation = "relu") model added.add(Dropout(0.5)) model.Dense(4096, activation = "relu") model added.add(Dropout(0.5)) model.Dense(1000, activation = "relu") model added.Dense(2, activation = "softmax") added
Compiling the model with the following parameters: optimizer = opt, loss = "sparse_categorical_crossentropy," metrics = ["accuracy"]


'''
Model Training
'''
History = model.fit(X_train, Y_train, epochs = 20, batch_size = 32, validation_data=(X_val, Y_val), callbacks = [monitor_val_acc, model_checkpoint])

'''
Model Evaluation
'''
sns import seaborn
import confusion_matrix from sklearn.metrics into matplotlib.pyplot as a plt