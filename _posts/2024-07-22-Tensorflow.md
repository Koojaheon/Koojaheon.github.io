---
layout: single
title: "첫 번째 텐서플로우(Tensorflow) 스터디입니다."
---

python
!pip install tensorflow
```

    Defaulting to user installation because normal site-packages is not writeable
    Requirement already satisfied: tensorflow in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (2.11.0)
    Requirement already satisfied: opt-einsum>=2.3.2 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from tensorflow) (3.3.0)
    Requirement already satisfied: google-pasta>=0.1.1 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from tensorflow) (0.2.0)
    Requirement already satisfied: grpcio<2.0,>=1.24.3 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from tensorflow) (1.62.2)
    Requirement already satisfied: tensorflow-estimator<2.12,>=2.11.0 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from tensorflow) (2.11.0)
    Requirement already satisfied: libclang>=13.0.0 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from tensorflow) (18.1.1)
    Requirement already satisfied: setuptools in /usr/local/miniconda3/lib/python3.7/site-packages (from tensorflow) (58.4.0)
    Requirement already satisfied: gast<=0.4.0,>=0.2.1 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from tensorflow) (0.4.0)
    Requirement already satisfied: protobuf<3.20,>=3.9.2 in /usr/local/miniconda3/lib/python3.7/site-packages (from tensorflow) (3.19.1)
    Requirement already satisfied: absl-py>=1.0.0 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from tensorflow) (2.1.0)
    Requirement already satisfied: tensorboard<2.12,>=2.11 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from tensorflow) (2.11.2)
    Requirement already satisfied: flatbuffers>=2.0 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from tensorflow) (24.3.25)
    Requirement already satisfied: packaging in /usr/local/miniconda3/lib/python3.7/site-packages (from tensorflow) (21.0)
    Requirement already satisfied: wrapt>=1.11.0 in /usr/local/miniconda3/lib/python3.7/site-packages (from tensorflow) (1.13.3)
    Requirement already satisfied: astunparse>=1.6.0 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from tensorflow) (1.6.3)
    Requirement already satisfied: numpy>=1.20 in /usr/local/miniconda3/lib/python3.7/site-packages (from tensorflow) (1.21.3)
    Requirement already satisfied: keras<2.12,>=2.11.0 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from tensorflow) (2.11.0)
    Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/miniconda3/lib/python3.7/site-packages (from tensorflow) (3.10.0.2)
    Requirement already satisfied: h5py>=2.9.0 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from tensorflow) (3.8.0)
    Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from tensorflow) (0.34.0)
    Requirement already satisfied: six>=1.12.0 in /usr/local/miniconda3/lib/python3.7/site-packages (from tensorflow) (1.16.0)
    Requirement already satisfied: termcolor>=1.1.0 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from tensorflow) (2.3.0)
    Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/miniconda3/lib/python3.7/site-packages (from astunparse>=1.6.0->tensorflow) (0.37.0)
    Requirement already satisfied: google-auth<3,>=1.6.3 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from tensorboard<2.12,>=2.11->tensorflow) (2.31.0)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from tensorboard<2.12,>=2.11->tensorflow) (1.8.1)
    Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from tensorboard<2.12,>=2.11->tensorflow) (0.6.1)
    Requirement already satisfied: werkzeug>=1.0.1 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from tensorboard<2.12,>=2.11->tensorflow) (2.2.3)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from tensorboard<2.12,>=2.11->tensorflow) (0.4.6)
    Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/miniconda3/lib/python3.7/site-packages (from tensorboard<2.12,>=2.11->tensorflow) (2.26.0)
    Requirement already satisfied: markdown>=2.6.8 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from tensorboard<2.12,>=2.11->tensorflow) (3.4.4)
    Requirement already satisfied: pyparsing>=2.0.2 in /usr/local/miniconda3/lib/python3.7/site-packages (from packaging->tensorflow) (3.0.4)
    Requirement already satisfied: cachetools<6.0,>=2.0.0 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.12,>=2.11->tensorflow) (5.3.3)
    Requirement already satisfied: rsa<5,>=3.1.4 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.12,>=2.11->tensorflow) (4.9)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from google-auth<3,>=1.6.3->tensorboard<2.12,>=2.11->tensorflow) (0.3.0)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.12,>=2.11->tensorflow) (2.0.0)
    Requirement already satisfied: importlib-metadata>=4.4 in /usr/local/miniconda3/lib/python3.7/site-packages (from markdown>=2.6.8->tensorboard<2.12,>=2.11->tensorflow) (4.8.1)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/miniconda3/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard<2.12,>=2.11->tensorflow) (3.1)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/miniconda3/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard<2.12,>=2.11->tensorflow) (1.26.7)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/miniconda3/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard<2.12,>=2.11->tensorflow) (2.0.0)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/miniconda3/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard<2.12,>=2.11->tensorflow) (2024.2.2)
    Requirement already satisfied: MarkupSafe>=2.1.1 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from werkzeug>=1.0.1->tensorboard<2.12,>=2.11->tensorflow) (2.1.5)
    Requirement already satisfied: zipp>=0.5 in /usr/local/miniconda3/lib/python3.7/site-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.12,>=2.11->tensorflow) (3.6.0)
    Requirement already satisfied: pyasn1<0.6.0,>=0.4.6 in /home/M3LABV/jhkoo/.local/lib/python3.7/site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.12,>=2.11->tensorflow) (0.5.1)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/miniconda3/lib/python3.7/site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.12,>=2.11->tensorflow) (3.2.2)



```python
  import tensorflow as tf
print(tf.__version__)
```

    2.11.0



```python
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD, Adam
```


```python
import numpy as np

try:

    loaded_data = np.loadtxt('/home/M3LABV/jhkoo/CSV/diabetes.csv', delimiter=',')
    
    # 모든 행(:)과 첫 번째 열부터 마지막 열 직전까지 (0:-1)의 데이터를 선택합니다.
    # 이는 모든 특성 변수를 포함하며, 타겟 변수는 제외됩니다.
    x_data = loaded_data[ :, 0:-1]
    
    # 모든 행(:)과 마지막 열[-1] 의 데이터를 선택합니다.
    # 이는 타겟 변수만을 포함합니다.
    # [-1]을 사용하여 2차원 배열 구조를 유지합니다.
    t_data = loaded_data[ :, [-1]]

    print("x_data.shape = ", x_data.shape)
    print("t_data.shape = ", t_data.shape)

except Exception as err:

  print(str(err))
```

    x_data.shape =  (759, 8)
    t_data.shape =  (759, 1)



```python
model = Sequential()

model.add(Dense(t_data.shape[1],
               input_shape=(x_data.shape[1],), activation = 'sigmoid'))
```


```python
model.compile(optimizer=SGD(learning_rate = 0.01), loss = 'binary_crossentropy', metrics =['accuracy'])

model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_1 (Dense)             (None, 1)                 9         
                                                                     
    =================================================================
    Total params: 9
    Trainable params: 9
    Non-trainable params: 0
    _________________________________________________________________



```python
hist = model.fit(x_data, t_data, epochs = 500, validation_split = 0.2, verbose = 2)
```

    Epoch 1/500
    19/19 - 0s - loss: 0.7976 - accuracy: 0.4036 - val_loss: 0.8007 - val_accuracy: 0.4145 - 407ms/epoch - 21ms/step
    Epoch 2/500
    19/19 - 0s - loss: 0.7614 - accuracy: 0.4283 - val_loss: 0.7667 - val_accuracy: 0.4342 - 36ms/epoch - 2ms/step
    Epoch 3/500
    19/19 - 0s - loss: 0.7314 - accuracy: 0.4629 - val_loss: 0.7386 - val_accuracy: 0.4737 - 35ms/epoch - 2ms/step
    Epoch 4/500
    19/19 - 0s - loss: 0.7066 - accuracy: 0.5008 - val_loss: 0.7152 - val_accuracy: 0.5132 - 36ms/epoch - 2ms/step
    Epoch 5/500
    19/19 - 0s - loss: 0.6861 - accuracy: 0.5437 - val_loss: 0.6961 - val_accuracy: 0.5329 - 33ms/epoch - 2ms/step
    Epoch 6/500
    19/19 - 0s - loss: 0.6692 - accuracy: 0.5717 - val_loss: 0.6802 - val_accuracy: 0.5395 - 35ms/epoch - 2ms/step
    Epoch 7/500
    19/19 - 0s - loss: 0.6553 - accuracy: 0.5898 - val_loss: 0.6670 - val_accuracy: 0.5724 - 35ms/epoch - 2ms/step
    Epoch 8/500
    19/19 - 0s - loss: 0.6436 - accuracy: 0.6211 - val_loss: 0.6560 - val_accuracy: 0.6053 - 36ms/epoch - 2ms/step
    Epoch 9/500
    19/19 - 0s - loss: 0.6340 - accuracy: 0.6524 - val_loss: 0.6468 - val_accuracy: 0.6118 - 35ms/epoch - 2ms/step
    Epoch 10/500
    19/19 - 0s - loss: 0.6260 - accuracy: 0.6606 - val_loss: 0.6391 - val_accuracy: 0.6447 - 34ms/epoch - 2ms/step
    Epoch 11/500
    19/19 - 0s - loss: 0.6192 - accuracy: 0.6755 - val_loss: 0.6326 - val_accuracy: 0.6447 - 35ms/epoch - 2ms/step
    Epoch 12/500
    19/19 - 0s - loss: 0.6134 - accuracy: 0.6837 - val_loss: 0.6271 - val_accuracy: 0.6645 - 35ms/epoch - 2ms/step
    Epoch 13/500
    19/19 - 0s - loss: 0.6085 - accuracy: 0.6903 - val_loss: 0.6224 - val_accuracy: 0.6579 - 35ms/epoch - 2ms/step
    Epoch 14/500
    19/19 - 0s - loss: 0.6043 - accuracy: 0.6936 - val_loss: 0.6183 - val_accuracy: 0.6645 - 35ms/epoch - 2ms/step
    Epoch 15/500
    19/19 - 0s - loss: 0.6007 - accuracy: 0.6985 - val_loss: 0.6147 - val_accuracy: 0.6908 - 36ms/epoch - 2ms/step
    Epoch 16/500
    19/19 - 0s - loss: 0.5975 - accuracy: 0.7035 - val_loss: 0.6117 - val_accuracy: 0.6974 - 33ms/epoch - 2ms/step
    Epoch 17/500
    19/19 - 0s - loss: 0.5948 - accuracy: 0.7002 - val_loss: 0.6089 - val_accuracy: 0.6974 - 35ms/epoch - 2ms/step
    Epoch 18/500
    19/19 - 0s - loss: 0.5923 - accuracy: 0.7035 - val_loss: 0.6065 - val_accuracy: 0.7105 - 34ms/epoch - 2ms/step
    Epoch 19/500
    19/19 - 0s - loss: 0.5901 - accuracy: 0.7100 - val_loss: 0.6043 - val_accuracy: 0.7171 - 33ms/epoch - 2ms/step
    Epoch 20/500
    19/19 - 0s - loss: 0.5881 - accuracy: 0.7117 - val_loss: 0.6024 - val_accuracy: 0.7105 - 35ms/epoch - 2ms/step
    Epoch 21/500
    19/19 - 0s - loss: 0.5864 - accuracy: 0.7100 - val_loss: 0.6006 - val_accuracy: 0.6908 - 34ms/epoch - 2ms/step
    Epoch 22/500
    19/19 - 0s - loss: 0.5848 - accuracy: 0.7100 - val_loss: 0.5990 - val_accuracy: 0.6974 - 34ms/epoch - 2ms/step
    Epoch 23/500
    19/19 - 0s - loss: 0.5832 - accuracy: 0.7100 - val_loss: 0.5974 - val_accuracy: 0.6776 - 33ms/epoch - 2ms/step
    Epoch 24/500
    19/19 - 0s - loss: 0.5819 - accuracy: 0.7100 - val_loss: 0.5960 - val_accuracy: 0.6776 - 33ms/epoch - 2ms/step
    Epoch 25/500
    19/19 - 0s - loss: 0.5805 - accuracy: 0.7084 - val_loss: 0.5947 - val_accuracy: 0.6776 - 33ms/epoch - 2ms/step
    Epoch 26/500
    19/19 - 0s - loss: 0.5793 - accuracy: 0.7051 - val_loss: 0.5934 - val_accuracy: 0.6776 - 34ms/epoch - 2ms/step
    Epoch 27/500
    19/19 - 0s - loss: 0.5781 - accuracy: 0.7051 - val_loss: 0.5922 - val_accuracy: 0.6776 - 35ms/epoch - 2ms/step
    Epoch 28/500
    19/19 - 0s - loss: 0.5770 - accuracy: 0.7051 - val_loss: 0.5911 - val_accuracy: 0.6711 - 35ms/epoch - 2ms/step
    Epoch 29/500
    19/19 - 0s - loss: 0.5759 - accuracy: 0.7051 - val_loss: 0.5900 - val_accuracy: 0.6711 - 34ms/epoch - 2ms/step
    Epoch 30/500
    19/19 - 0s - loss: 0.5749 - accuracy: 0.7035 - val_loss: 0.5889 - val_accuracy: 0.6711 - 34ms/epoch - 2ms/step
    Epoch 31/500
    19/19 - 0s - loss: 0.5739 - accuracy: 0.7018 - val_loss: 0.5879 - val_accuracy: 0.6711 - 34ms/epoch - 2ms/step
    Epoch 32/500
    19/19 - 0s - loss: 0.5729 - accuracy: 0.7035 - val_loss: 0.5869 - val_accuracy: 0.6776 - 33ms/epoch - 2ms/step
    Epoch 33/500
    19/19 - 0s - loss: 0.5720 - accuracy: 0.7002 - val_loss: 0.5859 - val_accuracy: 0.6776 - 34ms/epoch - 2ms/step
    Epoch 34/500
    19/19 - 0s - loss: 0.5710 - accuracy: 0.7035 - val_loss: 0.5850 - val_accuracy: 0.6776 - 33ms/epoch - 2ms/step
    Epoch 35/500
    19/19 - 0s - loss: 0.5701 - accuracy: 0.7035 - val_loss: 0.5840 - val_accuracy: 0.6776 - 34ms/epoch - 2ms/step
    Epoch 36/500
    19/19 - 0s - loss: 0.5693 - accuracy: 0.7035 - val_loss: 0.5831 - val_accuracy: 0.6776 - 32ms/epoch - 2ms/step
    Epoch 37/500
    19/19 - 0s - loss: 0.5684 - accuracy: 0.7035 - val_loss: 0.5822 - val_accuracy: 0.6842 - 35ms/epoch - 2ms/step
    Epoch 38/500
    19/19 - 0s - loss: 0.5676 - accuracy: 0.7035 - val_loss: 0.5814 - val_accuracy: 0.6842 - 35ms/epoch - 2ms/step
    Epoch 39/500
    19/19 - 0s - loss: 0.5667 - accuracy: 0.7035 - val_loss: 0.5805 - val_accuracy: 0.6842 - 34ms/epoch - 2ms/step
    Epoch 40/500
    19/19 - 0s - loss: 0.5660 - accuracy: 0.7035 - val_loss: 0.5797 - val_accuracy: 0.6842 - 34ms/epoch - 2ms/step
    Epoch 41/500
    19/19 - 0s - loss: 0.5651 - accuracy: 0.7035 - val_loss: 0.5788 - val_accuracy: 0.6842 - 35ms/epoch - 2ms/step
    Epoch 42/500
    19/19 - 0s - loss: 0.5644 - accuracy: 0.7018 - val_loss: 0.5780 - val_accuracy: 0.6842 - 35ms/epoch - 2ms/step
    Epoch 43/500
    19/19 - 0s - loss: 0.5636 - accuracy: 0.7018 - val_loss: 0.5772 - val_accuracy: 0.6842 - 35ms/epoch - 2ms/step
    Epoch 44/500
    19/19 - 0s - loss: 0.5628 - accuracy: 0.7018 - val_loss: 0.5764 - val_accuracy: 0.6908 - 36ms/epoch - 2ms/step
    Epoch 45/500
    19/19 - 0s - loss: 0.5620 - accuracy: 0.7035 - val_loss: 0.5756 - val_accuracy: 0.6908 - 34ms/epoch - 2ms/step
    Epoch 46/500
    19/19 - 0s - loss: 0.5613 - accuracy: 0.7035 - val_loss: 0.5748 - val_accuracy: 0.6974 - 34ms/epoch - 2ms/step
    Epoch 47/500
    19/19 - 0s - loss: 0.5605 - accuracy: 0.7051 - val_loss: 0.5740 - val_accuracy: 0.7039 - 36ms/epoch - 2ms/step
    Epoch 48/500
    19/19 - 0s - loss: 0.5598 - accuracy: 0.7100 - val_loss: 0.5733 - val_accuracy: 0.7039 - 34ms/epoch - 2ms/step
    Epoch 49/500
    19/19 - 0s - loss: 0.5591 - accuracy: 0.7100 - val_loss: 0.5725 - val_accuracy: 0.7039 - 34ms/epoch - 2ms/step
    Epoch 50/500
    19/19 - 0s - loss: 0.5584 - accuracy: 0.7117 - val_loss: 0.5718 - val_accuracy: 0.7039 - 34ms/epoch - 2ms/step
    Epoch 51/500
    19/19 - 0s - loss: 0.5576 - accuracy: 0.7100 - val_loss: 0.5711 - val_accuracy: 0.7039 - 34ms/epoch - 2ms/step
    Epoch 52/500
    19/19 - 0s - loss: 0.5569 - accuracy: 0.7150 - val_loss: 0.5703 - val_accuracy: 0.7039 - 34ms/epoch - 2ms/step
    Epoch 53/500
    19/19 - 0s - loss: 0.5563 - accuracy: 0.7199 - val_loss: 0.5696 - val_accuracy: 0.7039 - 36ms/epoch - 2ms/step
    Epoch 54/500
    19/19 - 0s - loss: 0.5556 - accuracy: 0.7216 - val_loss: 0.5689 - val_accuracy: 0.7039 - 35ms/epoch - 2ms/step
    Epoch 55/500
    19/19 - 0s - loss: 0.5549 - accuracy: 0.7216 - val_loss: 0.5682 - val_accuracy: 0.7039 - 34ms/epoch - 2ms/step
    Epoch 56/500
    19/19 - 0s - loss: 0.5542 - accuracy: 0.7232 - val_loss: 0.5675 - val_accuracy: 0.7039 - 38ms/epoch - 2ms/step
    Epoch 57/500
    19/19 - 0s - loss: 0.5536 - accuracy: 0.7232 - val_loss: 0.5668 - val_accuracy: 0.7039 - 37ms/epoch - 2ms/step
    Epoch 58/500
    19/19 - 0s - loss: 0.5529 - accuracy: 0.7232 - val_loss: 0.5661 - val_accuracy: 0.7039 - 37ms/epoch - 2ms/step
    Epoch 59/500
    19/19 - 0s - loss: 0.5523 - accuracy: 0.7232 - val_loss: 0.5655 - val_accuracy: 0.7039 - 34ms/epoch - 2ms/step
    Epoch 60/500
    19/19 - 0s - loss: 0.5517 - accuracy: 0.7265 - val_loss: 0.5648 - val_accuracy: 0.7039 - 34ms/epoch - 2ms/step
    Epoch 61/500
    19/19 - 0s - loss: 0.5510 - accuracy: 0.7298 - val_loss: 0.5641 - val_accuracy: 0.6974 - 35ms/epoch - 2ms/step
    Epoch 62/500
    19/19 - 0s - loss: 0.5504 - accuracy: 0.7298 - val_loss: 0.5635 - val_accuracy: 0.6974 - 34ms/epoch - 2ms/step
    Epoch 63/500
    19/19 - 0s - loss: 0.5498 - accuracy: 0.7298 - val_loss: 0.5628 - val_accuracy: 0.6974 - 35ms/epoch - 2ms/step
    Epoch 64/500
    19/19 - 0s - loss: 0.5492 - accuracy: 0.7315 - val_loss: 0.5622 - val_accuracy: 0.6974 - 33ms/epoch - 2ms/step
    Epoch 65/500
    19/19 - 0s - loss: 0.5486 - accuracy: 0.7331 - val_loss: 0.5616 - val_accuracy: 0.6974 - 35ms/epoch - 2ms/step
    Epoch 66/500
    19/19 - 0s - loss: 0.5479 - accuracy: 0.7331 - val_loss: 0.5609 - val_accuracy: 0.6974 - 34ms/epoch - 2ms/step
    Epoch 67/500
    19/19 - 0s - loss: 0.5473 - accuracy: 0.7348 - val_loss: 0.5603 - val_accuracy: 0.6974 - 34ms/epoch - 2ms/step
    Epoch 68/500
    19/19 - 0s - loss: 0.5468 - accuracy: 0.7364 - val_loss: 0.5597 - val_accuracy: 0.6974 - 34ms/epoch - 2ms/step
    Epoch 69/500
    19/19 - 0s - loss: 0.5462 - accuracy: 0.7348 - val_loss: 0.5591 - val_accuracy: 0.6974 - 34ms/epoch - 2ms/step
    Epoch 70/500
    19/19 - 0s - loss: 0.5456 - accuracy: 0.7348 - val_loss: 0.5585 - val_accuracy: 0.6974 - 35ms/epoch - 2ms/step
    Epoch 71/500
    19/19 - 0s - loss: 0.5450 - accuracy: 0.7364 - val_loss: 0.5579 - val_accuracy: 0.6974 - 35ms/epoch - 2ms/step
    Epoch 72/500
    19/19 - 0s - loss: 0.5445 - accuracy: 0.7331 - val_loss: 0.5573 - val_accuracy: 0.6974 - 35ms/epoch - 2ms/step
    Epoch 73/500
    19/19 - 0s - loss: 0.5439 - accuracy: 0.7348 - val_loss: 0.5567 - val_accuracy: 0.6974 - 35ms/epoch - 2ms/step
    Epoch 74/500
    19/19 - 0s - loss: 0.5434 - accuracy: 0.7364 - val_loss: 0.5561 - val_accuracy: 0.6974 - 34ms/epoch - 2ms/step
    Epoch 75/500
    19/19 - 0s - loss: 0.5428 - accuracy: 0.7364 - val_loss: 0.5556 - val_accuracy: 0.6974 - 36ms/epoch - 2ms/step
    Epoch 76/500
    19/19 - 0s - loss: 0.5423 - accuracy: 0.7397 - val_loss: 0.5550 - val_accuracy: 0.6908 - 34ms/epoch - 2ms/step
    Epoch 77/500
    19/19 - 0s - loss: 0.5417 - accuracy: 0.7397 - val_loss: 0.5544 - val_accuracy: 0.6908 - 36ms/epoch - 2ms/step
    Epoch 78/500
    19/19 - 0s - loss: 0.5412 - accuracy: 0.7381 - val_loss: 0.5539 - val_accuracy: 0.6908 - 34ms/epoch - 2ms/step
    Epoch 79/500
    19/19 - 0s - loss: 0.5406 - accuracy: 0.7397 - val_loss: 0.5533 - val_accuracy: 0.6974 - 33ms/epoch - 2ms/step
    Epoch 80/500
    19/19 - 0s - loss: 0.5401 - accuracy: 0.7381 - val_loss: 0.5528 - val_accuracy: 0.6974 - 36ms/epoch - 2ms/step
    Epoch 81/500
    19/19 - 0s - loss: 0.5396 - accuracy: 0.7381 - val_loss: 0.5523 - val_accuracy: 0.6974 - 57ms/epoch - 3ms/step
    Epoch 82/500
    19/19 - 0s - loss: 0.5391 - accuracy: 0.7381 - val_loss: 0.5517 - val_accuracy: 0.6974 - 35ms/epoch - 2ms/step
    Epoch 83/500
    19/19 - 0s - loss: 0.5386 - accuracy: 0.7397 - val_loss: 0.5512 - val_accuracy: 0.6974 - 35ms/epoch - 2ms/step
    Epoch 84/500
    19/19 - 0s - loss: 0.5381 - accuracy: 0.7364 - val_loss: 0.5507 - val_accuracy: 0.6974 - 35ms/epoch - 2ms/step
    Epoch 85/500
    19/19 - 0s - loss: 0.5376 - accuracy: 0.7364 - val_loss: 0.5502 - val_accuracy: 0.6974 - 34ms/epoch - 2ms/step
    Epoch 86/500
    19/19 - 0s - loss: 0.5371 - accuracy: 0.7364 - val_loss: 0.5496 - val_accuracy: 0.6974 - 33ms/epoch - 2ms/step
    Epoch 87/500
    19/19 - 0s - loss: 0.5366 - accuracy: 0.7381 - val_loss: 0.5491 - val_accuracy: 0.6974 - 34ms/epoch - 2ms/step
    Epoch 88/500
    19/19 - 0s - loss: 0.5362 - accuracy: 0.7364 - val_loss: 0.5486 - val_accuracy: 0.6974 - 33ms/epoch - 2ms/step
    Epoch 89/500
    19/19 - 0s - loss: 0.5356 - accuracy: 0.7364 - val_loss: 0.5481 - val_accuracy: 0.6974 - 35ms/epoch - 2ms/step
    Epoch 90/500
    19/19 - 0s - loss: 0.5352 - accuracy: 0.7397 - val_loss: 0.5476 - val_accuracy: 0.6974 - 33ms/epoch - 2ms/step
    Epoch 91/500
    19/19 - 0s - loss: 0.5347 - accuracy: 0.7397 - val_loss: 0.5471 - val_accuracy: 0.6974 - 33ms/epoch - 2ms/step
    Epoch 92/500
    19/19 - 0s - loss: 0.5343 - accuracy: 0.7414 - val_loss: 0.5467 - val_accuracy: 0.6974 - 33ms/epoch - 2ms/step
    Epoch 93/500
    19/19 - 0s - loss: 0.5338 - accuracy: 0.7397 - val_loss: 0.5462 - val_accuracy: 0.6974 - 33ms/epoch - 2ms/step
    Epoch 94/500
    19/19 - 0s - loss: 0.5334 - accuracy: 0.7397 - val_loss: 0.5457 - val_accuracy: 0.6974 - 34ms/epoch - 2ms/step
    Epoch 95/500
    19/19 - 0s - loss: 0.5329 - accuracy: 0.7381 - val_loss: 0.5452 - val_accuracy: 0.6974 - 33ms/epoch - 2ms/step
    Epoch 96/500
    19/19 - 0s - loss: 0.5325 - accuracy: 0.7381 - val_loss: 0.5448 - val_accuracy: 0.6974 - 34ms/epoch - 2ms/step
    Epoch 97/500
    19/19 - 0s - loss: 0.5320 - accuracy: 0.7364 - val_loss: 0.5443 - val_accuracy: 0.6974 - 34ms/epoch - 2ms/step
    Epoch 98/500
    19/19 - 0s - loss: 0.5316 - accuracy: 0.7364 - val_loss: 0.5438 - val_accuracy: 0.7039 - 33ms/epoch - 2ms/step
    Epoch 99/500
    19/19 - 0s - loss: 0.5311 - accuracy: 0.7397 - val_loss: 0.5434 - val_accuracy: 0.7105 - 33ms/epoch - 2ms/step
    Epoch 100/500
    19/19 - 0s - loss: 0.5307 - accuracy: 0.7381 - val_loss: 0.5429 - val_accuracy: 0.7105 - 33ms/epoch - 2ms/step
    Epoch 101/500
    19/19 - 0s - loss: 0.5303 - accuracy: 0.7430 - val_loss: 0.5425 - val_accuracy: 0.7105 - 32ms/epoch - 2ms/step
    Epoch 102/500
    19/19 - 0s - loss: 0.5299 - accuracy: 0.7430 - val_loss: 0.5421 - val_accuracy: 0.7105 - 33ms/epoch - 2ms/step
    Epoch 103/500
    19/19 - 0s - loss: 0.5295 - accuracy: 0.7430 - val_loss: 0.5416 - val_accuracy: 0.7105 - 33ms/epoch - 2ms/step
    Epoch 104/500
    19/19 - 0s - loss: 0.5290 - accuracy: 0.7430 - val_loss: 0.5412 - val_accuracy: 0.7105 - 33ms/epoch - 2ms/step
    Epoch 105/500
    19/19 - 0s - loss: 0.5286 - accuracy: 0.7430 - val_loss: 0.5408 - val_accuracy: 0.7171 - 33ms/epoch - 2ms/step
    Epoch 106/500
    19/19 - 0s - loss: 0.5282 - accuracy: 0.7446 - val_loss: 0.5403 - val_accuracy: 0.7171 - 33ms/epoch - 2ms/step
    Epoch 107/500
    19/19 - 0s - loss: 0.5278 - accuracy: 0.7463 - val_loss: 0.5399 - val_accuracy: 0.7171 - 34ms/epoch - 2ms/step
    Epoch 108/500
    19/19 - 0s - loss: 0.5274 - accuracy: 0.7463 - val_loss: 0.5395 - val_accuracy: 0.7171 - 33ms/epoch - 2ms/step
    Epoch 109/500
    19/19 - 0s - loss: 0.5270 - accuracy: 0.7479 - val_loss: 0.5391 - val_accuracy: 0.7171 - 34ms/epoch - 2ms/step
    Epoch 110/500
    19/19 - 0s - loss: 0.5266 - accuracy: 0.7496 - val_loss: 0.5387 - val_accuracy: 0.7171 - 34ms/epoch - 2ms/step
    Epoch 111/500
    19/19 - 0s - loss: 0.5262 - accuracy: 0.7496 - val_loss: 0.5383 - val_accuracy: 0.7171 - 34ms/epoch - 2ms/step
    Epoch 112/500
    19/19 - 0s - loss: 0.5259 - accuracy: 0.7512 - val_loss: 0.5379 - val_accuracy: 0.7171 - 34ms/epoch - 2ms/step
    Epoch 113/500
    19/19 - 0s - loss: 0.5255 - accuracy: 0.7512 - val_loss: 0.5375 - val_accuracy: 0.7171 - 34ms/epoch - 2ms/step
    Epoch 114/500
    19/19 - 0s - loss: 0.5251 - accuracy: 0.7529 - val_loss: 0.5371 - val_accuracy: 0.7171 - 33ms/epoch - 2ms/step
    Epoch 115/500
    19/19 - 0s - loss: 0.5248 - accuracy: 0.7529 - val_loss: 0.5367 - val_accuracy: 0.7171 - 34ms/epoch - 2ms/step
    Epoch 116/500
    19/19 - 0s - loss: 0.5244 - accuracy: 0.7529 - val_loss: 0.5363 - val_accuracy: 0.7171 - 34ms/epoch - 2ms/step
    Epoch 117/500
    19/19 - 0s - loss: 0.5240 - accuracy: 0.7512 - val_loss: 0.5359 - val_accuracy: 0.7171 - 39ms/epoch - 2ms/step
    Epoch 118/500
    19/19 - 0s - loss: 0.5236 - accuracy: 0.7512 - val_loss: 0.5355 - val_accuracy: 0.7171 - 36ms/epoch - 2ms/step
    Epoch 119/500
    19/19 - 0s - loss: 0.5232 - accuracy: 0.7512 - val_loss: 0.5352 - val_accuracy: 0.7171 - 36ms/epoch - 2ms/step
    Epoch 120/500
    19/19 - 0s - loss: 0.5229 - accuracy: 0.7512 - val_loss: 0.5348 - val_accuracy: 0.7171 - 36ms/epoch - 2ms/step
    Epoch 121/500
    19/19 - 0s - loss: 0.5225 - accuracy: 0.7512 - val_loss: 0.5344 - val_accuracy: 0.7171 - 35ms/epoch - 2ms/step
    Epoch 122/500
    19/19 - 0s - loss: 0.5222 - accuracy: 0.7496 - val_loss: 0.5341 - val_accuracy: 0.7171 - 36ms/epoch - 2ms/step
    Epoch 123/500
    19/19 - 0s - loss: 0.5219 - accuracy: 0.7512 - val_loss: 0.5337 - val_accuracy: 0.7171 - 35ms/epoch - 2ms/step
    Epoch 124/500
    19/19 - 0s - loss: 0.5215 - accuracy: 0.7512 - val_loss: 0.5333 - val_accuracy: 0.7171 - 35ms/epoch - 2ms/step
    Epoch 125/500
    19/19 - 0s - loss: 0.5212 - accuracy: 0.7496 - val_loss: 0.5330 - val_accuracy: 0.7303 - 36ms/epoch - 2ms/step
    Epoch 126/500
    19/19 - 0s - loss: 0.5208 - accuracy: 0.7529 - val_loss: 0.5326 - val_accuracy: 0.7368 - 34ms/epoch - 2ms/step
    Epoch 127/500
    19/19 - 0s - loss: 0.5205 - accuracy: 0.7545 - val_loss: 0.5323 - val_accuracy: 0.7368 - 34ms/epoch - 2ms/step
    Epoch 128/500
    19/19 - 0s - loss: 0.5202 - accuracy: 0.7545 - val_loss: 0.5319 - val_accuracy: 0.7368 - 33ms/epoch - 2ms/step
    Epoch 129/500
    19/19 - 0s - loss: 0.5198 - accuracy: 0.7562 - val_loss: 0.5316 - val_accuracy: 0.7368 - 33ms/epoch - 2ms/step
    Epoch 130/500
    19/19 - 0s - loss: 0.5195 - accuracy: 0.7562 - val_loss: 0.5313 - val_accuracy: 0.7368 - 34ms/epoch - 2ms/step
    Epoch 131/500
    19/19 - 0s - loss: 0.5192 - accuracy: 0.7562 - val_loss: 0.5309 - val_accuracy: 0.7368 - 33ms/epoch - 2ms/step
    Epoch 132/500
    19/19 - 0s - loss: 0.5189 - accuracy: 0.7562 - val_loss: 0.5306 - val_accuracy: 0.7368 - 32ms/epoch - 2ms/step
    Epoch 133/500
    19/19 - 0s - loss: 0.5186 - accuracy: 0.7562 - val_loss: 0.5302 - val_accuracy: 0.7368 - 34ms/epoch - 2ms/step
    Epoch 134/500
    19/19 - 0s - loss: 0.5182 - accuracy: 0.7562 - val_loss: 0.5299 - val_accuracy: 0.7368 - 35ms/epoch - 2ms/step
    Epoch 135/500
    19/19 - 0s - loss: 0.5179 - accuracy: 0.7578 - val_loss: 0.5296 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 136/500
    19/19 - 0s - loss: 0.5176 - accuracy: 0.7578 - val_loss: 0.5293 - val_accuracy: 0.7368 - 35ms/epoch - 2ms/step
    Epoch 137/500
    19/19 - 0s - loss: 0.5173 - accuracy: 0.7578 - val_loss: 0.5290 - val_accuracy: 0.7368 - 34ms/epoch - 2ms/step
    Epoch 138/500
    19/19 - 0s - loss: 0.5170 - accuracy: 0.7578 - val_loss: 0.5286 - val_accuracy: 0.7368 - 32ms/epoch - 2ms/step
    Epoch 139/500
    19/19 - 0s - loss: 0.5167 - accuracy: 0.7578 - val_loss: 0.5283 - val_accuracy: 0.7368 - 33ms/epoch - 2ms/step
    Epoch 140/500
    19/19 - 0s - loss: 0.5164 - accuracy: 0.7578 - val_loss: 0.5280 - val_accuracy: 0.7368 - 34ms/epoch - 2ms/step
    Epoch 141/500
    19/19 - 0s - loss: 0.5161 - accuracy: 0.7595 - val_loss: 0.5277 - val_accuracy: 0.7368 - 33ms/epoch - 2ms/step
    Epoch 142/500
    19/19 - 0s - loss: 0.5158 - accuracy: 0.7595 - val_loss: 0.5274 - val_accuracy: 0.7368 - 34ms/epoch - 2ms/step
    Epoch 143/500
    19/19 - 0s - loss: 0.5156 - accuracy: 0.7595 - val_loss: 0.5271 - val_accuracy: 0.7368 - 33ms/epoch - 2ms/step
    Epoch 144/500
    19/19 - 0s - loss: 0.5152 - accuracy: 0.7595 - val_loss: 0.5268 - val_accuracy: 0.7368 - 33ms/epoch - 2ms/step
    Epoch 145/500
    19/19 - 0s - loss: 0.5149 - accuracy: 0.7611 - val_loss: 0.5265 - val_accuracy: 0.7368 - 33ms/epoch - 2ms/step
    Epoch 146/500
    19/19 - 0s - loss: 0.5146 - accuracy: 0.7595 - val_loss: 0.5262 - val_accuracy: 0.7434 - 37ms/epoch - 2ms/step
    Epoch 147/500
    19/19 - 0s - loss: 0.5144 - accuracy: 0.7611 - val_loss: 0.5259 - val_accuracy: 0.7434 - 37ms/epoch - 2ms/step
    Epoch 148/500
    19/19 - 0s - loss: 0.5141 - accuracy: 0.7595 - val_loss: 0.5256 - val_accuracy: 0.7434 - 38ms/epoch - 2ms/step
    Epoch 149/500
    19/19 - 0s - loss: 0.5138 - accuracy: 0.7595 - val_loss: 0.5253 - val_accuracy: 0.7434 - 36ms/epoch - 2ms/step
    Epoch 150/500
    19/19 - 0s - loss: 0.5135 - accuracy: 0.7595 - val_loss: 0.5250 - val_accuracy: 0.7434 - 36ms/epoch - 2ms/step
    Epoch 151/500
    19/19 - 0s - loss: 0.5133 - accuracy: 0.7595 - val_loss: 0.5247 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 152/500
    19/19 - 0s - loss: 0.5130 - accuracy: 0.7595 - val_loss: 0.5245 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 153/500
    19/19 - 0s - loss: 0.5127 - accuracy: 0.7595 - val_loss: 0.5242 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 154/500
    19/19 - 0s - loss: 0.5124 - accuracy: 0.7595 - val_loss: 0.5239 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 155/500
    19/19 - 0s - loss: 0.5122 - accuracy: 0.7611 - val_loss: 0.5236 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 156/500
    19/19 - 0s - loss: 0.5119 - accuracy: 0.7611 - val_loss: 0.5234 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 157/500
    19/19 - 0s - loss: 0.5117 - accuracy: 0.7611 - val_loss: 0.5231 - val_accuracy: 0.7500 - 34ms/epoch - 2ms/step
    Epoch 158/500
    19/19 - 0s - loss: 0.5114 - accuracy: 0.7611 - val_loss: 0.5228 - val_accuracy: 0.7500 - 33ms/epoch - 2ms/step
    Epoch 159/500
    19/19 - 0s - loss: 0.5112 - accuracy: 0.7611 - val_loss: 0.5226 - val_accuracy: 0.7500 - 37ms/epoch - 2ms/step
    Epoch 160/500
    19/19 - 0s - loss: 0.5109 - accuracy: 0.7611 - val_loss: 0.5223 - val_accuracy: 0.7566 - 36ms/epoch - 2ms/step
    Epoch 161/500
    19/19 - 0s - loss: 0.5106 - accuracy: 0.7611 - val_loss: 0.5220 - val_accuracy: 0.7566 - 36ms/epoch - 2ms/step
    Epoch 162/500
    19/19 - 0s - loss: 0.5104 - accuracy: 0.7611 - val_loss: 0.5218 - val_accuracy: 0.7566 - 36ms/epoch - 2ms/step
    Epoch 163/500
    19/19 - 0s - loss: 0.5102 - accuracy: 0.7611 - val_loss: 0.5215 - val_accuracy: 0.7632 - 37ms/epoch - 2ms/step
    Epoch 164/500
    19/19 - 0s - loss: 0.5099 - accuracy: 0.7611 - val_loss: 0.5213 - val_accuracy: 0.7632 - 36ms/epoch - 2ms/step
    Epoch 165/500
    19/19 - 0s - loss: 0.5097 - accuracy: 0.7611 - val_loss: 0.5210 - val_accuracy: 0.7632 - 36ms/epoch - 2ms/step
    Epoch 166/500
    19/19 - 0s - loss: 0.5094 - accuracy: 0.7628 - val_loss: 0.5208 - val_accuracy: 0.7632 - 36ms/epoch - 2ms/step
    Epoch 167/500
    19/19 - 0s - loss: 0.5092 - accuracy: 0.7628 - val_loss: 0.5205 - val_accuracy: 0.7632 - 36ms/epoch - 2ms/step
    Epoch 168/500
    19/19 - 0s - loss: 0.5090 - accuracy: 0.7628 - val_loss: 0.5203 - val_accuracy: 0.7632 - 34ms/epoch - 2ms/step
    Epoch 169/500
    19/19 - 0s - loss: 0.5087 - accuracy: 0.7628 - val_loss: 0.5200 - val_accuracy: 0.7632 - 34ms/epoch - 2ms/step
    Epoch 170/500
    19/19 - 0s - loss: 0.5085 - accuracy: 0.7628 - val_loss: 0.5198 - val_accuracy: 0.7632 - 34ms/epoch - 2ms/step
    Epoch 171/500
    19/19 - 0s - loss: 0.5083 - accuracy: 0.7628 - val_loss: 0.5195 - val_accuracy: 0.7632 - 34ms/epoch - 2ms/step
    Epoch 172/500
    19/19 - 0s - loss: 0.5080 - accuracy: 0.7611 - val_loss: 0.5193 - val_accuracy: 0.7632 - 36ms/epoch - 2ms/step
    Epoch 173/500
    19/19 - 0s - loss: 0.5079 - accuracy: 0.7611 - val_loss: 0.5191 - val_accuracy: 0.7632 - 36ms/epoch - 2ms/step
    Epoch 174/500
    19/19 - 0s - loss: 0.5076 - accuracy: 0.7611 - val_loss: 0.5188 - val_accuracy: 0.7632 - 36ms/epoch - 2ms/step
    Epoch 175/500
    19/19 - 0s - loss: 0.5074 - accuracy: 0.7595 - val_loss: 0.5186 - val_accuracy: 0.7632 - 36ms/epoch - 2ms/step
    Epoch 176/500
    19/19 - 0s - loss: 0.5072 - accuracy: 0.7611 - val_loss: 0.5184 - val_accuracy: 0.7632 - 36ms/epoch - 2ms/step
    Epoch 177/500
    19/19 - 0s - loss: 0.5069 - accuracy: 0.7611 - val_loss: 0.5181 - val_accuracy: 0.7632 - 35ms/epoch - 2ms/step
    Epoch 178/500
    19/19 - 0s - loss: 0.5067 - accuracy: 0.7595 - val_loss: 0.5179 - val_accuracy: 0.7632 - 35ms/epoch - 2ms/step
    Epoch 179/500
    19/19 - 0s - loss: 0.5065 - accuracy: 0.7611 - val_loss: 0.5177 - val_accuracy: 0.7632 - 35ms/epoch - 2ms/step
    Epoch 180/500
    19/19 - 0s - loss: 0.5063 - accuracy: 0.7595 - val_loss: 0.5175 - val_accuracy: 0.7632 - 36ms/epoch - 2ms/step
    Epoch 181/500
    19/19 - 0s - loss: 0.5061 - accuracy: 0.7628 - val_loss: 0.5172 - val_accuracy: 0.7632 - 36ms/epoch - 2ms/step
    Epoch 182/500
    19/19 - 0s - loss: 0.5059 - accuracy: 0.7611 - val_loss: 0.5170 - val_accuracy: 0.7632 - 36ms/epoch - 2ms/step
    Epoch 183/500
    19/19 - 0s - loss: 0.5057 - accuracy: 0.7628 - val_loss: 0.5168 - val_accuracy: 0.7632 - 37ms/epoch - 2ms/step
    Epoch 184/500
    19/19 - 0s - loss: 0.5054 - accuracy: 0.7628 - val_loss: 0.5166 - val_accuracy: 0.7632 - 35ms/epoch - 2ms/step
    Epoch 185/500
    19/19 - 0s - loss: 0.5052 - accuracy: 0.7661 - val_loss: 0.5164 - val_accuracy: 0.7632 - 33ms/epoch - 2ms/step
    Epoch 186/500
    19/19 - 0s - loss: 0.5050 - accuracy: 0.7661 - val_loss: 0.5162 - val_accuracy: 0.7566 - 34ms/epoch - 2ms/step
    Epoch 187/500
    19/19 - 0s - loss: 0.5048 - accuracy: 0.7661 - val_loss: 0.5160 - val_accuracy: 0.7566 - 34ms/epoch - 2ms/step
    Epoch 188/500
    19/19 - 0s - loss: 0.5046 - accuracy: 0.7661 - val_loss: 0.5157 - val_accuracy: 0.7632 - 34ms/epoch - 2ms/step
    Epoch 189/500
    19/19 - 0s - loss: 0.5044 - accuracy: 0.7661 - val_loss: 0.5155 - val_accuracy: 0.7632 - 35ms/epoch - 2ms/step
    Epoch 190/500
    19/19 - 0s - loss: 0.5042 - accuracy: 0.7661 - val_loss: 0.5153 - val_accuracy: 0.7632 - 35ms/epoch - 2ms/step
    Epoch 191/500
    19/19 - 0s - loss: 0.5040 - accuracy: 0.7661 - val_loss: 0.5151 - val_accuracy: 0.7632 - 36ms/epoch - 2ms/step
    Epoch 192/500
    19/19 - 0s - loss: 0.5038 - accuracy: 0.7661 - val_loss: 0.5149 - val_accuracy: 0.7632 - 44ms/epoch - 2ms/step
    Epoch 193/500
    19/19 - 0s - loss: 0.5036 - accuracy: 0.7661 - val_loss: 0.5147 - val_accuracy: 0.7632 - 37ms/epoch - 2ms/step
    Epoch 194/500
    19/19 - 0s - loss: 0.5035 - accuracy: 0.7661 - val_loss: 0.5145 - val_accuracy: 0.7632 - 36ms/epoch - 2ms/step
    Epoch 195/500
    19/19 - 0s - loss: 0.5032 - accuracy: 0.7677 - val_loss: 0.5143 - val_accuracy: 0.7632 - 36ms/epoch - 2ms/step
    Epoch 196/500
    19/19 - 0s - loss: 0.5030 - accuracy: 0.7677 - val_loss: 0.5141 - val_accuracy: 0.7632 - 37ms/epoch - 2ms/step
    Epoch 197/500
    19/19 - 0s - loss: 0.5029 - accuracy: 0.7677 - val_loss: 0.5139 - val_accuracy: 0.7632 - 36ms/epoch - 2ms/step
    Epoch 198/500
    19/19 - 0s - loss: 0.5026 - accuracy: 0.7677 - val_loss: 0.5137 - val_accuracy: 0.7632 - 33ms/epoch - 2ms/step
    Epoch 199/500
    19/19 - 0s - loss: 0.5025 - accuracy: 0.7677 - val_loss: 0.5136 - val_accuracy: 0.7632 - 37ms/epoch - 2ms/step
    Epoch 200/500
    19/19 - 0s - loss: 0.5023 - accuracy: 0.7677 - val_loss: 0.5134 - val_accuracy: 0.7632 - 36ms/epoch - 2ms/step
    Epoch 201/500
    19/19 - 0s - loss: 0.5021 - accuracy: 0.7677 - val_loss: 0.5132 - val_accuracy: 0.7632 - 36ms/epoch - 2ms/step
    Epoch 202/500
    19/19 - 0s - loss: 0.5020 - accuracy: 0.7677 - val_loss: 0.5130 - val_accuracy: 0.7632 - 36ms/epoch - 2ms/step
    Epoch 203/500
    19/19 - 0s - loss: 0.5017 - accuracy: 0.7694 - val_loss: 0.5128 - val_accuracy: 0.7632 - 35ms/epoch - 2ms/step
    Epoch 204/500
    19/19 - 0s - loss: 0.5016 - accuracy: 0.7677 - val_loss: 0.5126 - val_accuracy: 0.7632 - 34ms/epoch - 2ms/step
    Epoch 205/500
    19/19 - 0s - loss: 0.5014 - accuracy: 0.7694 - val_loss: 0.5124 - val_accuracy: 0.7632 - 35ms/epoch - 2ms/step
    Epoch 206/500
    19/19 - 0s - loss: 0.5012 - accuracy: 0.7694 - val_loss: 0.5123 - val_accuracy: 0.7632 - 36ms/epoch - 2ms/step
    Epoch 207/500
    19/19 - 0s - loss: 0.5010 - accuracy: 0.7694 - val_loss: 0.5121 - val_accuracy: 0.7632 - 37ms/epoch - 2ms/step
    Epoch 208/500
    19/19 - 0s - loss: 0.5009 - accuracy: 0.7694 - val_loss: 0.5119 - val_accuracy: 0.7632 - 36ms/epoch - 2ms/step
    Epoch 209/500
    19/19 - 0s - loss: 0.5007 - accuracy: 0.7694 - val_loss: 0.5117 - val_accuracy: 0.7632 - 36ms/epoch - 2ms/step
    Epoch 210/500
    19/19 - 0s - loss: 0.5005 - accuracy: 0.7694 - val_loss: 0.5115 - val_accuracy: 0.7632 - 35ms/epoch - 2ms/step
    Epoch 211/500
    19/19 - 0s - loss: 0.5004 - accuracy: 0.7694 - val_loss: 0.5114 - val_accuracy: 0.7632 - 35ms/epoch - 2ms/step
    Epoch 212/500
    19/19 - 0s - loss: 0.5002 - accuracy: 0.7694 - val_loss: 0.5112 - val_accuracy: 0.7632 - 35ms/epoch - 2ms/step
    Epoch 213/500
    19/19 - 0s - loss: 0.5000 - accuracy: 0.7694 - val_loss: 0.5110 - val_accuracy: 0.7632 - 36ms/epoch - 2ms/step
    Epoch 214/500
    19/19 - 0s - loss: 0.4999 - accuracy: 0.7694 - val_loss: 0.5108 - val_accuracy: 0.7632 - 35ms/epoch - 2ms/step
    Epoch 215/500
    19/19 - 0s - loss: 0.4997 - accuracy: 0.7694 - val_loss: 0.5107 - val_accuracy: 0.7632 - 35ms/epoch - 2ms/step
    Epoch 216/500
    19/19 - 0s - loss: 0.4995 - accuracy: 0.7694 - val_loss: 0.5105 - val_accuracy: 0.7566 - 37ms/epoch - 2ms/step
    Epoch 217/500
    19/19 - 0s - loss: 0.4993 - accuracy: 0.7694 - val_loss: 0.5103 - val_accuracy: 0.7566 - 37ms/epoch - 2ms/step
    Epoch 218/500
    19/19 - 0s - loss: 0.4992 - accuracy: 0.7694 - val_loss: 0.5102 - val_accuracy: 0.7500 - 36ms/epoch - 2ms/step
    Epoch 219/500
    19/19 - 0s - loss: 0.4991 - accuracy: 0.7710 - val_loss: 0.5100 - val_accuracy: 0.7500 - 35ms/epoch - 2ms/step
    Epoch 220/500
    19/19 - 0s - loss: 0.4989 - accuracy: 0.7710 - val_loss: 0.5098 - val_accuracy: 0.7500 - 34ms/epoch - 2ms/step
    Epoch 221/500
    19/19 - 0s - loss: 0.4987 - accuracy: 0.7710 - val_loss: 0.5097 - val_accuracy: 0.7500 - 36ms/epoch - 2ms/step
    Epoch 222/500
    19/19 - 0s - loss: 0.4986 - accuracy: 0.7710 - val_loss: 0.5095 - val_accuracy: 0.7500 - 36ms/epoch - 2ms/step
    Epoch 223/500
    19/19 - 0s - loss: 0.4984 - accuracy: 0.7710 - val_loss: 0.5094 - val_accuracy: 0.7500 - 35ms/epoch - 2ms/step
    Epoch 224/500
    19/19 - 0s - loss: 0.4982 - accuracy: 0.7710 - val_loss: 0.5092 - val_accuracy: 0.7500 - 35ms/epoch - 2ms/step
    Epoch 225/500
    19/19 - 0s - loss: 0.4981 - accuracy: 0.7710 - val_loss: 0.5091 - val_accuracy: 0.7500 - 36ms/epoch - 2ms/step
    Epoch 226/500
    19/19 - 0s - loss: 0.4979 - accuracy: 0.7710 - val_loss: 0.5089 - val_accuracy: 0.7500 - 35ms/epoch - 2ms/step
    Epoch 227/500
    19/19 - 0s - loss: 0.4978 - accuracy: 0.7710 - val_loss: 0.5087 - val_accuracy: 0.7500 - 36ms/epoch - 2ms/step
    Epoch 228/500
    19/19 - 0s - loss: 0.4976 - accuracy: 0.7710 - val_loss: 0.5086 - val_accuracy: 0.7500 - 35ms/epoch - 2ms/step
    Epoch 229/500
    19/19 - 0s - loss: 0.4975 - accuracy: 0.7710 - val_loss: 0.5084 - val_accuracy: 0.7500 - 35ms/epoch - 2ms/step
    Epoch 230/500
    19/19 - 0s - loss: 0.4974 - accuracy: 0.7694 - val_loss: 0.5083 - val_accuracy: 0.7500 - 36ms/epoch - 2ms/step
    Epoch 231/500
    19/19 - 0s - loss: 0.4972 - accuracy: 0.7710 - val_loss: 0.5081 - val_accuracy: 0.7500 - 34ms/epoch - 2ms/step
    Epoch 232/500
    19/19 - 0s - loss: 0.4971 - accuracy: 0.7710 - val_loss: 0.5080 - val_accuracy: 0.7500 - 37ms/epoch - 2ms/step
    Epoch 233/500
    19/19 - 0s - loss: 0.4969 - accuracy: 0.7694 - val_loss: 0.5078 - val_accuracy: 0.7500 - 37ms/epoch - 2ms/step
    Epoch 234/500
    19/19 - 0s - loss: 0.4968 - accuracy: 0.7710 - val_loss: 0.5077 - val_accuracy: 0.7500 - 37ms/epoch - 2ms/step
    Epoch 235/500
    19/19 - 0s - loss: 0.4966 - accuracy: 0.7694 - val_loss: 0.5075 - val_accuracy: 0.7500 - 35ms/epoch - 2ms/step
    Epoch 236/500
    19/19 - 0s - loss: 0.4965 - accuracy: 0.7710 - val_loss: 0.5074 - val_accuracy: 0.7500 - 35ms/epoch - 2ms/step
    Epoch 237/500
    19/19 - 0s - loss: 0.4963 - accuracy: 0.7710 - val_loss: 0.5073 - val_accuracy: 0.7500 - 36ms/epoch - 2ms/step
    Epoch 238/500
    19/19 - 0s - loss: 0.4962 - accuracy: 0.7710 - val_loss: 0.5071 - val_accuracy: 0.7434 - 37ms/epoch - 2ms/step
    Epoch 239/500
    19/19 - 0s - loss: 0.4961 - accuracy: 0.7694 - val_loss: 0.5070 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 240/500
    19/19 - 0s - loss: 0.4959 - accuracy: 0.7694 - val_loss: 0.5068 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 241/500
    19/19 - 0s - loss: 0.4958 - accuracy: 0.7694 - val_loss: 0.5067 - val_accuracy: 0.7500 - 35ms/epoch - 2ms/step
    Epoch 242/500
    19/19 - 0s - loss: 0.4957 - accuracy: 0.7694 - val_loss: 0.5066 - val_accuracy: 0.7500 - 36ms/epoch - 2ms/step
    Epoch 243/500
    19/19 - 0s - loss: 0.4955 - accuracy: 0.7694 - val_loss: 0.5064 - val_accuracy: 0.7500 - 35ms/epoch - 2ms/step
    Epoch 244/500
    19/19 - 0s - loss: 0.4954 - accuracy: 0.7694 - val_loss: 0.5063 - val_accuracy: 0.7500 - 35ms/epoch - 2ms/step
    Epoch 245/500
    19/19 - 0s - loss: 0.4953 - accuracy: 0.7694 - val_loss: 0.5061 - val_accuracy: 0.7500 - 37ms/epoch - 2ms/step
    Epoch 246/500
    19/19 - 0s - loss: 0.4951 - accuracy: 0.7694 - val_loss: 0.5060 - val_accuracy: 0.7500 - 37ms/epoch - 2ms/step
    Epoch 247/500
    19/19 - 0s - loss: 0.4950 - accuracy: 0.7694 - val_loss: 0.5059 - val_accuracy: 0.7500 - 35ms/epoch - 2ms/step
    Epoch 248/500
    19/19 - 0s - loss: 0.4949 - accuracy: 0.7694 - val_loss: 0.5057 - val_accuracy: 0.7500 - 36ms/epoch - 2ms/step
    Epoch 249/500
    19/19 - 0s - loss: 0.4947 - accuracy: 0.7694 - val_loss: 0.5056 - val_accuracy: 0.7500 - 37ms/epoch - 2ms/step
    Epoch 250/500
    19/19 - 0s - loss: 0.4946 - accuracy: 0.7694 - val_loss: 0.5055 - val_accuracy: 0.7500 - 37ms/epoch - 2ms/step
    Epoch 251/500
    19/19 - 0s - loss: 0.4945 - accuracy: 0.7694 - val_loss: 0.5054 - val_accuracy: 0.7500 - 37ms/epoch - 2ms/step
    Epoch 252/500
    19/19 - 0s - loss: 0.4944 - accuracy: 0.7694 - val_loss: 0.5052 - val_accuracy: 0.7500 - 37ms/epoch - 2ms/step
    Epoch 253/500
    19/19 - 0s - loss: 0.4942 - accuracy: 0.7694 - val_loss: 0.5051 - val_accuracy: 0.7500 - 36ms/epoch - 2ms/step
    Epoch 254/500
    19/19 - 0s - loss: 0.4941 - accuracy: 0.7694 - val_loss: 0.5050 - val_accuracy: 0.7500 - 37ms/epoch - 2ms/step
    Epoch 255/500
    19/19 - 0s - loss: 0.4940 - accuracy: 0.7694 - val_loss: 0.5048 - val_accuracy: 0.7500 - 35ms/epoch - 2ms/step
    Epoch 256/500
    19/19 - 0s - loss: 0.4938 - accuracy: 0.7694 - val_loss: 0.5047 - val_accuracy: 0.7500 - 35ms/epoch - 2ms/step
    Epoch 257/500
    19/19 - 0s - loss: 0.4937 - accuracy: 0.7694 - val_loss: 0.5046 - val_accuracy: 0.7500 - 36ms/epoch - 2ms/step
    Epoch 258/500
    19/19 - 0s - loss: 0.4936 - accuracy: 0.7677 - val_loss: 0.5045 - val_accuracy: 0.7500 - 35ms/epoch - 2ms/step
    Epoch 259/500
    19/19 - 0s - loss: 0.4935 - accuracy: 0.7694 - val_loss: 0.5043 - val_accuracy: 0.7500 - 34ms/epoch - 2ms/step
    Epoch 260/500
    19/19 - 0s - loss: 0.4934 - accuracy: 0.7694 - val_loss: 0.5042 - val_accuracy: 0.7500 - 34ms/epoch - 2ms/step
    Epoch 261/500
    19/19 - 0s - loss: 0.4932 - accuracy: 0.7694 - val_loss: 0.5041 - val_accuracy: 0.7500 - 34ms/epoch - 2ms/step
    Epoch 262/500
    19/19 - 0s - loss: 0.4931 - accuracy: 0.7694 - val_loss: 0.5040 - val_accuracy: 0.7500 - 34ms/epoch - 2ms/step
    Epoch 263/500
    19/19 - 0s - loss: 0.4930 - accuracy: 0.7694 - val_loss: 0.5039 - val_accuracy: 0.7500 - 36ms/epoch - 2ms/step
    Epoch 264/500
    19/19 - 0s - loss: 0.4929 - accuracy: 0.7694 - val_loss: 0.5037 - val_accuracy: 0.7500 - 35ms/epoch - 2ms/step
    Epoch 265/500
    19/19 - 0s - loss: 0.4928 - accuracy: 0.7694 - val_loss: 0.5036 - val_accuracy: 0.7500 - 34ms/epoch - 2ms/step
    Epoch 266/500
    19/19 - 0s - loss: 0.4927 - accuracy: 0.7694 - val_loss: 0.5035 - val_accuracy: 0.7500 - 34ms/epoch - 2ms/step
    Epoch 267/500
    19/19 - 0s - loss: 0.4926 - accuracy: 0.7694 - val_loss: 0.5034 - val_accuracy: 0.7500 - 36ms/epoch - 2ms/step
    Epoch 268/500
    19/19 - 0s - loss: 0.4924 - accuracy: 0.7694 - val_loss: 0.5033 - val_accuracy: 0.7500 - 34ms/epoch - 2ms/step
    Epoch 269/500
    19/19 - 0s - loss: 0.4923 - accuracy: 0.7710 - val_loss: 0.5032 - val_accuracy: 0.7500 - 36ms/epoch - 2ms/step
    Epoch 270/500
    19/19 - 0s - loss: 0.4922 - accuracy: 0.7694 - val_loss: 0.5031 - val_accuracy: 0.7500 - 35ms/epoch - 2ms/step
    Epoch 271/500
    19/19 - 0s - loss: 0.4921 - accuracy: 0.7694 - val_loss: 0.5029 - val_accuracy: 0.7500 - 36ms/epoch - 2ms/step
    Epoch 272/500
    19/19 - 0s - loss: 0.4920 - accuracy: 0.7710 - val_loss: 0.5028 - val_accuracy: 0.7500 - 36ms/epoch - 2ms/step
    Epoch 273/500
    19/19 - 0s - loss: 0.4919 - accuracy: 0.7727 - val_loss: 0.5027 - val_accuracy: 0.7500 - 36ms/epoch - 2ms/step
    Epoch 274/500
    19/19 - 0s - loss: 0.4918 - accuracy: 0.7727 - val_loss: 0.5026 - val_accuracy: 0.7500 - 35ms/epoch - 2ms/step
    Epoch 275/500
    19/19 - 0s - loss: 0.4916 - accuracy: 0.7743 - val_loss: 0.5025 - val_accuracy: 0.7500 - 35ms/epoch - 2ms/step
    Epoch 276/500
    19/19 - 0s - loss: 0.4915 - accuracy: 0.7743 - val_loss: 0.5024 - val_accuracy: 0.7500 - 34ms/epoch - 2ms/step
    Epoch 277/500
    19/19 - 0s - loss: 0.4914 - accuracy: 0.7743 - val_loss: 0.5023 - val_accuracy: 0.7500 - 34ms/epoch - 2ms/step
    Epoch 278/500
    19/19 - 0s - loss: 0.4913 - accuracy: 0.7759 - val_loss: 0.5022 - val_accuracy: 0.7500 - 35ms/epoch - 2ms/step
    Epoch 279/500
    19/19 - 0s - loss: 0.4912 - accuracy: 0.7743 - val_loss: 0.5021 - val_accuracy: 0.7500 - 34ms/epoch - 2ms/step
    Epoch 280/500
    19/19 - 0s - loss: 0.4911 - accuracy: 0.7759 - val_loss: 0.5020 - val_accuracy: 0.7500 - 34ms/epoch - 2ms/step
    Epoch 281/500
    19/19 - 0s - loss: 0.4910 - accuracy: 0.7743 - val_loss: 0.5019 - val_accuracy: 0.7500 - 35ms/epoch - 2ms/step
    Epoch 282/500
    19/19 - 0s - loss: 0.4909 - accuracy: 0.7743 - val_loss: 0.5018 - val_accuracy: 0.7500 - 34ms/epoch - 2ms/step
    Epoch 283/500
    19/19 - 0s - loss: 0.4908 - accuracy: 0.7743 - val_loss: 0.5017 - val_accuracy: 0.7500 - 36ms/epoch - 2ms/step
    Epoch 284/500
    19/19 - 0s - loss: 0.4907 - accuracy: 0.7743 - val_loss: 0.5016 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 285/500
    19/19 - 0s - loss: 0.4906 - accuracy: 0.7743 - val_loss: 0.5014 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 286/500
    19/19 - 0s - loss: 0.4905 - accuracy: 0.7727 - val_loss: 0.5013 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 287/500
    19/19 - 0s - loss: 0.4904 - accuracy: 0.7727 - val_loss: 0.5012 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 288/500
    19/19 - 0s - loss: 0.4903 - accuracy: 0.7727 - val_loss: 0.5011 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 289/500
    19/19 - 0s - loss: 0.4902 - accuracy: 0.7727 - val_loss: 0.5010 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 290/500
    19/19 - 0s - loss: 0.4901 - accuracy: 0.7727 - val_loss: 0.5009 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 291/500
    19/19 - 0s - loss: 0.4900 - accuracy: 0.7727 - val_loss: 0.5009 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 292/500
    19/19 - 0s - loss: 0.4899 - accuracy: 0.7710 - val_loss: 0.5008 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 293/500
    19/19 - 0s - loss: 0.4898 - accuracy: 0.7727 - val_loss: 0.5007 - val_accuracy: 0.7434 - 37ms/epoch - 2ms/step
    Epoch 294/500
    19/19 - 0s - loss: 0.4897 - accuracy: 0.7727 - val_loss: 0.5006 - val_accuracy: 0.7434 - 36ms/epoch - 2ms/step
    Epoch 295/500
    19/19 - 0s - loss: 0.4896 - accuracy: 0.7743 - val_loss: 0.5005 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 296/500
    19/19 - 0s - loss: 0.4895 - accuracy: 0.7727 - val_loss: 0.5004 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 297/500
    19/19 - 0s - loss: 0.4894 - accuracy: 0.7727 - val_loss: 0.5003 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 298/500
    19/19 - 0s - loss: 0.4893 - accuracy: 0.7727 - val_loss: 0.5002 - val_accuracy: 0.7434 - 36ms/epoch - 2ms/step
    Epoch 299/500
    19/19 - 0s - loss: 0.4892 - accuracy: 0.7743 - val_loss: 0.5001 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 300/500
    19/19 - 0s - loss: 0.4891 - accuracy: 0.7743 - val_loss: 0.5000 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 301/500
    19/19 - 0s - loss: 0.4890 - accuracy: 0.7727 - val_loss: 0.4999 - val_accuracy: 0.7434 - 32ms/epoch - 2ms/step
    Epoch 302/500
    19/19 - 0s - loss: 0.4890 - accuracy: 0.7743 - val_loss: 0.4998 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 303/500
    19/19 - 0s - loss: 0.4888 - accuracy: 0.7743 - val_loss: 0.4997 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 304/500
    19/19 - 0s - loss: 0.4888 - accuracy: 0.7743 - val_loss: 0.4996 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 305/500
    19/19 - 0s - loss: 0.4887 - accuracy: 0.7743 - val_loss: 0.4995 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 306/500
    19/19 - 0s - loss: 0.4886 - accuracy: 0.7743 - val_loss: 0.4995 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 307/500
    19/19 - 0s - loss: 0.4885 - accuracy: 0.7743 - val_loss: 0.4994 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 308/500
    19/19 - 0s - loss: 0.4884 - accuracy: 0.7743 - val_loss: 0.4993 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 309/500
    19/19 - 0s - loss: 0.4883 - accuracy: 0.7743 - val_loss: 0.4992 - val_accuracy: 0.7434 - 36ms/epoch - 2ms/step
    Epoch 310/500
    19/19 - 0s - loss: 0.4882 - accuracy: 0.7743 - val_loss: 0.4991 - val_accuracy: 0.7434 - 36ms/epoch - 2ms/step
    Epoch 311/500
    19/19 - 0s - loss: 0.4882 - accuracy: 0.7759 - val_loss: 0.4990 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 312/500
    19/19 - 0s - loss: 0.4881 - accuracy: 0.7743 - val_loss: 0.4989 - val_accuracy: 0.7434 - 36ms/epoch - 2ms/step
    Epoch 313/500
    19/19 - 0s - loss: 0.4880 - accuracy: 0.7743 - val_loss: 0.4989 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 314/500
    19/19 - 0s - loss: 0.4879 - accuracy: 0.7743 - val_loss: 0.4988 - val_accuracy: 0.7434 - 36ms/epoch - 2ms/step
    Epoch 315/500
    19/19 - 0s - loss: 0.4878 - accuracy: 0.7743 - val_loss: 0.4987 - val_accuracy: 0.7434 - 37ms/epoch - 2ms/step
    Epoch 316/500
    19/19 - 0s - loss: 0.4877 - accuracy: 0.7743 - val_loss: 0.4986 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 317/500
    19/19 - 0s - loss: 0.4876 - accuracy: 0.7743 - val_loss: 0.4985 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 318/500
    19/19 - 0s - loss: 0.4876 - accuracy: 0.7727 - val_loss: 0.4984 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 319/500
    19/19 - 0s - loss: 0.4874 - accuracy: 0.7743 - val_loss: 0.4984 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 320/500
    19/19 - 0s - loss: 0.4874 - accuracy: 0.7759 - val_loss: 0.4983 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 321/500
    19/19 - 0s - loss: 0.4873 - accuracy: 0.7759 - val_loss: 0.4982 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 322/500
    19/19 - 0s - loss: 0.4872 - accuracy: 0.7759 - val_loss: 0.4981 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 323/500
    19/19 - 0s - loss: 0.4871 - accuracy: 0.7759 - val_loss: 0.4980 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 324/500
    19/19 - 0s - loss: 0.4871 - accuracy: 0.7759 - val_loss: 0.4980 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 325/500
    19/19 - 0s - loss: 0.4870 - accuracy: 0.7759 - val_loss: 0.4979 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 326/500
    19/19 - 0s - loss: 0.4869 - accuracy: 0.7759 - val_loss: 0.4978 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 327/500
    19/19 - 0s - loss: 0.4868 - accuracy: 0.7759 - val_loss: 0.4977 - val_accuracy: 0.7434 - 36ms/epoch - 2ms/step
    Epoch 328/500
    19/19 - 0s - loss: 0.4867 - accuracy: 0.7759 - val_loss: 0.4977 - val_accuracy: 0.7434 - 36ms/epoch - 2ms/step
    Epoch 329/500
    19/19 - 0s - loss: 0.4867 - accuracy: 0.7759 - val_loss: 0.4976 - val_accuracy: 0.7434 - 36ms/epoch - 2ms/step
    Epoch 330/500
    19/19 - 0s - loss: 0.4866 - accuracy: 0.7759 - val_loss: 0.4975 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 331/500
    19/19 - 0s - loss: 0.4865 - accuracy: 0.7759 - val_loss: 0.4974 - val_accuracy: 0.7434 - 36ms/epoch - 2ms/step
    Epoch 332/500
    19/19 - 0s - loss: 0.4864 - accuracy: 0.7759 - val_loss: 0.4974 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 333/500
    19/19 - 0s - loss: 0.4864 - accuracy: 0.7759 - val_loss: 0.4973 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 334/500
    19/19 - 0s - loss: 0.4863 - accuracy: 0.7759 - val_loss: 0.4972 - val_accuracy: 0.7434 - 37ms/epoch - 2ms/step
    Epoch 335/500
    19/19 - 0s - loss: 0.4862 - accuracy: 0.7759 - val_loss: 0.4971 - val_accuracy: 0.7434 - 37ms/epoch - 2ms/step
    Epoch 336/500
    19/19 - 0s - loss: 0.4861 - accuracy: 0.7759 - val_loss: 0.4971 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 337/500
    19/19 - 0s - loss: 0.4861 - accuracy: 0.7759 - val_loss: 0.4970 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 338/500
    19/19 - 0s - loss: 0.4860 - accuracy: 0.7759 - val_loss: 0.4969 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 339/500
    19/19 - 0s - loss: 0.4859 - accuracy: 0.7776 - val_loss: 0.4968 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 340/500
    19/19 - 0s - loss: 0.4858 - accuracy: 0.7776 - val_loss: 0.4968 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 341/500
    19/19 - 0s - loss: 0.4858 - accuracy: 0.7776 - val_loss: 0.4967 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 342/500
    19/19 - 0s - loss: 0.4857 - accuracy: 0.7759 - val_loss: 0.4966 - val_accuracy: 0.7434 - 36ms/epoch - 2ms/step
    Epoch 343/500
    19/19 - 0s - loss: 0.4857 - accuracy: 0.7759 - val_loss: 0.4966 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 344/500
    19/19 - 0s - loss: 0.4856 - accuracy: 0.7776 - val_loss: 0.4965 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 345/500
    19/19 - 0s - loss: 0.4855 - accuracy: 0.7776 - val_loss: 0.4964 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 346/500
    19/19 - 0s - loss: 0.4854 - accuracy: 0.7776 - val_loss: 0.4964 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 347/500
    19/19 - 0s - loss: 0.4854 - accuracy: 0.7776 - val_loss: 0.4963 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 348/500
    19/19 - 0s - loss: 0.4853 - accuracy: 0.7776 - val_loss: 0.4962 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 349/500
    19/19 - 0s - loss: 0.4852 - accuracy: 0.7776 - val_loss: 0.4962 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 350/500
    19/19 - 0s - loss: 0.4851 - accuracy: 0.7776 - val_loss: 0.4961 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 351/500
    19/19 - 0s - loss: 0.4851 - accuracy: 0.7776 - val_loss: 0.4960 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 352/500
    19/19 - 0s - loss: 0.4850 - accuracy: 0.7776 - val_loss: 0.4960 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 353/500
    19/19 - 0s - loss: 0.4849 - accuracy: 0.7776 - val_loss: 0.4959 - val_accuracy: 0.7434 - 32ms/epoch - 2ms/step
    Epoch 354/500
    19/19 - 0s - loss: 0.4848 - accuracy: 0.7776 - val_loss: 0.4958 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 355/500
    19/19 - 0s - loss: 0.4848 - accuracy: 0.7776 - val_loss: 0.4958 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 356/500
    19/19 - 0s - loss: 0.4847 - accuracy: 0.7776 - val_loss: 0.4957 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 357/500
    19/19 - 0s - loss: 0.4847 - accuracy: 0.7776 - val_loss: 0.4956 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 358/500
    19/19 - 0s - loss: 0.4846 - accuracy: 0.7776 - val_loss: 0.4956 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 359/500
    19/19 - 0s - loss: 0.4845 - accuracy: 0.7776 - val_loss: 0.4955 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 360/500
    19/19 - 0s - loss: 0.4845 - accuracy: 0.7776 - val_loss: 0.4955 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 361/500
    19/19 - 0s - loss: 0.4844 - accuracy: 0.7776 - val_loss: 0.4954 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 362/500
    19/19 - 0s - loss: 0.4844 - accuracy: 0.7776 - val_loss: 0.4953 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 363/500
    19/19 - 0s - loss: 0.4843 - accuracy: 0.7759 - val_loss: 0.4953 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 364/500
    19/19 - 0s - loss: 0.4842 - accuracy: 0.7759 - val_loss: 0.4952 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 365/500
    19/19 - 0s - loss: 0.4842 - accuracy: 0.7759 - val_loss: 0.4952 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 366/500
    19/19 - 0s - loss: 0.4841 - accuracy: 0.7759 - val_loss: 0.4951 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 367/500
    19/19 - 0s - loss: 0.4840 - accuracy: 0.7776 - val_loss: 0.4950 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 368/500
    19/19 - 0s - loss: 0.4840 - accuracy: 0.7759 - val_loss: 0.4950 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 369/500
    19/19 - 0s - loss: 0.4839 - accuracy: 0.7759 - val_loss: 0.4949 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 370/500
    19/19 - 0s - loss: 0.4838 - accuracy: 0.7759 - val_loss: 0.4949 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 371/500
    19/19 - 0s - loss: 0.4838 - accuracy: 0.7759 - val_loss: 0.4948 - val_accuracy: 0.7368 - 33ms/epoch - 2ms/step
    Epoch 372/500
    19/19 - 0s - loss: 0.4837 - accuracy: 0.7743 - val_loss: 0.4947 - val_accuracy: 0.7368 - 34ms/epoch - 2ms/step
    Epoch 373/500
    19/19 - 0s - loss: 0.4837 - accuracy: 0.7743 - val_loss: 0.4947 - val_accuracy: 0.7368 - 34ms/epoch - 2ms/step
    Epoch 374/500
    19/19 - 0s - loss: 0.4836 - accuracy: 0.7743 - val_loss: 0.4946 - val_accuracy: 0.7368 - 33ms/epoch - 2ms/step
    Epoch 375/500
    19/19 - 0s - loss: 0.4836 - accuracy: 0.7743 - val_loss: 0.4946 - val_accuracy: 0.7368 - 34ms/epoch - 2ms/step
    Epoch 376/500
    19/19 - 0s - loss: 0.4835 - accuracy: 0.7743 - val_loss: 0.4945 - val_accuracy: 0.7368 - 33ms/epoch - 2ms/step
    Epoch 377/500
    19/19 - 0s - loss: 0.4834 - accuracy: 0.7743 - val_loss: 0.4945 - val_accuracy: 0.7368 - 33ms/epoch - 2ms/step
    Epoch 378/500
    19/19 - 0s - loss: 0.4834 - accuracy: 0.7743 - val_loss: 0.4944 - val_accuracy: 0.7368 - 34ms/epoch - 2ms/step
    Epoch 379/500
    19/19 - 0s - loss: 0.4833 - accuracy: 0.7743 - val_loss: 0.4943 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 380/500
    19/19 - 0s - loss: 0.4832 - accuracy: 0.7743 - val_loss: 0.4943 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 381/500
    19/19 - 0s - loss: 0.4832 - accuracy: 0.7727 - val_loss: 0.4942 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 382/500
    19/19 - 0s - loss: 0.4831 - accuracy: 0.7743 - val_loss: 0.4942 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 383/500
    19/19 - 0s - loss: 0.4831 - accuracy: 0.7727 - val_loss: 0.4941 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 384/500
    19/19 - 0s - loss: 0.4830 - accuracy: 0.7727 - val_loss: 0.4941 - val_accuracy: 0.7434 - 36ms/epoch - 2ms/step
    Epoch 385/500
    19/19 - 0s - loss: 0.4830 - accuracy: 0.7727 - val_loss: 0.4940 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 386/500
    19/19 - 0s - loss: 0.4829 - accuracy: 0.7727 - val_loss: 0.4940 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 387/500
    19/19 - 0s - loss: 0.4829 - accuracy: 0.7743 - val_loss: 0.4939 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 388/500
    19/19 - 0s - loss: 0.4828 - accuracy: 0.7743 - val_loss: 0.4939 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 389/500
    19/19 - 0s - loss: 0.4827 - accuracy: 0.7727 - val_loss: 0.4938 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 390/500
    19/19 - 0s - loss: 0.4827 - accuracy: 0.7727 - val_loss: 0.4938 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 391/500
    19/19 - 0s - loss: 0.4826 - accuracy: 0.7727 - val_loss: 0.4937 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 392/500
    19/19 - 0s - loss: 0.4826 - accuracy: 0.7727 - val_loss: 0.4937 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 393/500
    19/19 - 0s - loss: 0.4825 - accuracy: 0.7727 - val_loss: 0.4936 - val_accuracy: 0.7434 - 36ms/epoch - 2ms/step
    Epoch 394/500
    19/19 - 0s - loss: 0.4825 - accuracy: 0.7727 - val_loss: 0.4936 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 395/500
    19/19 - 0s - loss: 0.4824 - accuracy: 0.7727 - val_loss: 0.4935 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 396/500
    19/19 - 0s - loss: 0.4824 - accuracy: 0.7727 - val_loss: 0.4935 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 397/500
    19/19 - 0s - loss: 0.4823 - accuracy: 0.7727 - val_loss: 0.4934 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 398/500
    19/19 - 0s - loss: 0.4822 - accuracy: 0.7727 - val_loss: 0.4934 - val_accuracy: 0.7434 - 36ms/epoch - 2ms/step
    Epoch 399/500
    19/19 - 0s - loss: 0.4822 - accuracy: 0.7727 - val_loss: 0.4933 - val_accuracy: 0.7434 - 36ms/epoch - 2ms/step
    Epoch 400/500
    19/19 - 0s - loss: 0.4822 - accuracy: 0.7727 - val_loss: 0.4933 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 401/500
    19/19 - 0s - loss: 0.4821 - accuracy: 0.7727 - val_loss: 0.4932 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 402/500
    19/19 - 0s - loss: 0.4820 - accuracy: 0.7727 - val_loss: 0.4932 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 403/500
    19/19 - 0s - loss: 0.4820 - accuracy: 0.7727 - val_loss: 0.4931 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 404/500
    19/19 - 0s - loss: 0.4820 - accuracy: 0.7727 - val_loss: 0.4931 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 405/500
    19/19 - 0s - loss: 0.4819 - accuracy: 0.7727 - val_loss: 0.4930 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 406/500
    19/19 - 0s - loss: 0.4818 - accuracy: 0.7727 - val_loss: 0.4930 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 407/500
    19/19 - 0s - loss: 0.4818 - accuracy: 0.7727 - val_loss: 0.4929 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 408/500
    19/19 - 0s - loss: 0.4817 - accuracy: 0.7727 - val_loss: 0.4929 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 409/500
    19/19 - 0s - loss: 0.4817 - accuracy: 0.7727 - val_loss: 0.4928 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 410/500
    19/19 - 0s - loss: 0.4816 - accuracy: 0.7727 - val_loss: 0.4928 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 411/500
    19/19 - 0s - loss: 0.4816 - accuracy: 0.7727 - val_loss: 0.4927 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 412/500
    19/19 - 0s - loss: 0.4815 - accuracy: 0.7727 - val_loss: 0.4927 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 413/500
    19/19 - 0s - loss: 0.4815 - accuracy: 0.7727 - val_loss: 0.4927 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 414/500
    19/19 - 0s - loss: 0.4814 - accuracy: 0.7710 - val_loss: 0.4926 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 415/500
    19/19 - 0s - loss: 0.4814 - accuracy: 0.7727 - val_loss: 0.4926 - val_accuracy: 0.7434 - 33ms/epoch - 2ms/step
    Epoch 416/500
    19/19 - 0s - loss: 0.4813 - accuracy: 0.7727 - val_loss: 0.4925 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 417/500
    19/19 - 0s - loss: 0.4813 - accuracy: 0.7727 - val_loss: 0.4925 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 418/500
    19/19 - 0s - loss: 0.4813 - accuracy: 0.7727 - val_loss: 0.4924 - val_accuracy: 0.7434 - 35ms/epoch - 2ms/step
    Epoch 419/500
    19/19 - 0s - loss: 0.4812 - accuracy: 0.7727 - val_loss: 0.4924 - val_accuracy: 0.7434 - 34ms/epoch - 2ms/step
    Epoch 420/500
    19/19 - 0s - loss: 0.4811 - accuracy: 0.7710 - val_loss: 0.4924 - val_accuracy: 0.7434 - 36ms/epoch - 2ms/step
    Epoch 421/500
    19/19 - 0s - loss: 0.4811 - accuracy: 0.7710 - val_loss: 0.4923 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 422/500
    19/19 - 0s - loss: 0.4811 - accuracy: 0.7727 - val_loss: 0.4923 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 423/500
    19/19 - 0s - loss: 0.4810 - accuracy: 0.7710 - val_loss: 0.4922 - val_accuracy: 0.7368 - 37ms/epoch - 2ms/step
    Epoch 424/500
    19/19 - 0s - loss: 0.4810 - accuracy: 0.7710 - val_loss: 0.4922 - val_accuracy: 0.7368 - 37ms/epoch - 2ms/step
    Epoch 425/500
    19/19 - 0s - loss: 0.4809 - accuracy: 0.7710 - val_loss: 0.4921 - val_accuracy: 0.7368 - 35ms/epoch - 2ms/step
    Epoch 426/500
    19/19 - 0s - loss: 0.4809 - accuracy: 0.7710 - val_loss: 0.4921 - val_accuracy: 0.7368 - 35ms/epoch - 2ms/step
    Epoch 427/500
    19/19 - 0s - loss: 0.4808 - accuracy: 0.7710 - val_loss: 0.4921 - val_accuracy: 0.7368 - 38ms/epoch - 2ms/step
    Epoch 428/500
    19/19 - 0s - loss: 0.4808 - accuracy: 0.7710 - val_loss: 0.4920 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 429/500
    19/19 - 0s - loss: 0.4808 - accuracy: 0.7727 - val_loss: 0.4920 - val_accuracy: 0.7368 - 35ms/epoch - 2ms/step
    Epoch 430/500
    19/19 - 0s - loss: 0.4807 - accuracy: 0.7727 - val_loss: 0.4919 - val_accuracy: 0.7368 - 37ms/epoch - 2ms/step
    Epoch 431/500
    19/19 - 0s - loss: 0.4807 - accuracy: 0.7727 - val_loss: 0.4919 - val_accuracy: 0.7368 - 35ms/epoch - 2ms/step
    Epoch 432/500
    19/19 - 0s - loss: 0.4807 - accuracy: 0.7727 - val_loss: 0.4919 - val_accuracy: 0.7368 - 35ms/epoch - 2ms/step
    Epoch 433/500
    19/19 - 0s - loss: 0.4806 - accuracy: 0.7727 - val_loss: 0.4918 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 434/500
    19/19 - 0s - loss: 0.4805 - accuracy: 0.7727 - val_loss: 0.4918 - val_accuracy: 0.7368 - 34ms/epoch - 2ms/step
    Epoch 435/500
    19/19 - 0s - loss: 0.4805 - accuracy: 0.7727 - val_loss: 0.4917 - val_accuracy: 0.7368 - 35ms/epoch - 2ms/step
    Epoch 436/500
    19/19 - 0s - loss: 0.4804 - accuracy: 0.7727 - val_loss: 0.4917 - val_accuracy: 0.7368 - 35ms/epoch - 2ms/step
    Epoch 437/500
    19/19 - 0s - loss: 0.4804 - accuracy: 0.7727 - val_loss: 0.4917 - val_accuracy: 0.7368 - 35ms/epoch - 2ms/step
    Epoch 438/500
    19/19 - 0s - loss: 0.4804 - accuracy: 0.7727 - val_loss: 0.4916 - val_accuracy: 0.7368 - 34ms/epoch - 2ms/step
    Epoch 439/500
    19/19 - 0s - loss: 0.4803 - accuracy: 0.7727 - val_loss: 0.4916 - val_accuracy: 0.7368 - 34ms/epoch - 2ms/step
    Epoch 440/500
    19/19 - 0s - loss: 0.4803 - accuracy: 0.7727 - val_loss: 0.4915 - val_accuracy: 0.7368 - 34ms/epoch - 2ms/step
    Epoch 441/500
    19/19 - 0s - loss: 0.4802 - accuracy: 0.7727 - val_loss: 0.4915 - val_accuracy: 0.7368 - 34ms/epoch - 2ms/step
    Epoch 442/500
    19/19 - 0s - loss: 0.4802 - accuracy: 0.7727 - val_loss: 0.4915 - val_accuracy: 0.7368 - 34ms/epoch - 2ms/step
    Epoch 443/500
    19/19 - 0s - loss: 0.4801 - accuracy: 0.7727 - val_loss: 0.4914 - val_accuracy: 0.7368 - 35ms/epoch - 2ms/step
    Epoch 444/500
    19/19 - 0s - loss: 0.4801 - accuracy: 0.7727 - val_loss: 0.4914 - val_accuracy: 0.7368 - 34ms/epoch - 2ms/step
    Epoch 445/500
    19/19 - 0s - loss: 0.4801 - accuracy: 0.7727 - val_loss: 0.4914 - val_accuracy: 0.7368 - 35ms/epoch - 2ms/step
    Epoch 446/500
    19/19 - 0s - loss: 0.4800 - accuracy: 0.7727 - val_loss: 0.4913 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 447/500
    19/19 - 0s - loss: 0.4800 - accuracy: 0.7727 - val_loss: 0.4913 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 448/500
    19/19 - 0s - loss: 0.4799 - accuracy: 0.7727 - val_loss: 0.4913 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 449/500
    19/19 - 0s - loss: 0.4799 - accuracy: 0.7727 - val_loss: 0.4912 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 450/500
    19/19 - 0s - loss: 0.4799 - accuracy: 0.7727 - val_loss: 0.4912 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 451/500
    19/19 - 0s - loss: 0.4798 - accuracy: 0.7727 - val_loss: 0.4911 - val_accuracy: 0.7368 - 37ms/epoch - 2ms/step
    Epoch 452/500
    19/19 - 0s - loss: 0.4798 - accuracy: 0.7727 - val_loss: 0.4911 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 453/500
    19/19 - 0s - loss: 0.4798 - accuracy: 0.7727 - val_loss: 0.4911 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 454/500
    19/19 - 0s - loss: 0.4797 - accuracy: 0.7727 - val_loss: 0.4910 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 455/500
    19/19 - 0s - loss: 0.4797 - accuracy: 0.7727 - val_loss: 0.4910 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 456/500
    19/19 - 0s - loss: 0.4796 - accuracy: 0.7727 - val_loss: 0.4910 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 457/500
    19/19 - 0s - loss: 0.4796 - accuracy: 0.7727 - val_loss: 0.4909 - val_accuracy: 0.7368 - 35ms/epoch - 2ms/step
    Epoch 458/500
    19/19 - 0s - loss: 0.4796 - accuracy: 0.7727 - val_loss: 0.4909 - val_accuracy: 0.7368 - 37ms/epoch - 2ms/step
    Epoch 459/500
    19/19 - 0s - loss: 0.4795 - accuracy: 0.7727 - val_loss: 0.4909 - val_accuracy: 0.7368 - 37ms/epoch - 2ms/step
    Epoch 460/500
    19/19 - 0s - loss: 0.4795 - accuracy: 0.7727 - val_loss: 0.4908 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 461/500
    19/19 - 0s - loss: 0.4794 - accuracy: 0.7727 - val_loss: 0.4908 - val_accuracy: 0.7368 - 37ms/epoch - 2ms/step
    Epoch 462/500
    19/19 - 0s - loss: 0.4794 - accuracy: 0.7727 - val_loss: 0.4908 - val_accuracy: 0.7368 - 35ms/epoch - 2ms/step
    Epoch 463/500
    19/19 - 0s - loss: 0.4793 - accuracy: 0.7727 - val_loss: 0.4907 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 464/500
    19/19 - 0s - loss: 0.4793 - accuracy: 0.7727 - val_loss: 0.4907 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 465/500
    19/19 - 0s - loss: 0.4793 - accuracy: 0.7727 - val_loss: 0.4907 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 466/500
    19/19 - 0s - loss: 0.4793 - accuracy: 0.7727 - val_loss: 0.4906 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 467/500
    19/19 - 0s - loss: 0.4792 - accuracy: 0.7727 - val_loss: 0.4906 - val_accuracy: 0.7368 - 34ms/epoch - 2ms/step
    Epoch 468/500
    19/19 - 0s - loss: 0.4792 - accuracy: 0.7727 - val_loss: 0.4906 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 469/500
    19/19 - 0s - loss: 0.4792 - accuracy: 0.7727 - val_loss: 0.4905 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 470/500
    19/19 - 0s - loss: 0.4792 - accuracy: 0.7727 - val_loss: 0.4905 - val_accuracy: 0.7368 - 37ms/epoch - 2ms/step
    Epoch 471/500
    19/19 - 0s - loss: 0.4791 - accuracy: 0.7727 - val_loss: 0.4905 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 472/500
    19/19 - 0s - loss: 0.4790 - accuracy: 0.7727 - val_loss: 0.4904 - val_accuracy: 0.7368 - 34ms/epoch - 2ms/step
    Epoch 473/500
    19/19 - 0s - loss: 0.4790 - accuracy: 0.7743 - val_loss: 0.4904 - val_accuracy: 0.7368 - 38ms/epoch - 2ms/step
    Epoch 474/500
    19/19 - 0s - loss: 0.4790 - accuracy: 0.7727 - val_loss: 0.4904 - val_accuracy: 0.7368 - 37ms/epoch - 2ms/step
    Epoch 475/500
    19/19 - 0s - loss: 0.4789 - accuracy: 0.7743 - val_loss: 0.4903 - val_accuracy: 0.7368 - 35ms/epoch - 2ms/step
    Epoch 476/500
    19/19 - 0s - loss: 0.4789 - accuracy: 0.7743 - val_loss: 0.4903 - val_accuracy: 0.7368 - 38ms/epoch - 2ms/step
    Epoch 477/500
    19/19 - 0s - loss: 0.4788 - accuracy: 0.7759 - val_loss: 0.4903 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 478/500
    19/19 - 0s - loss: 0.4789 - accuracy: 0.7743 - val_loss: 0.4903 - val_accuracy: 0.7368 - 37ms/epoch - 2ms/step
    Epoch 479/500
    19/19 - 0s - loss: 0.4788 - accuracy: 0.7743 - val_loss: 0.4902 - val_accuracy: 0.7368 - 38ms/epoch - 2ms/step
    Epoch 480/500
    19/19 - 0s - loss: 0.4787 - accuracy: 0.7759 - val_loss: 0.4902 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 481/500
    19/19 - 0s - loss: 0.4787 - accuracy: 0.7759 - val_loss: 0.4902 - val_accuracy: 0.7368 - 37ms/epoch - 2ms/step
    Epoch 482/500
    19/19 - 0s - loss: 0.4787 - accuracy: 0.7759 - val_loss: 0.4901 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 483/500
    19/19 - 0s - loss: 0.4787 - accuracy: 0.7759 - val_loss: 0.4901 - val_accuracy: 0.7368 - 34ms/epoch - 2ms/step
    Epoch 484/500
    19/19 - 0s - loss: 0.4786 - accuracy: 0.7759 - val_loss: 0.4901 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 485/500
    19/19 - 0s - loss: 0.4786 - accuracy: 0.7759 - val_loss: 0.4900 - val_accuracy: 0.7368 - 37ms/epoch - 2ms/step
    Epoch 486/500
    19/19 - 0s - loss: 0.4786 - accuracy: 0.7759 - val_loss: 0.4900 - val_accuracy: 0.7368 - 37ms/epoch - 2ms/step
    Epoch 487/500
    19/19 - 0s - loss: 0.4785 - accuracy: 0.7759 - val_loss: 0.4900 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 488/500
    19/19 - 0s - loss: 0.4785 - accuracy: 0.7759 - val_loss: 0.4900 - val_accuracy: 0.7368 - 35ms/epoch - 2ms/step
    Epoch 489/500
    19/19 - 0s - loss: 0.4784 - accuracy: 0.7759 - val_loss: 0.4899 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 490/500
    19/19 - 0s - loss: 0.4784 - accuracy: 0.7759 - val_loss: 0.4899 - val_accuracy: 0.7368 - 37ms/epoch - 2ms/step
    Epoch 491/500
    19/19 - 0s - loss: 0.4784 - accuracy: 0.7759 - val_loss: 0.4899 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 492/500
    19/19 - 0s - loss: 0.4783 - accuracy: 0.7759 - val_loss: 0.4899 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 493/500
    19/19 - 0s - loss: 0.4783 - accuracy: 0.7759 - val_loss: 0.4898 - val_accuracy: 0.7368 - 35ms/epoch - 2ms/step
    Epoch 494/500
    19/19 - 0s - loss: 0.4783 - accuracy: 0.7759 - val_loss: 0.4898 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 495/500
    19/19 - 0s - loss: 0.4783 - accuracy: 0.7759 - val_loss: 0.4898 - val_accuracy: 0.7368 - 35ms/epoch - 2ms/step
    Epoch 496/500
    19/19 - 0s - loss: 0.4783 - accuracy: 0.7759 - val_loss: 0.4897 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 497/500
    19/19 - 0s - loss: 0.4782 - accuracy: 0.7759 - val_loss: 0.4897 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 498/500
    19/19 - 0s - loss: 0.4782 - accuracy: 0.7759 - val_loss: 0.4897 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 499/500
    19/19 - 0s - loss: 0.4782 - accuracy: 0.7759 - val_loss: 0.4897 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step
    Epoch 500/500
    19/19 - 0s - loss: 0.4781 - accuracy: 0.7759 - val_loss: 0.4896 - val_accuracy: 0.7368 - 36ms/epoch - 2ms/step



```python
model.evaluate(x_data, t_data)
```

    24/24 [==============================] - 0s 861us/step - loss: 0.4804 - accuracy: 0.7681





    [0.48035287857055664, 0.7681159377098083]




```python
import matplotlib.pyplot as plt

plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'], label = 'train loss')
plt.plot(hist.history['val_loss'], label = 'validation loss')

plt.legend(loc = 'best')

plt.show()
```


    
![png](output_8_0.png)
    



```python
import matplotlib.pyplot as plt

plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.grid()

plt.plot(hist.history['accuracy'], label = 'train accuracy')
plt.plot(hist.history['val_accuracy'], label = 'validation accuracy')

plt.legend(loc = 'best')

plt.show()
```


    
![png](output_9_0.png)
    



```python
import tensorflow as tf
import numpy as np

x_data = np.array([2, 4, 6, 8, 10,
                  12, 14, 16, 18, 20]).astype('float32')

t_data = np.array([0, 0, 0, 0, 0,
                   0, 1, 1, 1, 1,]).astype('float32')
```


```python
model = tf.keras.models.Sequential() # 모델의 구축

# 8 : 은닉층 노드 개수
# input_shape = (1,) : 입력층 노드 개수
# 1 : 출력층 노드 개수
model.add(tf.keras.layers.Dense(8, input_shape=(1,), activation = 'sigmoid')) 

model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
```


```python
model.compile(tf.keras.optimizers.SGD(learning_rate = 0.1), loss = 'binary_crossentropy', metrics =['accuracy'])

model.summary()


# Param # : 최적의 값으로 학습되는 가중치와 바이어스 개수
# 16 - 가중치개수 (1x8) +  바이어스 개수(8)
```

    Model: "sequential_2"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 8)                 16        
                                                                     
     dense_1 (Dense)             (None, 1)                 9         
                                                                     
    =================================================================
    Total params: 25
    Trainable params: 25
    Non-trainable params: 0
    _________________________________________________________________



```python
model.fit(x_data, t_data, epochs = 500)
```

    Epoch 1/500
    1/1 [==============================] - 0s 4ms/step - loss: 0.1432 - accuracy: 0.9000
    Epoch 2/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1068 - accuracy: 0.9000
    Epoch 3/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1431 - accuracy: 0.9000
    Epoch 4/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1067 - accuracy: 0.9000
    Epoch 5/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1430 - accuracy: 0.9000
    Epoch 6/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1067 - accuracy: 0.9000
    Epoch 7/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1430 - accuracy: 0.9000
    Epoch 8/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1066 - accuracy: 0.9000
    Epoch 9/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1429 - accuracy: 0.9000
    Epoch 10/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1066 - accuracy: 0.9000
    Epoch 11/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1429 - accuracy: 0.9000
    Epoch 12/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1066 - accuracy: 0.9000
    Epoch 13/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1428 - accuracy: 0.9000
    Epoch 14/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1065 - accuracy: 0.9000
    Epoch 15/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1428 - accuracy: 0.9000
    Epoch 16/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1065 - accuracy: 0.9000
    Epoch 17/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1427 - accuracy: 0.9000
    Epoch 18/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1064 - accuracy: 0.9000
    Epoch 19/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1426 - accuracy: 0.9000
    Epoch 20/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1064 - accuracy: 0.9000
    Epoch 21/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1426 - accuracy: 0.9000
    Epoch 22/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1063 - accuracy: 0.9000
    Epoch 23/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1425 - accuracy: 0.9000
    Epoch 24/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1063 - accuracy: 0.9000
    Epoch 25/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1425 - accuracy: 0.9000
    Epoch 26/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1062 - accuracy: 0.9000
    Epoch 27/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1424 - accuracy: 0.9000
    Epoch 28/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1062 - accuracy: 0.9000
    Epoch 29/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1423 - accuracy: 0.9000
    Epoch 30/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1062 - accuracy: 0.9000
    Epoch 31/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1423 - accuracy: 0.9000
    Epoch 32/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1061 - accuracy: 0.9000
    Epoch 33/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1422 - accuracy: 0.9000
    Epoch 34/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1061 - accuracy: 0.9000
    Epoch 35/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1422 - accuracy: 0.9000
    Epoch 36/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1060 - accuracy: 0.9000
    Epoch 37/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1421 - accuracy: 0.9000
    Epoch 38/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1060 - accuracy: 0.9000
    Epoch 39/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1421 - accuracy: 0.9000
    Epoch 40/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1059 - accuracy: 0.9000
    Epoch 41/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1420 - accuracy: 0.9000
    Epoch 42/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1059 - accuracy: 0.9000
    Epoch 43/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1419 - accuracy: 0.9000
    Epoch 44/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1059 - accuracy: 0.9000
    Epoch 45/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1419 - accuracy: 0.9000
    Epoch 46/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1058 - accuracy: 0.9000
    Epoch 47/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1418 - accuracy: 0.9000
    Epoch 48/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1058 - accuracy: 0.9000
    Epoch 49/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1418 - accuracy: 0.9000
    Epoch 50/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1057 - accuracy: 0.9000
    Epoch 51/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1417 - accuracy: 0.9000
    Epoch 52/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1057 - accuracy: 0.9000
    Epoch 53/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1417 - accuracy: 0.9000
    Epoch 54/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1056 - accuracy: 0.9000
    Epoch 55/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1416 - accuracy: 0.9000
    Epoch 56/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1056 - accuracy: 0.9000
    Epoch 57/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1415 - accuracy: 0.9000
    Epoch 58/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1056 - accuracy: 0.9000
    Epoch 59/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1415 - accuracy: 0.9000
    Epoch 60/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1055 - accuracy: 0.9000
    Epoch 61/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1414 - accuracy: 0.9000
    Epoch 62/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1055 - accuracy: 0.9000
    Epoch 63/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1414 - accuracy: 0.9000
    Epoch 64/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1054 - accuracy: 0.9000
    Epoch 65/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1413 - accuracy: 0.9000
    Epoch 66/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1054 - accuracy: 0.9000
    Epoch 67/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1413 - accuracy: 0.9000
    Epoch 68/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1053 - accuracy: 0.9000
    Epoch 69/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1412 - accuracy: 0.9000
    Epoch 70/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1053 - accuracy: 0.9000
    Epoch 71/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1411 - accuracy: 0.9000
    Epoch 72/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1053 - accuracy: 0.9000
    Epoch 73/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1411 - accuracy: 0.9000
    Epoch 74/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1052 - accuracy: 0.9000
    Epoch 75/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1410 - accuracy: 0.9000
    Epoch 76/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1052 - accuracy: 0.9000
    Epoch 77/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1410 - accuracy: 0.9000
    Epoch 78/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1051 - accuracy: 0.9000
    Epoch 79/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1409 - accuracy: 0.9000
    Epoch 80/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1051 - accuracy: 0.9000
    Epoch 81/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1409 - accuracy: 0.9000
    Epoch 82/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1050 - accuracy: 0.9000
    Epoch 83/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1408 - accuracy: 0.9000
    Epoch 84/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1050 - accuracy: 0.9000
    Epoch 85/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1407 - accuracy: 0.9000
    Epoch 86/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1050 - accuracy: 0.9000
    Epoch 87/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1407 - accuracy: 0.9000
    Epoch 88/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1049 - accuracy: 0.9000
    Epoch 89/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1406 - accuracy: 0.9000
    Epoch 90/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1049 - accuracy: 0.9000
    Epoch 91/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1406 - accuracy: 0.9000
    Epoch 92/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1048 - accuracy: 0.9000
    Epoch 93/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1405 - accuracy: 0.9000
    Epoch 94/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1048 - accuracy: 0.9000
    Epoch 95/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1405 - accuracy: 0.9000
    Epoch 96/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1047 - accuracy: 0.9000
    Epoch 97/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1404 - accuracy: 0.9000
    Epoch 98/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1047 - accuracy: 0.9000
    Epoch 99/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1404 - accuracy: 0.9000
    Epoch 100/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1047 - accuracy: 0.9000
    Epoch 101/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1403 - accuracy: 0.9000
    Epoch 102/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1046 - accuracy: 0.9000
    Epoch 103/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1402 - accuracy: 0.9000
    Epoch 104/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1046 - accuracy: 0.9000
    Epoch 105/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1402 - accuracy: 0.9000
    Epoch 106/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1045 - accuracy: 0.9000
    Epoch 107/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1401 - accuracy: 0.9000
    Epoch 108/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1045 - accuracy: 0.9000
    Epoch 109/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1401 - accuracy: 0.9000
    Epoch 110/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1045 - accuracy: 0.9000
    Epoch 111/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1400 - accuracy: 0.9000
    Epoch 112/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1044 - accuracy: 0.9000
    Epoch 113/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1400 - accuracy: 0.9000
    Epoch 114/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1044 - accuracy: 0.9000
    Epoch 115/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1399 - accuracy: 0.9000
    Epoch 116/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1043 - accuracy: 0.9000
    Epoch 117/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1398 - accuracy: 0.9000
    Epoch 118/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1043 - accuracy: 0.9000
    Epoch 119/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1398 - accuracy: 0.9000
    Epoch 120/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1042 - accuracy: 0.9000
    Epoch 121/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1397 - accuracy: 0.9000
    Epoch 122/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1042 - accuracy: 0.9000
    Epoch 123/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1397 - accuracy: 0.9000
    Epoch 124/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1042 - accuracy: 0.9000
    Epoch 125/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1396 - accuracy: 0.9000
    Epoch 126/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1041 - accuracy: 0.9000
    Epoch 127/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1396 - accuracy: 0.9000
    Epoch 128/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1041 - accuracy: 0.9000
    Epoch 129/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1395 - accuracy: 0.9000
    Epoch 130/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1040 - accuracy: 0.9000
    Epoch 131/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1395 - accuracy: 0.9000
    Epoch 132/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1040 - accuracy: 0.9000
    Epoch 133/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1394 - accuracy: 0.9000
    Epoch 134/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1040 - accuracy: 0.9000
    Epoch 135/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1393 - accuracy: 0.9000
    Epoch 136/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1039 - accuracy: 0.9000
    Epoch 137/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1393 - accuracy: 0.9000
    Epoch 138/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1039 - accuracy: 0.9000
    Epoch 139/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1392 - accuracy: 0.9000
    Epoch 140/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1038 - accuracy: 0.9000
    Epoch 141/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1392 - accuracy: 0.9000
    Epoch 142/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1038 - accuracy: 0.9000
    Epoch 143/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1391 - accuracy: 0.9000
    Epoch 144/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1038 - accuracy: 0.9000
    Epoch 145/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1391 - accuracy: 0.9000
    Epoch 146/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1037 - accuracy: 0.9000
    Epoch 147/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1390 - accuracy: 0.9000
    Epoch 148/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1037 - accuracy: 0.9000
    Epoch 149/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1390 - accuracy: 0.9000
    Epoch 150/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1036 - accuracy: 0.9000
    Epoch 151/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1389 - accuracy: 0.9000
    Epoch 152/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1036 - accuracy: 0.9000
    Epoch 153/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1389 - accuracy: 0.9000
    Epoch 154/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1036 - accuracy: 0.9000
    Epoch 155/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1388 - accuracy: 0.9000
    Epoch 156/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1035 - accuracy: 0.9000
    Epoch 157/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1387 - accuracy: 0.9000
    Epoch 158/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1035 - accuracy: 0.9000
    Epoch 159/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1387 - accuracy: 0.9000
    Epoch 160/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1034 - accuracy: 0.9000
    Epoch 161/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1386 - accuracy: 0.9000
    Epoch 162/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1034 - accuracy: 0.9000
    Epoch 163/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1386 - accuracy: 0.9000
    Epoch 164/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1034 - accuracy: 0.9000
    Epoch 165/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1385 - accuracy: 0.9000
    Epoch 166/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1033 - accuracy: 0.9000
    Epoch 167/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1385 - accuracy: 0.9000
    Epoch 168/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1033 - accuracy: 0.9000
    Epoch 169/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1384 - accuracy: 0.9000
    Epoch 170/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1032 - accuracy: 0.9000
    Epoch 171/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1384 - accuracy: 0.9000
    Epoch 172/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1032 - accuracy: 0.9000
    Epoch 173/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1383 - accuracy: 0.9000
    Epoch 174/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1032 - accuracy: 0.9000
    Epoch 175/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1383 - accuracy: 0.9000
    Epoch 176/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1031 - accuracy: 0.9000
    Epoch 177/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1382 - accuracy: 0.9000
    Epoch 178/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1031 - accuracy: 0.9000
    Epoch 179/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1381 - accuracy: 0.9000
    Epoch 180/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1030 - accuracy: 0.9000
    Epoch 181/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1381 - accuracy: 0.9000
    Epoch 182/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1030 - accuracy: 0.9000
    Epoch 183/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1380 - accuracy: 0.9000
    Epoch 184/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1030 - accuracy: 0.9000
    Epoch 185/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1380 - accuracy: 0.9000
    Epoch 186/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1029 - accuracy: 0.9000
    Epoch 187/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1379 - accuracy: 0.9000
    Epoch 188/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1029 - accuracy: 0.9000
    Epoch 189/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1379 - accuracy: 0.9000
    Epoch 190/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1028 - accuracy: 0.9000
    Epoch 191/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1378 - accuracy: 0.9000
    Epoch 192/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1028 - accuracy: 0.9000
    Epoch 193/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1378 - accuracy: 0.9000
    Epoch 194/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1028 - accuracy: 0.9000
    Epoch 195/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1377 - accuracy: 0.9000
    Epoch 196/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1027 - accuracy: 0.9000
    Epoch 197/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1377 - accuracy: 0.9000
    Epoch 198/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1027 - accuracy: 0.9000
    Epoch 199/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1376 - accuracy: 0.9000
    Epoch 200/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1026 - accuracy: 0.9000
    Epoch 201/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1376 - accuracy: 0.9000
    Epoch 202/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1026 - accuracy: 0.9000
    Epoch 203/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1375 - accuracy: 0.9000
    Epoch 204/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1026 - accuracy: 0.9000
    Epoch 205/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1374 - accuracy: 0.9000
    Epoch 206/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1025 - accuracy: 0.9000
    Epoch 207/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1374 - accuracy: 0.9000
    Epoch 208/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1025 - accuracy: 0.9000
    Epoch 209/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1373 - accuracy: 0.9000
    Epoch 210/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1024 - accuracy: 0.9000
    Epoch 211/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1373 - accuracy: 0.9000
    Epoch 212/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1024 - accuracy: 0.9000
    Epoch 213/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1372 - accuracy: 0.9000
    Epoch 214/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1024 - accuracy: 0.9000
    Epoch 215/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1372 - accuracy: 0.9000
    Epoch 216/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1023 - accuracy: 0.9000
    Epoch 217/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1371 - accuracy: 0.9000
    Epoch 218/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1023 - accuracy: 0.9000
    Epoch 219/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1371 - accuracy: 0.9000
    Epoch 220/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1023 - accuracy: 0.9000
    Epoch 221/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1370 - accuracy: 0.9000
    Epoch 222/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1022 - accuracy: 0.9000
    Epoch 223/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1370 - accuracy: 0.9000
    Epoch 224/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1022 - accuracy: 0.9000
    Epoch 225/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1369 - accuracy: 0.9000
    Epoch 226/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1021 - accuracy: 0.9000
    Epoch 227/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1369 - accuracy: 0.9000
    Epoch 228/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1021 - accuracy: 0.9000
    Epoch 229/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1368 - accuracy: 0.9000
    Epoch 230/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1021 - accuracy: 0.9000
    Epoch 231/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1368 - accuracy: 0.9000
    Epoch 232/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1020 - accuracy: 0.9000
    Epoch 233/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1367 - accuracy: 0.9000
    Epoch 234/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1020 - accuracy: 0.9000
    Epoch 235/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1366 - accuracy: 0.9000
    Epoch 236/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1019 - accuracy: 0.9000
    Epoch 237/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1366 - accuracy: 0.9000
    Epoch 238/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1019 - accuracy: 0.9000
    Epoch 239/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1365 - accuracy: 0.9000
    Epoch 240/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1019 - accuracy: 0.9000
    Epoch 241/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1365 - accuracy: 0.9000
    Epoch 242/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1018 - accuracy: 0.9000
    Epoch 243/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1364 - accuracy: 0.9000
    Epoch 244/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1018 - accuracy: 0.9000
    Epoch 245/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1364 - accuracy: 0.9000
    Epoch 246/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1018 - accuracy: 0.9000
    Epoch 247/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1363 - accuracy: 0.9000
    Epoch 248/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1017 - accuracy: 0.9000
    Epoch 249/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1363 - accuracy: 0.9000
    Epoch 250/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1017 - accuracy: 0.9000
    Epoch 251/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1362 - accuracy: 0.9000
    Epoch 252/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1016 - accuracy: 0.9000
    Epoch 253/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1362 - accuracy: 0.9000
    Epoch 254/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1016 - accuracy: 0.9000
    Epoch 255/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1361 - accuracy: 0.9000
    Epoch 256/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1016 - accuracy: 0.9000
    Epoch 257/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1361 - accuracy: 0.9000
    Epoch 258/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1015 - accuracy: 0.9000
    Epoch 259/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1360 - accuracy: 0.9000
    Epoch 260/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1015 - accuracy: 0.9000
    Epoch 261/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1360 - accuracy: 0.9000
    Epoch 262/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1015 - accuracy: 0.9000
    Epoch 263/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1359 - accuracy: 0.9000
    Epoch 264/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1014 - accuracy: 0.9000
    Epoch 265/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1359 - accuracy: 0.9000
    Epoch 266/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1014 - accuracy: 0.9000
    Epoch 267/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1358 - accuracy: 0.9000
    Epoch 268/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1013 - accuracy: 0.9000
    Epoch 269/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1358 - accuracy: 0.9000
    Epoch 270/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1013 - accuracy: 0.9000
    Epoch 271/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1357 - accuracy: 0.9000
    Epoch 272/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1013 - accuracy: 0.9000
    Epoch 273/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1357 - accuracy: 0.9000
    Epoch 274/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1012 - accuracy: 0.9000
    Epoch 275/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1356 - accuracy: 0.9000
    Epoch 276/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1012 - accuracy: 0.9000
    Epoch 277/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1355 - accuracy: 0.9000
    Epoch 278/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1012 - accuracy: 0.9000
    Epoch 279/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1355 - accuracy: 0.9000
    Epoch 280/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1011 - accuracy: 0.9000
    Epoch 281/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1354 - accuracy: 0.9000
    Epoch 282/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1011 - accuracy: 0.9000
    Epoch 283/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1354 - accuracy: 0.9000
    Epoch 284/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1010 - accuracy: 0.9000
    Epoch 285/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1353 - accuracy: 0.9000
    Epoch 286/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1010 - accuracy: 0.9000
    Epoch 287/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1353 - accuracy: 0.9000
    Epoch 288/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1010 - accuracy: 0.9000
    Epoch 289/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1352 - accuracy: 0.9000
    Epoch 290/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1009 - accuracy: 0.9000
    Epoch 291/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1352 - accuracy: 0.9000
    Epoch 292/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1009 - accuracy: 0.9000
    Epoch 293/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1351 - accuracy: 0.9000
    Epoch 294/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1009 - accuracy: 0.9000
    Epoch 295/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1351 - accuracy: 0.9000
    Epoch 296/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1008 - accuracy: 0.9000
    Epoch 297/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1350 - accuracy: 0.9000
    Epoch 298/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1008 - accuracy: 0.9000
    Epoch 299/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1350 - accuracy: 0.9000
    Epoch 300/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1007 - accuracy: 0.9000
    Epoch 301/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1349 - accuracy: 0.9000
    Epoch 302/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1007 - accuracy: 0.9000
    Epoch 303/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1349 - accuracy: 0.9000
    Epoch 304/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1007 - accuracy: 0.9000
    Epoch 305/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1348 - accuracy: 0.9000
    Epoch 306/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1006 - accuracy: 0.9000
    Epoch 307/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1348 - accuracy: 0.9000
    Epoch 308/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1006 - accuracy: 0.9000
    Epoch 309/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1347 - accuracy: 0.9000
    Epoch 310/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1006 - accuracy: 0.9000
    Epoch 311/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1347 - accuracy: 0.9000
    Epoch 312/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1005 - accuracy: 0.9000
    Epoch 313/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1346 - accuracy: 0.9000
    Epoch 314/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1005 - accuracy: 0.9000
    Epoch 315/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1346 - accuracy: 0.9000
    Epoch 316/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1005 - accuracy: 0.9000
    Epoch 317/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1345 - accuracy: 0.9000
    Epoch 318/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1004 - accuracy: 0.9000
    Epoch 319/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1345 - accuracy: 0.9000
    Epoch 320/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1004 - accuracy: 0.9000
    Epoch 321/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1344 - accuracy: 0.9000
    Epoch 322/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1003 - accuracy: 0.9000
    Epoch 323/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1344 - accuracy: 0.9000
    Epoch 324/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1003 - accuracy: 0.9000
    Epoch 325/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1343 - accuracy: 0.9000
    Epoch 326/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1003 - accuracy: 0.9000
    Epoch 327/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1343 - accuracy: 0.9000
    Epoch 328/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1002 - accuracy: 0.9000
    Epoch 329/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1342 - accuracy: 0.9000
    Epoch 330/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1002 - accuracy: 0.9000
    Epoch 331/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1342 - accuracy: 0.9000
    Epoch 332/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1002 - accuracy: 0.9000
    Epoch 333/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1341 - accuracy: 0.9000
    Epoch 334/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1001 - accuracy: 0.9000
    Epoch 335/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1341 - accuracy: 0.9000
    Epoch 336/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1001 - accuracy: 0.9000
    Epoch 337/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1340 - accuracy: 0.9000
    Epoch 338/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1001 - accuracy: 0.9000
    Epoch 339/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1340 - accuracy: 0.9000
    Epoch 340/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1000 - accuracy: 0.9000
    Epoch 341/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1339 - accuracy: 0.9000
    Epoch 342/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1000 - accuracy: 0.9000
    Epoch 343/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1339 - accuracy: 0.9000
    Epoch 344/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0999 - accuracy: 0.9000
    Epoch 345/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1338 - accuracy: 0.9000
    Epoch 346/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0999 - accuracy: 0.9000
    Epoch 347/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1338 - accuracy: 0.9000
    Epoch 348/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0999 - accuracy: 0.9000
    Epoch 349/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1337 - accuracy: 0.9000
    Epoch 350/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0998 - accuracy: 0.9000
    Epoch 351/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1337 - accuracy: 0.9000
    Epoch 352/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0998 - accuracy: 0.9000
    Epoch 353/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1336 - accuracy: 0.9000
    Epoch 354/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0998 - accuracy: 0.9000
    Epoch 355/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1336 - accuracy: 0.9000
    Epoch 356/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0997 - accuracy: 0.9000
    Epoch 357/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1335 - accuracy: 0.9000
    Epoch 358/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0997 - accuracy: 0.9000
    Epoch 359/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1334 - accuracy: 0.9000
    Epoch 360/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0997 - accuracy: 0.9000
    Epoch 361/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1334 - accuracy: 0.9000
    Epoch 362/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0996 - accuracy: 0.9000
    Epoch 363/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1333 - accuracy: 0.9000
    Epoch 364/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0996 - accuracy: 0.9000
    Epoch 365/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1333 - accuracy: 0.9000
    Epoch 366/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0996 - accuracy: 0.9000
    Epoch 367/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1332 - accuracy: 0.9000
    Epoch 368/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0995 - accuracy: 0.9000
    Epoch 369/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1332 - accuracy: 0.9000
    Epoch 370/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0995 - accuracy: 0.9000
    Epoch 371/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1331 - accuracy: 0.9000
    Epoch 372/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0995 - accuracy: 0.9000
    Epoch 373/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1331 - accuracy: 0.9000
    Epoch 374/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0994 - accuracy: 0.9000
    Epoch 375/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1330 - accuracy: 0.9000
    Epoch 376/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0994 - accuracy: 0.9000
    Epoch 377/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1330 - accuracy: 0.9000
    Epoch 378/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0993 - accuracy: 0.9000
    Epoch 379/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1329 - accuracy: 0.9000
    Epoch 380/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0993 - accuracy: 0.9000
    Epoch 381/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1329 - accuracy: 0.9000
    Epoch 382/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0993 - accuracy: 0.9000
    Epoch 383/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1328 - accuracy: 0.9000
    Epoch 384/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0992 - accuracy: 0.9000
    Epoch 385/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1328 - accuracy: 0.9000
    Epoch 386/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0992 - accuracy: 0.9000
    Epoch 387/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1327 - accuracy: 0.9000
    Epoch 388/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0992 - accuracy: 0.9000
    Epoch 389/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1327 - accuracy: 0.9000
    Epoch 390/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0991 - accuracy: 0.9000
    Epoch 391/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1326 - accuracy: 0.9000
    Epoch 392/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0991 - accuracy: 0.9000
    Epoch 393/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1326 - accuracy: 0.9000
    Epoch 394/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0991 - accuracy: 0.9000
    Epoch 395/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1325 - accuracy: 0.9000
    Epoch 396/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0990 - accuracy: 0.9000
    Epoch 397/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1325 - accuracy: 0.9000
    Epoch 398/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0990 - accuracy: 0.9000
    Epoch 399/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1325 - accuracy: 0.9000
    Epoch 400/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0990 - accuracy: 0.9000
    Epoch 401/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1324 - accuracy: 0.9000
    Epoch 402/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0989 - accuracy: 0.9000
    Epoch 403/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1324 - accuracy: 0.9000
    Epoch 404/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0989 - accuracy: 0.9000
    Epoch 405/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1323 - accuracy: 0.9000
    Epoch 406/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0989 - accuracy: 0.9000
    Epoch 407/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1323 - accuracy: 0.9000
    Epoch 408/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0988 - accuracy: 0.9000
    Epoch 409/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1322 - accuracy: 0.9000
    Epoch 410/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0988 - accuracy: 0.9000
    Epoch 411/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1322 - accuracy: 0.9000
    Epoch 412/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0988 - accuracy: 0.9000
    Epoch 413/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1321 - accuracy: 0.9000
    Epoch 414/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0987 - accuracy: 0.9000
    Epoch 415/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1321 - accuracy: 0.9000
    Epoch 416/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0987 - accuracy: 0.9000
    Epoch 417/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1320 - accuracy: 0.9000
    Epoch 418/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0987 - accuracy: 0.9000
    Epoch 419/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1320 - accuracy: 0.9000
    Epoch 420/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0986 - accuracy: 0.9000
    Epoch 421/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1319 - accuracy: 0.9000
    Epoch 422/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0986 - accuracy: 0.9000
    Epoch 423/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1319 - accuracy: 0.9000
    Epoch 424/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0985 - accuracy: 0.9000
    Epoch 425/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1318 - accuracy: 0.9000
    Epoch 426/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0985 - accuracy: 0.9000
    Epoch 427/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1318 - accuracy: 0.9000
    Epoch 428/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0985 - accuracy: 0.9000
    Epoch 429/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1317 - accuracy: 0.9000
    Epoch 430/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0984 - accuracy: 0.9000
    Epoch 431/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1317 - accuracy: 0.9000
    Epoch 432/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0984 - accuracy: 0.9000
    Epoch 433/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1316 - accuracy: 0.9000
    Epoch 434/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0984 - accuracy: 0.9000
    Epoch 435/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1316 - accuracy: 0.9000
    Epoch 436/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0983 - accuracy: 0.9000
    Epoch 437/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1315 - accuracy: 0.9000
    Epoch 438/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0983 - accuracy: 0.9000
    Epoch 439/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1315 - accuracy: 0.9000
    Epoch 440/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0983 - accuracy: 0.9000
    Epoch 441/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1314 - accuracy: 0.9000
    Epoch 442/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0982 - accuracy: 0.9000
    Epoch 443/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1314 - accuracy: 0.9000
    Epoch 444/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0982 - accuracy: 0.9000
    Epoch 445/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1313 - accuracy: 0.9000
    Epoch 446/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0982 - accuracy: 0.9000
    Epoch 447/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1313 - accuracy: 0.9000
    Epoch 448/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0981 - accuracy: 1.0000
    Epoch 449/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1312 - accuracy: 0.9000
    Epoch 450/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0981 - accuracy: 1.0000
    Epoch 451/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1312 - accuracy: 0.9000
    Epoch 452/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0981 - accuracy: 1.0000
    Epoch 453/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1311 - accuracy: 0.9000
    Epoch 454/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0980 - accuracy: 1.0000
    Epoch 455/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1311 - accuracy: 0.9000
    Epoch 456/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0980 - accuracy: 1.0000
    Epoch 457/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1310 - accuracy: 0.9000
    Epoch 458/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0980 - accuracy: 1.0000
    Epoch 459/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1310 - accuracy: 0.9000
    Epoch 460/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0979 - accuracy: 1.0000
    Epoch 461/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1309 - accuracy: 0.9000
    Epoch 462/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0979 - accuracy: 1.0000
    Epoch 463/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1309 - accuracy: 0.9000
    Epoch 464/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0979 - accuracy: 1.0000
    Epoch 465/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1308 - accuracy: 0.9000
    Epoch 466/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0978 - accuracy: 1.0000
    Epoch 467/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1308 - accuracy: 0.9000
    Epoch 468/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0978 - accuracy: 1.0000
    Epoch 469/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1307 - accuracy: 0.9000
    Epoch 470/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0978 - accuracy: 1.0000
    Epoch 471/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1307 - accuracy: 0.9000
    Epoch 472/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0977 - accuracy: 1.0000
    Epoch 473/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1306 - accuracy: 0.9000
    Epoch 474/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0977 - accuracy: 1.0000
    Epoch 475/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1306 - accuracy: 0.9000
    Epoch 476/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0977 - accuracy: 1.0000
    Epoch 477/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1305 - accuracy: 0.9000
    Epoch 478/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0976 - accuracy: 1.0000
    Epoch 479/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1305 - accuracy: 0.9000
    Epoch 480/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0976 - accuracy: 1.0000
    Epoch 481/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1304 - accuracy: 0.9000
    Epoch 482/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0976 - accuracy: 1.0000
    Epoch 483/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1304 - accuracy: 0.9000
    Epoch 484/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0975 - accuracy: 1.0000
    Epoch 485/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1303 - accuracy: 0.9000
    Epoch 486/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0975 - accuracy: 1.0000
    Epoch 487/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1303 - accuracy: 0.9000
    Epoch 488/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0975 - accuracy: 1.0000
    Epoch 489/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1303 - accuracy: 0.9000
    Epoch 490/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0974 - accuracy: 1.0000
    Epoch 491/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1302 - accuracy: 0.9000
    Epoch 492/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0974 - accuracy: 1.0000
    Epoch 493/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1302 - accuracy: 0.9000
    Epoch 494/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0974 - accuracy: 1.0000
    Epoch 495/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1301 - accuracy: 0.9000
    Epoch 496/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0973 - accuracy: 1.0000
    Epoch 497/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1301 - accuracy: 0.9000
    Epoch 498/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0973 - accuracy: 1.0000
    Epoch 499/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1300 - accuracy: 0.9000
    Epoch 500/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0973 - accuracy: 1.0000





    <keras.callbacks.History at 0x2b6e40e99e10>




```python
test_data = np.array([0.5, 3.0, 3.5, 11.0, 13.0, 31.0])

sigmoid_value = model.predict(test_data)

logical_value = tf.cast(sigmoid_value > 0.5, dtype = tf.float32)

for i in range(len(test_data)):
    print(test_data[i],
         sigmoid_value[i],
         logical_value.numpy()[i])
```

    1/1 [==============================] - 0s 15ms/step
    0.5 [0.00139493] [0.]
    3.0 [0.00097443] [0.]
    3.5 [0.0010512] [0.]
    11.0 [0.29125124] [0.]
    13.0 [0.8618062] [1.]
    31.0 [0.98304623] [1.]



```python
model.fit(x_data, t_data, epochs = 500)
```

    Epoch 1/500
    1/1 [==============================] - 0s 4ms/step - loss: 0.1300 - accuracy: 0.9000
    Epoch 2/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0972 - accuracy: 1.0000
    Epoch 3/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1299 - accuracy: 0.9000
    Epoch 4/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0972 - accuracy: 1.0000
    Epoch 5/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1299 - accuracy: 0.9000
    Epoch 6/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0972 - accuracy: 1.0000
    Epoch 7/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1298 - accuracy: 0.9000
    Epoch 8/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0971 - accuracy: 1.0000
    Epoch 9/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1298 - accuracy: 0.9000
    Epoch 10/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0971 - accuracy: 1.0000
    Epoch 11/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1297 - accuracy: 0.9000
    Epoch 12/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0971 - accuracy: 1.0000
    Epoch 13/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1297 - accuracy: 0.9000
    Epoch 14/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0970 - accuracy: 1.0000
    Epoch 15/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1296 - accuracy: 0.9000
    Epoch 16/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0970 - accuracy: 1.0000
    Epoch 17/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1296 - accuracy: 0.9000
    Epoch 18/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0970 - accuracy: 1.0000
    Epoch 19/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1295 - accuracy: 0.9000
    Epoch 20/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0969 - accuracy: 1.0000
    Epoch 21/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1295 - accuracy: 0.9000
    Epoch 22/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0969 - accuracy: 1.0000
    Epoch 23/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1294 - accuracy: 0.9000
    Epoch 24/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0969 - accuracy: 1.0000
    Epoch 25/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1294 - accuracy: 0.9000
    Epoch 26/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0968 - accuracy: 1.0000
    Epoch 27/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1293 - accuracy: 0.9000
    Epoch 28/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0968 - accuracy: 1.0000
    Epoch 29/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1293 - accuracy: 0.9000
    Epoch 30/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0968 - accuracy: 1.0000
    Epoch 31/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1293 - accuracy: 0.9000
    Epoch 32/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0968 - accuracy: 1.0000
    Epoch 33/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1292 - accuracy: 0.9000
    Epoch 34/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0967 - accuracy: 1.0000
    Epoch 35/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1292 - accuracy: 0.9000
    Epoch 36/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0967 - accuracy: 1.0000
    Epoch 37/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1291 - accuracy: 0.9000
    Epoch 38/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0967 - accuracy: 1.0000
    Epoch 39/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1291 - accuracy: 0.9000
    Epoch 40/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0966 - accuracy: 1.0000
    Epoch 41/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1290 - accuracy: 0.9000
    Epoch 42/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0966 - accuracy: 1.0000
    Epoch 43/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1290 - accuracy: 0.9000
    Epoch 44/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0966 - accuracy: 1.0000
    Epoch 45/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1289 - accuracy: 0.9000
    Epoch 46/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0965 - accuracy: 1.0000
    Epoch 47/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1289 - accuracy: 0.9000
    Epoch 48/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0965 - accuracy: 1.0000
    Epoch 49/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1288 - accuracy: 0.9000
    Epoch 50/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0965 - accuracy: 1.0000
    Epoch 51/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1288 - accuracy: 0.9000
    Epoch 52/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0964 - accuracy: 1.0000
    Epoch 53/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1287 - accuracy: 0.9000
    Epoch 54/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0964 - accuracy: 1.0000
    Epoch 55/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1287 - accuracy: 0.9000
    Epoch 56/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0964 - accuracy: 1.0000
    Epoch 57/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1286 - accuracy: 0.9000
    Epoch 58/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0963 - accuracy: 1.0000
    Epoch 59/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1286 - accuracy: 0.9000
    Epoch 60/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0963 - accuracy: 1.0000
    Epoch 61/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1285 - accuracy: 0.9000
    Epoch 62/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0963 - accuracy: 1.0000
    Epoch 63/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1285 - accuracy: 0.9000
    Epoch 64/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0962 - accuracy: 1.0000
    Epoch 65/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1284 - accuracy: 0.9000
    Epoch 66/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0962 - accuracy: 1.0000
    Epoch 67/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1284 - accuracy: 0.9000
    Epoch 68/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0962 - accuracy: 1.0000
    Epoch 69/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1284 - accuracy: 0.9000
    Epoch 70/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0961 - accuracy: 1.0000
    Epoch 71/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1283 - accuracy: 0.9000
    Epoch 72/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0961 - accuracy: 1.0000
    Epoch 73/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1283 - accuracy: 0.9000
    Epoch 74/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0961 - accuracy: 1.0000
    Epoch 75/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1282 - accuracy: 0.9000
    Epoch 76/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0960 - accuracy: 1.0000
    Epoch 77/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1282 - accuracy: 0.9000
    Epoch 78/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0960 - accuracy: 1.0000
    Epoch 79/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1281 - accuracy: 0.9000
    Epoch 80/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0960 - accuracy: 1.0000
    Epoch 81/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1281 - accuracy: 0.9000
    Epoch 82/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0960 - accuracy: 1.0000
    Epoch 83/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1280 - accuracy: 0.9000
    Epoch 84/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0959 - accuracy: 1.0000
    Epoch 85/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1280 - accuracy: 0.9000
    Epoch 86/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0959 - accuracy: 1.0000
    Epoch 87/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1279 - accuracy: 0.9000
    Epoch 88/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0959 - accuracy: 1.0000
    Epoch 89/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1279 - accuracy: 0.9000
    Epoch 90/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0958 - accuracy: 1.0000
    Epoch 91/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1278 - accuracy: 0.9000
    Epoch 92/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0958 - accuracy: 1.0000
    Epoch 93/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1278 - accuracy: 0.9000
    Epoch 94/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0958 - accuracy: 1.0000
    Epoch 95/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1277 - accuracy: 0.9000
    Epoch 96/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0957 - accuracy: 1.0000
    Epoch 97/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1277 - accuracy: 0.9000
    Epoch 98/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0957 - accuracy: 1.0000
    Epoch 99/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1277 - accuracy: 0.9000
    Epoch 100/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0957 - accuracy: 1.0000
    Epoch 101/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1276 - accuracy: 0.9000
    Epoch 102/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0956 - accuracy: 1.0000
    Epoch 103/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1276 - accuracy: 0.9000
    Epoch 104/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0956 - accuracy: 1.0000
    Epoch 105/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1275 - accuracy: 0.9000
    Epoch 106/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0956 - accuracy: 1.0000
    Epoch 107/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1275 - accuracy: 0.9000
    Epoch 108/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0955 - accuracy: 1.0000
    Epoch 109/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1274 - accuracy: 0.9000
    Epoch 110/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0955 - accuracy: 1.0000
    Epoch 111/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1274 - accuracy: 0.9000
    Epoch 112/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0955 - accuracy: 1.0000
    Epoch 113/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1273 - accuracy: 0.9000
    Epoch 114/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0955 - accuracy: 1.0000
    Epoch 115/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1273 - accuracy: 0.9000
    Epoch 116/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0954 - accuracy: 1.0000
    Epoch 117/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1272 - accuracy: 0.9000
    Epoch 118/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0954 - accuracy: 1.0000
    Epoch 119/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1272 - accuracy: 0.9000
    Epoch 120/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0954 - accuracy: 1.0000
    Epoch 121/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1271 - accuracy: 0.9000
    Epoch 122/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0953 - accuracy: 1.0000
    Epoch 123/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1271 - accuracy: 0.9000
    Epoch 124/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0953 - accuracy: 1.0000
    Epoch 125/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1271 - accuracy: 0.9000
    Epoch 126/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0953 - accuracy: 1.0000
    Epoch 127/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1270 - accuracy: 0.9000
    Epoch 128/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0952 - accuracy: 1.0000
    Epoch 129/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1270 - accuracy: 0.9000
    Epoch 130/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0952 - accuracy: 1.0000
    Epoch 131/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1269 - accuracy: 0.9000
    Epoch 132/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0952 - accuracy: 1.0000
    Epoch 133/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1269 - accuracy: 0.9000
    Epoch 134/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0951 - accuracy: 1.0000
    Epoch 135/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1268 - accuracy: 0.9000
    Epoch 136/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0951 - accuracy: 1.0000
    Epoch 137/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1268 - accuracy: 0.9000
    Epoch 138/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0951 - accuracy: 1.0000
    Epoch 139/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1267 - accuracy: 0.9000
    Epoch 140/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0951 - accuracy: 1.0000
    Epoch 141/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1267 - accuracy: 0.9000
    Epoch 142/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0950 - accuracy: 1.0000
    Epoch 143/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1266 - accuracy: 0.9000
    Epoch 144/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0950 - accuracy: 1.0000
    Epoch 145/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1266 - accuracy: 0.9000
    Epoch 146/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0950 - accuracy: 1.0000
    Epoch 147/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1265 - accuracy: 0.9000
    Epoch 148/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0949 - accuracy: 1.0000
    Epoch 149/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1265 - accuracy: 0.9000
    Epoch 150/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0949 - accuracy: 1.0000
    Epoch 151/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1265 - accuracy: 0.9000
    Epoch 152/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0949 - accuracy: 1.0000
    Epoch 153/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1264 - accuracy: 0.9000
    Epoch 154/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0948 - accuracy: 1.0000
    Epoch 155/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1264 - accuracy: 0.9000
    Epoch 156/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0948 - accuracy: 1.0000
    Epoch 157/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1263 - accuracy: 0.9000
    Epoch 158/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0948 - accuracy: 1.0000
    Epoch 159/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1263 - accuracy: 0.9000
    Epoch 160/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0947 - accuracy: 1.0000
    Epoch 161/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1262 - accuracy: 0.9000
    Epoch 162/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0947 - accuracy: 1.0000
    Epoch 163/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1262 - accuracy: 0.9000
    Epoch 164/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0947 - accuracy: 1.0000
    Epoch 165/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1261 - accuracy: 0.9000
    Epoch 166/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0947 - accuracy: 1.0000
    Epoch 167/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1261 - accuracy: 0.9000
    Epoch 168/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0946 - accuracy: 1.0000
    Epoch 169/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1260 - accuracy: 0.9000
    Epoch 170/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0946 - accuracy: 1.0000
    Epoch 171/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1260 - accuracy: 0.9000
    Epoch 172/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0946 - accuracy: 1.0000
    Epoch 173/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1260 - accuracy: 0.9000
    Epoch 174/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0945 - accuracy: 1.0000
    Epoch 175/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1259 - accuracy: 0.9000
    Epoch 176/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0945 - accuracy: 1.0000
    Epoch 177/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1259 - accuracy: 0.9000
    Epoch 178/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0945 - accuracy: 1.0000
    Epoch 179/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1258 - accuracy: 0.9000
    Epoch 180/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0944 - accuracy: 1.0000
    Epoch 181/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1258 - accuracy: 0.9000
    Epoch 182/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0944 - accuracy: 1.0000
    Epoch 183/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1257 - accuracy: 0.9000
    Epoch 184/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0944 - accuracy: 1.0000
    Epoch 185/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1257 - accuracy: 0.9000
    Epoch 186/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0944 - accuracy: 1.0000
    Epoch 187/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1256 - accuracy: 0.9000
    Epoch 188/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0943 - accuracy: 1.0000
    Epoch 189/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1256 - accuracy: 0.9000
    Epoch 190/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0943 - accuracy: 1.0000
    Epoch 191/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1255 - accuracy: 0.9000
    Epoch 192/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0943 - accuracy: 1.0000
    Epoch 193/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1255 - accuracy: 0.9000
    Epoch 194/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0942 - accuracy: 1.0000
    Epoch 195/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1255 - accuracy: 0.9000
    Epoch 196/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0942 - accuracy: 1.0000
    Epoch 197/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1254 - accuracy: 0.9000
    Epoch 198/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0942 - accuracy: 1.0000
    Epoch 199/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1254 - accuracy: 0.9000
    Epoch 200/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0941 - accuracy: 1.0000
    Epoch 201/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1253 - accuracy: 0.9000
    Epoch 202/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0941 - accuracy: 1.0000
    Epoch 203/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1253 - accuracy: 0.9000
    Epoch 204/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0941 - accuracy: 1.0000
    Epoch 205/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1252 - accuracy: 0.9000
    Epoch 206/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0941 - accuracy: 1.0000
    Epoch 207/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1252 - accuracy: 0.9000
    Epoch 208/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0940 - accuracy: 1.0000
    Epoch 209/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1251 - accuracy: 0.9000
    Epoch 210/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0940 - accuracy: 1.0000
    Epoch 211/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1251 - accuracy: 0.9000
    Epoch 212/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0940 - accuracy: 1.0000
    Epoch 213/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1250 - accuracy: 0.9000
    Epoch 214/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0939 - accuracy: 1.0000
    Epoch 215/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1250 - accuracy: 0.9000
    Epoch 216/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0939 - accuracy: 1.0000
    Epoch 217/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1250 - accuracy: 0.9000
    Epoch 218/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0939 - accuracy: 1.0000
    Epoch 219/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1249 - accuracy: 0.9000
    Epoch 220/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0939 - accuracy: 1.0000
    Epoch 221/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1249 - accuracy: 0.9000
    Epoch 222/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0938 - accuracy: 1.0000
    Epoch 223/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1248 - accuracy: 0.9000
    Epoch 224/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0938 - accuracy: 1.0000
    Epoch 225/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1248 - accuracy: 0.9000
    Epoch 226/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0938 - accuracy: 1.0000
    Epoch 227/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1247 - accuracy: 0.9000
    Epoch 228/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0937 - accuracy: 1.0000
    Epoch 229/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1247 - accuracy: 0.9000
    Epoch 230/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0937 - accuracy: 1.0000
    Epoch 231/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1246 - accuracy: 0.9000
    Epoch 232/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0937 - accuracy: 1.0000
    Epoch 233/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1246 - accuracy: 0.9000
    Epoch 234/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0936 - accuracy: 1.0000
    Epoch 235/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1246 - accuracy: 0.9000
    Epoch 236/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0936 - accuracy: 1.0000
    Epoch 237/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1245 - accuracy: 0.9000
    Epoch 238/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0936 - accuracy: 1.0000
    Epoch 239/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1245 - accuracy: 0.9000
    Epoch 240/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0936 - accuracy: 1.0000
    Epoch 241/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1244 - accuracy: 0.9000
    Epoch 242/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0935 - accuracy: 1.0000
    Epoch 243/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1244 - accuracy: 0.9000
    Epoch 244/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0935 - accuracy: 1.0000
    Epoch 245/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1243 - accuracy: 0.9000
    Epoch 246/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0935 - accuracy: 1.0000
    Epoch 247/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1243 - accuracy: 0.9000
    Epoch 248/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0934 - accuracy: 1.0000
    Epoch 249/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1242 - accuracy: 0.9000
    Epoch 250/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0934 - accuracy: 1.0000
    Epoch 251/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1242 - accuracy: 0.9000
    Epoch 252/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0934 - accuracy: 1.0000
    Epoch 253/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1242 - accuracy: 0.9000
    Epoch 254/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0934 - accuracy: 1.0000
    Epoch 255/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1241 - accuracy: 0.9000
    Epoch 256/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0933 - accuracy: 1.0000
    Epoch 257/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1241 - accuracy: 0.9000
    Epoch 258/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0933 - accuracy: 1.0000
    Epoch 259/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1240 - accuracy: 0.9000
    Epoch 260/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0933 - accuracy: 1.0000
    Epoch 261/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1240 - accuracy: 0.9000
    Epoch 262/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0932 - accuracy: 1.0000
    Epoch 263/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1239 - accuracy: 0.9000
    Epoch 264/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0932 - accuracy: 1.0000
    Epoch 265/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1239 - accuracy: 0.9000
    Epoch 266/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0932 - accuracy: 1.0000
    Epoch 267/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1238 - accuracy: 0.9000
    Epoch 268/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0932 - accuracy: 1.0000
    Epoch 269/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1238 - accuracy: 0.9000
    Epoch 270/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0931 - accuracy: 1.0000
    Epoch 271/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1238 - accuracy: 0.9000
    Epoch 272/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0931 - accuracy: 1.0000
    Epoch 273/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1237 - accuracy: 0.9000
    Epoch 274/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0931 - accuracy: 1.0000
    Epoch 275/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1237 - accuracy: 0.9000
    Epoch 276/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0930 - accuracy: 1.0000
    Epoch 277/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1236 - accuracy: 0.9000
    Epoch 278/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0930 - accuracy: 1.0000
    Epoch 279/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1236 - accuracy: 0.9000
    Epoch 280/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0930 - accuracy: 1.0000
    Epoch 281/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1235 - accuracy: 0.9000
    Epoch 282/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0930 - accuracy: 1.0000
    Epoch 283/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1235 - accuracy: 0.9000
    Epoch 284/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0929 - accuracy: 1.0000
    Epoch 285/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1234 - accuracy: 0.9000
    Epoch 286/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0929 - accuracy: 1.0000
    Epoch 287/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1234 - accuracy: 0.9000
    Epoch 288/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0929 - accuracy: 1.0000
    Epoch 289/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1234 - accuracy: 0.9000
    Epoch 290/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0928 - accuracy: 1.0000
    Epoch 291/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1233 - accuracy: 0.9000
    Epoch 292/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0928 - accuracy: 1.0000
    Epoch 293/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1233 - accuracy: 0.9000
    Epoch 294/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0928 - accuracy: 1.0000
    Epoch 295/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1232 - accuracy: 0.9000
    Epoch 296/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0928 - accuracy: 1.0000
    Epoch 297/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1232 - accuracy: 0.9000
    Epoch 298/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0927 - accuracy: 1.0000
    Epoch 299/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1231 - accuracy: 0.9000
    Epoch 300/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0927 - accuracy: 1.0000
    Epoch 301/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1231 - accuracy: 0.9000
    Epoch 302/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0927 - accuracy: 1.0000
    Epoch 303/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1230 - accuracy: 0.9000
    Epoch 304/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0926 - accuracy: 1.0000
    Epoch 305/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1230 - accuracy: 0.9000
    Epoch 306/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0926 - accuracy: 1.0000
    Epoch 307/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1230 - accuracy: 0.9000
    Epoch 308/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0926 - accuracy: 1.0000
    Epoch 309/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1229 - accuracy: 0.9000
    Epoch 310/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0926 - accuracy: 1.0000
    Epoch 311/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1229 - accuracy: 0.9000
    Epoch 312/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0925 - accuracy: 1.0000
    Epoch 313/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1228 - accuracy: 0.9000
    Epoch 314/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0925 - accuracy: 1.0000
    Epoch 315/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1228 - accuracy: 0.9000
    Epoch 316/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0925 - accuracy: 1.0000
    Epoch 317/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1227 - accuracy: 0.9000
    Epoch 318/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0924 - accuracy: 1.0000
    Epoch 319/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1227 - accuracy: 0.9000
    Epoch 320/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0924 - accuracy: 1.0000
    Epoch 321/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1227 - accuracy: 0.9000
    Epoch 322/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0924 - accuracy: 1.0000
    Epoch 323/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1226 - accuracy: 0.9000
    Epoch 324/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0924 - accuracy: 1.0000
    Epoch 325/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1226 - accuracy: 0.9000
    Epoch 326/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0923 - accuracy: 1.0000
    Epoch 327/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1225 - accuracy: 0.9000
    Epoch 328/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0923 - accuracy: 1.0000
    Epoch 329/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1225 - accuracy: 0.9000
    Epoch 330/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0923 - accuracy: 1.0000
    Epoch 331/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1224 - accuracy: 0.9000
    Epoch 332/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0922 - accuracy: 1.0000
    Epoch 333/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1224 - accuracy: 0.9000
    Epoch 334/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0922 - accuracy: 1.0000
    Epoch 335/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1223 - accuracy: 0.9000
    Epoch 336/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0922 - accuracy: 1.0000
    Epoch 337/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1223 - accuracy: 0.9000
    Epoch 338/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0922 - accuracy: 1.0000
    Epoch 339/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1223 - accuracy: 0.9000
    Epoch 340/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0921 - accuracy: 1.0000
    Epoch 341/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1222 - accuracy: 0.9000
    Epoch 342/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0921 - accuracy: 1.0000
    Epoch 343/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1222 - accuracy: 0.9000
    Epoch 344/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0921 - accuracy: 1.0000
    Epoch 345/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1221 - accuracy: 0.9000
    Epoch 346/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0920 - accuracy: 1.0000
    Epoch 347/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1221 - accuracy: 0.9000
    Epoch 348/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0920 - accuracy: 1.0000
    Epoch 349/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1220 - accuracy: 0.9000
    Epoch 350/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0920 - accuracy: 1.0000
    Epoch 351/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1220 - accuracy: 0.9000
    Epoch 352/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0920 - accuracy: 1.0000
    Epoch 353/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1220 - accuracy: 0.9000
    Epoch 354/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0919 - accuracy: 1.0000
    Epoch 355/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1219 - accuracy: 0.9000
    Epoch 356/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0919 - accuracy: 1.0000
    Epoch 357/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1219 - accuracy: 0.9000
    Epoch 358/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0919 - accuracy: 1.0000
    Epoch 359/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1218 - accuracy: 0.9000
    Epoch 360/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0918 - accuracy: 1.0000
    Epoch 361/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1218 - accuracy: 0.9000
    Epoch 362/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0918 - accuracy: 1.0000
    Epoch 363/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1217 - accuracy: 0.9000
    Epoch 364/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0918 - accuracy: 1.0000
    Epoch 365/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1217 - accuracy: 0.9000
    Epoch 366/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0918 - accuracy: 1.0000
    Epoch 367/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1216 - accuracy: 0.9000
    Epoch 368/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0917 - accuracy: 1.0000
    Epoch 369/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1216 - accuracy: 0.9000
    Epoch 370/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0917 - accuracy: 1.0000
    Epoch 371/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1216 - accuracy: 0.9000
    Epoch 372/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0917 - accuracy: 1.0000
    Epoch 373/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1215 - accuracy: 0.9000
    Epoch 374/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0917 - accuracy: 1.0000
    Epoch 375/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1215 - accuracy: 0.9000
    Epoch 376/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0916 - accuracy: 1.0000
    Epoch 377/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1214 - accuracy: 0.9000
    Epoch 378/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0916 - accuracy: 1.0000
    Epoch 379/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1214 - accuracy: 0.9000
    Epoch 380/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0916 - accuracy: 1.0000
    Epoch 381/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1213 - accuracy: 0.9000
    Epoch 382/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0915 - accuracy: 1.0000
    Epoch 383/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1213 - accuracy: 0.9000
    Epoch 384/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0915 - accuracy: 1.0000
    Epoch 385/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1213 - accuracy: 0.9000
    Epoch 386/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0915 - accuracy: 1.0000
    Epoch 387/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1212 - accuracy: 0.9000
    Epoch 388/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0915 - accuracy: 1.0000
    Epoch 389/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1212 - accuracy: 0.9000
    Epoch 390/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0914 - accuracy: 1.0000
    Epoch 391/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1211 - accuracy: 0.9000
    Epoch 392/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0914 - accuracy: 1.0000
    Epoch 393/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1211 - accuracy: 0.9000
    Epoch 394/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0914 - accuracy: 1.0000
    Epoch 395/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1210 - accuracy: 0.9000
    Epoch 396/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0914 - accuracy: 1.0000
    Epoch 397/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1210 - accuracy: 0.9000
    Epoch 398/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0913 - accuracy: 1.0000
    Epoch 399/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1210 - accuracy: 0.9000
    Epoch 400/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0913 - accuracy: 1.0000
    Epoch 401/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1209 - accuracy: 0.9000
    Epoch 402/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0913 - accuracy: 1.0000
    Epoch 403/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1209 - accuracy: 0.9000
    Epoch 404/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0912 - accuracy: 1.0000
    Epoch 405/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1208 - accuracy: 0.9000
    Epoch 406/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0912 - accuracy: 1.0000
    Epoch 407/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1208 - accuracy: 0.9000
    Epoch 408/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0912 - accuracy: 1.0000
    Epoch 409/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1207 - accuracy: 0.9000
    Epoch 410/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0912 - accuracy: 1.0000
    Epoch 411/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1207 - accuracy: 0.9000
    Epoch 412/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0911 - accuracy: 1.0000
    Epoch 413/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1207 - accuracy: 0.9000
    Epoch 414/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0911 - accuracy: 1.0000
    Epoch 415/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1206 - accuracy: 0.9000
    Epoch 416/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0911 - accuracy: 1.0000
    Epoch 417/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1206 - accuracy: 0.9000
    Epoch 418/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0911 - accuracy: 1.0000
    Epoch 419/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1205 - accuracy: 0.9000
    Epoch 420/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0910 - accuracy: 1.0000
    Epoch 421/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1205 - accuracy: 0.9000
    Epoch 422/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0910 - accuracy: 1.0000
    Epoch 423/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1204 - accuracy: 0.9000
    Epoch 424/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0910 - accuracy: 1.0000
    Epoch 425/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1204 - accuracy: 0.9000
    Epoch 426/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0909 - accuracy: 1.0000
    Epoch 427/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1204 - accuracy: 0.9000
    Epoch 428/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0909 - accuracy: 1.0000
    Epoch 429/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1203 - accuracy: 0.9000
    Epoch 430/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0909 - accuracy: 1.0000
    Epoch 431/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1203 - accuracy: 0.9000
    Epoch 432/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0909 - accuracy: 1.0000
    Epoch 433/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1202 - accuracy: 0.9000
    Epoch 434/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0908 - accuracy: 1.0000
    Epoch 435/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1202 - accuracy: 0.9000
    Epoch 436/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0908 - accuracy: 1.0000
    Epoch 437/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1201 - accuracy: 0.9000
    Epoch 438/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0908 - accuracy: 1.0000
    Epoch 439/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1201 - accuracy: 0.9000
    Epoch 440/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0908 - accuracy: 1.0000
    Epoch 441/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1201 - accuracy: 0.9000
    Epoch 442/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0907 - accuracy: 1.0000
    Epoch 443/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1200 - accuracy: 0.9000
    Epoch 444/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0907 - accuracy: 1.0000
    Epoch 445/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1200 - accuracy: 0.9000
    Epoch 446/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0907 - accuracy: 1.0000
    Epoch 447/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1199 - accuracy: 0.9000
    Epoch 448/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0906 - accuracy: 1.0000
    Epoch 449/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1199 - accuracy: 0.9000
    Epoch 450/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0906 - accuracy: 1.0000
    Epoch 451/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1198 - accuracy: 0.9000
    Epoch 452/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0906 - accuracy: 1.0000
    Epoch 453/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1198 - accuracy: 0.9000
    Epoch 454/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0906 - accuracy: 1.0000
    Epoch 455/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1198 - accuracy: 0.9000
    Epoch 456/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0905 - accuracy: 1.0000
    Epoch 457/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1197 - accuracy: 0.9000
    Epoch 458/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0905 - accuracy: 1.0000
    Epoch 459/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1197 - accuracy: 0.9000
    Epoch 460/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0905 - accuracy: 1.0000
    Epoch 461/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1196 - accuracy: 0.9000
    Epoch 462/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0905 - accuracy: 1.0000
    Epoch 463/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1196 - accuracy: 0.9000
    Epoch 464/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0904 - accuracy: 1.0000
    Epoch 465/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1195 - accuracy: 0.9000
    Epoch 466/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0904 - accuracy: 1.0000
    Epoch 467/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1195 - accuracy: 0.9000
    Epoch 468/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0904 - accuracy: 1.0000
    Epoch 469/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1195 - accuracy: 0.9000
    Epoch 470/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0904 - accuracy: 1.0000
    Epoch 471/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1194 - accuracy: 0.9000
    Epoch 472/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0903 - accuracy: 1.0000
    Epoch 473/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1194 - accuracy: 0.9000
    Epoch 474/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0903 - accuracy: 1.0000
    Epoch 475/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1193 - accuracy: 0.9000
    Epoch 476/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0903 - accuracy: 1.0000
    Epoch 477/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1193 - accuracy: 0.9000
    Epoch 478/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0902 - accuracy: 1.0000
    Epoch 479/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1192 - accuracy: 0.9000
    Epoch 480/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0902 - accuracy: 1.0000
    Epoch 481/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1192 - accuracy: 0.9000
    Epoch 482/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0902 - accuracy: 1.0000
    Epoch 483/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1192 - accuracy: 0.9000
    Epoch 484/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0902 - accuracy: 1.0000
    Epoch 485/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1191 - accuracy: 0.9000
    Epoch 486/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0901 - accuracy: 1.0000
    Epoch 487/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1191 - accuracy: 0.9000
    Epoch 488/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0901 - accuracy: 1.0000
    Epoch 489/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1190 - accuracy: 0.9000
    Epoch 490/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0901 - accuracy: 1.0000
    Epoch 491/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1190 - accuracy: 0.9000
    Epoch 492/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0901 - accuracy: 1.0000
    Epoch 493/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1189 - accuracy: 0.9000
    Epoch 494/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.0900 - accuracy: 1.0000
    Epoch 495/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1189 - accuracy: 0.9000
    Epoch 496/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0900 - accuracy: 1.0000
    Epoch 497/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.1189 - accuracy: 0.9000
    Epoch 498/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0900 - accuracy: 1.0000
    Epoch 499/500
    1/1 [==============================] - 0s 2ms/step - loss: 0.1188 - accuracy: 0.9000
    Epoch 500/500
    1/1 [==============================] - 0s 3ms/step - loss: 0.0900 - accuracy: 1.0000





    <keras.callbacks.History at 0x2b6e40ed0c90>



# Deep Learning MNIST Example

[1] 데이터 불러오기 및 확인


```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, t_train), (x_test, t_test) = mnist.load_data()

print('₩n train shape = ', x_train.shape,
     ', train label shape = ', t_train.shape)

print(' test shape = ', x_test.shape,
     ', test label shape =', t_test.shape)

print('₩n train label =', t_train) # 학습데이터 정답 출력
print(' test label = ', t_test) # 테스트 데이터 정답 출력
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11490434/11490434 [==============================] - 1s 0us/step
    ₩n train shape =  (60000, 28, 28) , train label shape =  (60000,)
     test shape =  (10000, 28, 28) , test label shape = (10000,)
    ₩n train label = [5 0 4 ... 5 6 8]
     test label =  [7 2 1 ... 4 5 6]



```python
import matplotlib.pyplot as plt

# 25개의 이미지 출력
plt.figure(figsize = (10,10))

for index in range(25):
    
    plt.subplot(5, 5, index + 1)
    plt.imshow(x_train[index], cmap = 'gray')
    plt.axis('off')
    #plt.title(str(t_train[index]))
    
plt.show()
```


    
![png](output_19_0.png)
    

[2] 데이터 전처리 (정규화, 원핫 인코딩)

```python
# 학습 데이터 / 테스트 데이터 정규화 (Normalization)

x_train = (x_train - 0.0) / (255.0 - 0.0)

x_test= (x_test - 0.0) / (255.0 - 0.0)

# 정답 데이터 원핫 인코딩 (One-Hot Encoding)
# MNIST 정답 데이터는 0~9까지 총 10개의 숫자 가운데 하나이므로, num_classes = 10 지정하여 10개의 리스트를 만들어서 원핫 인코딩을 수행함.

t_train = tf.keras.utils.to_categorical(t_train, num_classes = 10)

t_test = tf.keras.utils.to_categorical(t_test, num_classes = 10)
```

[3] 모델 구축 및 컴파일


```python
model = tf.keras.Sequential()

# 28 x 28 크기 2차원 이미지를 784개의 1차원 벡터로 변환
model.add(tf.keras.layers.Flatten(input_shape = (28,28)))

model.add(tf.keras.layers.Dense(100, activation = 'relu'))

model.add(tf.keras.layers.Dense(10, activation = 'softmax'))
```


```python
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])

model.summary()
```

    Model: "sequential_9"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_6 (Flatten)         (None, 784)               0         
                                                                     
     dense_13 (Dense)            (None, 100)               78500     
                                                                     
     dense_14 (Dense)            (None, 10)                1010      
                                                                     
    =================================================================
    Total params: 79,510
    Trainable params: 79,510
    Non-trainable params: 0
    _________________________________________________________________



```python
hist = model.fit(x_train, t_train, epochs = 30, validation_split = 0.3)
```

    Epoch 1/30
    1313/1313 [==============================] - 3s 2ms/step - loss: 0.0038 - accuracy: 0.9989 - val_loss: 0.1741 - val_accuracy: 0.9731
    Epoch 2/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 0.0045 - accuracy: 0.9984 - val_loss: 0.1707 - val_accuracy: 0.9741
    Epoch 3/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 0.0030 - accuracy: 0.9991 - val_loss: 0.1677 - val_accuracy: 0.9736
    Epoch 4/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 0.0035 - accuracy: 0.9988 - val_loss: 0.1761 - val_accuracy: 0.9734
    Epoch 5/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 0.0034 - accuracy: 0.9990 - val_loss: 0.1933 - val_accuracy: 0.9706
    Epoch 6/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 0.0021 - accuracy: 0.9994 - val_loss: 0.1653 - val_accuracy: 0.9747
    Epoch 7/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 0.0036 - accuracy: 0.9986 - val_loss: 0.2026 - val_accuracy: 0.9701
    Epoch 8/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 0.0025 - accuracy: 0.9993 - val_loss: 0.1776 - val_accuracy: 0.9738
    Epoch 9/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 0.0061 - accuracy: 0.9982 - val_loss: 0.1761 - val_accuracy: 0.9741
    Epoch 10/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 0.0021 - accuracy: 0.9994 - val_loss: 0.1670 - val_accuracy: 0.9756
    Epoch 11/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 6.5975e-04 - accuracy: 0.9997 - val_loss: 0.1856 - val_accuracy: 0.9719
    Epoch 12/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 0.0056 - accuracy: 0.9980 - val_loss: 0.1747 - val_accuracy: 0.9736
    Epoch 13/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 0.0010 - accuracy: 0.9997 - val_loss: 0.1890 - val_accuracy: 0.9731
    Epoch 14/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 0.0047 - accuracy: 0.9986 - val_loss: 0.2082 - val_accuracy: 0.9697
    Epoch 15/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 0.0019 - accuracy: 0.9994 - val_loss: 0.2218 - val_accuracy: 0.9692
    Epoch 16/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 0.0033 - accuracy: 0.9989 - val_loss: 0.1885 - val_accuracy: 0.9738
    Epoch 17/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 7.1703e-04 - accuracy: 0.9998 - val_loss: 0.1819 - val_accuracy: 0.9748
    Epoch 18/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 5.7642e-05 - accuracy: 1.0000 - val_loss: 0.1818 - val_accuracy: 0.9749
    Epoch 19/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 3.7758e-05 - accuracy: 1.0000 - val_loss: 0.1830 - val_accuracy: 0.9752
    Epoch 20/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 2.8296e-05 - accuracy: 1.0000 - val_loss: 0.1845 - val_accuracy: 0.9757
    Epoch 21/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 2.3998e-05 - accuracy: 1.0000 - val_loss: 0.1910 - val_accuracy: 0.9756
    Epoch 22/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 0.0088 - accuracy: 0.9973 - val_loss: 0.1989 - val_accuracy: 0.9736
    Epoch 23/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 0.0021 - accuracy: 0.9994 - val_loss: 0.2157 - val_accuracy: 0.9724
    Epoch 24/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 0.0029 - accuracy: 0.9990 - val_loss: 0.2177 - val_accuracy: 0.9725
    Epoch 25/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 0.0019 - accuracy: 0.9993 - val_loss: 0.2002 - val_accuracy: 0.9741
    Epoch 26/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 0.0046 - accuracy: 0.9988 - val_loss: 0.2062 - val_accuracy: 0.9733
    Epoch 27/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 4.3011e-04 - accuracy: 0.9999 - val_loss: 0.2179 - val_accuracy: 0.9724
    Epoch 28/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 0.0033 - accuracy: 0.9990 - val_loss: 0.2265 - val_accuracy: 0.9714
    Epoch 29/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 0.0022 - accuracy: 0.9993 - val_loss: 0.2132 - val_accuracy: 0.9726
    Epoch 30/30
    1313/1313 [==============================] - 2s 2ms/step - loss: 0.0028 - accuracy: 0.9990 - val_loss: 0.2273 - val_accuracy: 0.9703



```python
model.evaluate(x_test, t_test)
```

    313/313 [==============================] - 0s 848us/step - loss: 0.1868 - accuracy: 0.9749





    [0.18676701188087463, 0.9749000072479248]



[6] 손실 및 정확도


```python
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'], label = 'train loss')
plt.plot(hist.history['val_loss'], label = 'validation loss')

plt.legend(loc = 'best')

plt.show()
```


    
![png](output_28_0.png)
    



```python
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()

plt.plot(hist.history['accuracy'], label = 'train accuracy')
plt.plot(hist.history['val_accuracy'], label = 'validation accuracy')

plt.legend(loc = 'best')

plt.show()
```


    
![png](output_29_0.png)
    


[7] 혼돈 행렬 (confusion matrix)

- 구축한 모델의 강점과 약점, 즉 어떤 데이터에 대해서 모델이 혼란스러워 하는지 등을 파악할 수 있음
- from sklearn.metrics import confusion_matrix


```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure(figsize = (10, 10))

predicted_value = model.predict(x_test)

# cmap = confusion_matrix(...) : 여기서 cmap은 혼동 행렬을 저장할 변수이다

# np.argmax(t_test, axis = -1) : t_test는 테스트 데이터의 실제 레이블(타겟)이다.
# 만약 레이블이 원핫 인코딩이 되어 있다면, np.argmax(t_test, axis = -1)은 각 레이블 벡터에서 가장 큰 값의 인덱스를 찾아 해당 클래스의 인덱스를 반환한다.
# np.argmax(predicted_value, axis = -1) : predicted_value는 모델이 예측한 값이다.
# 예측 값이 원핫 인코딩 또는 클래스별 확률일 경우, np.argmax(predicted_value, axis = -1)은 각 예측 벡터에서 가장 큰 값의 인덱스를 찾아 모델이 예측한 클래스를 반환한다.

# confusion_matrix(...) : 이 함수는 실제 레이블과 예측 레이블을 입력으로 받아 혼동 행렬을 계산합니다.
# 혼동 행렬은 정답과 예측이 얼마나 일치하는지 나타내는 행렬로, 각 행은 실제 클래스, 각 열은 예측 클래스에 해당한다.
# 행렬의 각 요소 (i, j)는 실제 클래스 i가 예측 클래스 j로 분류된 횟수를 나타낸다.

cmap = confusion_matrix(np.argmax(t_test, axis = -1),
                           np.argmax(predicted_value, axis = -1))

sns.heatmap(cmap, annot = True, fmt = 'd')

plt.show()
```

    313/313 [==============================] - 0s 695us/step



    
![png](output_31_1.png)
    



```python
print(cmap)
```

    [[ 967    0    1    1    0    1    3    2    4    1]
     [   0 1123    2    5    0    1    2    1    1    0]
     [   2    3  998    5    1    0    3   10    7    3]
     [   0    0    4  980    1   10    0    4    1   10]
     [   2    0    3    1  946    0    6    4    2   18]
     [   1    0    0    7    0  870    9    1    3    1]
     [   7    4    1    2    7    6  929    1    1    0]
     [   0    5    7    5    0    0    0 1005    1    5]
     [   2    1    2    6    3    4    2    3  947    4]
     [   2    2    0    4    6    3    0    6    2  984]]



```python
print('₩n')


# i : 현재 클래스 레이블
# np.max(cmap[i]) : 현재 클래스에서 가장 많이 예측된 수
# np.sum(cmap[i]) : 현재 클래스의 총 예측 수
# np.max(cmap[i])/np.sum(cmap[i]) : 현재 클래스의 정확도(가장 많이 예측된 수 / 총 예측된 수).

for i in range(10):
    print(('label = %d (%d/%d) accuracy = %.3f') %
          (i, np.max(cmap[i]), np.sum(cmap[i]),
          np.max(cmap[i])/np.sum(cmap[i])))
```

    ₩n
    label = 0 (967/980) accuracy = 0.987
    label = 1 (1123/1135) accuracy = 0.989
    label = 2 (998/1032) accuracy = 0.967
    label = 3 (980/1010) accuracy = 0.970
    label = 4 (946/982) accuracy = 0.963
    label = 5 (870/892) accuracy = 0.975
    label = 6 (929/958) accuracy = 0.970
    label = 7 (1005/1028) accuracy = 0.978
    label = 8 (947/974) accuracy = 0.972
    label = 9 (984/1009) accuracy = 0.975


# Fashion MNIST Example


```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

(x_train, t_train), (x_test, t_test) = fashion_mnist.load_data()

print('train shape = ', x_train.shape,
     ', train label shape = ', t_train.shape)

print('test shape = ', x_test.shape,
     ', test label shape = ', t_test.shape)

print('train label = ', t_train) # 학습데이터 정답 출력
print('test label = ', t_test) # 테스트 데이터 정답 출력
```

    train shape =  (60000, 28, 28) , train label shape =  (60000,)
    test shape =  (10000, 28, 28) , test label shape =  (10000,)
    train label =  [9 0 0 ... 3 0 5]
    test label =  [9 2 1 ... 8 1 5]



```python
import matplotlib.pyplot as plt

# 25개의 이미지 출력
plt.figure(figsize=(10,10))

for index in range(25):   # 25 개의 이미지 출력
    
    plt.subplot(5, 5, index + 1) # 5행 5열
    plt.imshow(x_train[index], cmap='gray')
    plt.axis('off')
    
plt.show()
```


    
![png](output_36_0.png)
    



```python
# 학습 데이터 / 테스트 데이터 정규화 (Normalization)

x_train = (x_train - 0.0) / (255.0 - 0.0)

x_test= (x_test - 0.0) / (255.0 - 0.0)

# 정답 데이터 원핫 인코딩 (One-Hot Encoding) 은 수행하지 않음
# MNIST 정답 데이터는 0~9까지 총 10개의 숫자 가운데 하나이므로, num_classes = 10 지정하여 10개의 리스트를 만들어서 원핫 인코딩을 수행함.

# t_train = tf.keras.utils.to_categorical(t_train, num_classes = 10)

# t_test = tf.keras.utils.to_categorical(t_test, num_classes = 10)
```


```python
model = tf.keras.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

model.add(tf.keras.layers.Dense(100, activation = 'relu'))

model.add(tf.keras.layers.Dense(10, activation = 'softmax'))
```


```python
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

model.summary()
```

    Model: "sequential_13"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_9 (Flatten)         (None, 784)               0         
                                                                     
     dense_19 (Dense)            (None, 100)               78500     
                                                                     
     dense_20 (Dense)            (None, 10)                1010      
                                                                     
    =================================================================
    Total params: 79,510
    Trainable params: 79,510
    Non-trainable params: 0
    _________________________________________________________________



```python
hist = model.fit(x_train, t_train, epochs = 50, validation_split = 0.5)
```

    Epoch 1/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0519 - accuracy: 0.9818 - val_loss: 0.5141 - val_accuracy: 0.9019
    Epoch 2/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0461 - accuracy: 0.9827 - val_loss: 0.5394 - val_accuracy: 0.9004
    Epoch 3/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0487 - accuracy: 0.9824 - val_loss: 0.5315 - val_accuracy: 0.8994
    Epoch 4/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0516 - accuracy: 0.9804 - val_loss: 0.5288 - val_accuracy: 0.8986
    Epoch 5/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0432 - accuracy: 0.9846 - val_loss: 0.5711 - val_accuracy: 0.8985
    Epoch 6/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0451 - accuracy: 0.9838 - val_loss: 0.5588 - val_accuracy: 0.8991
    Epoch 7/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0444 - accuracy: 0.9833 - val_loss: 0.5799 - val_accuracy: 0.8936
    Epoch 8/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0442 - accuracy: 0.9839 - val_loss: 0.5583 - val_accuracy: 0.9003
    Epoch 9/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0530 - accuracy: 0.9807 - val_loss: 0.5399 - val_accuracy: 0.9010
    Epoch 10/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0395 - accuracy: 0.9853 - val_loss: 0.5778 - val_accuracy: 0.8974
    Epoch 11/50
    938/938 [==============================] - 3s 3ms/step - loss: 0.0449 - accuracy: 0.9833 - val_loss: 0.5544 - val_accuracy: 0.8999
    Epoch 12/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0418 - accuracy: 0.9846 - val_loss: 0.5651 - val_accuracy: 0.8990
    Epoch 13/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0413 - accuracy: 0.9849 - val_loss: 0.5721 - val_accuracy: 0.9001
    Epoch 14/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0447 - accuracy: 0.9831 - val_loss: 0.5716 - val_accuracy: 0.8992
    Epoch 15/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0391 - accuracy: 0.9857 - val_loss: 0.6347 - val_accuracy: 0.8961
    Epoch 16/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0418 - accuracy: 0.9845 - val_loss: 0.6657 - val_accuracy: 0.8960
    Epoch 17/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0405 - accuracy: 0.9856 - val_loss: 0.6277 - val_accuracy: 0.8984
    Epoch 18/50
    938/938 [==============================] - 4s 4ms/step - loss: 0.0419 - accuracy: 0.9845 - val_loss: 0.5964 - val_accuracy: 0.8990
    Epoch 19/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0404 - accuracy: 0.9854 - val_loss: 0.6214 - val_accuracy: 0.8994
    Epoch 20/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0412 - accuracy: 0.9853 - val_loss: 0.7103 - val_accuracy: 0.8928
    Epoch 21/50
    938/938 [==============================] - 3s 3ms/step - loss: 0.0378 - accuracy: 0.9862 - val_loss: 0.6359 - val_accuracy: 0.8964
    Epoch 22/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0363 - accuracy: 0.9870 - val_loss: 0.6200 - val_accuracy: 0.8974
    Epoch 23/50
    938/938 [==============================] - 3s 3ms/step - loss: 0.0348 - accuracy: 0.9874 - val_loss: 0.6305 - val_accuracy: 0.9005
    Epoch 24/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0357 - accuracy: 0.9880 - val_loss: 0.6336 - val_accuracy: 0.8948
    Epoch 25/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0354 - accuracy: 0.9865 - val_loss: 0.6251 - val_accuracy: 0.9020
    Epoch 26/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0317 - accuracy: 0.9889 - val_loss: 0.6451 - val_accuracy: 0.8993
    Epoch 27/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0375 - accuracy: 0.9866 - val_loss: 0.7602 - val_accuracy: 0.8792
    Epoch 28/50
    938/938 [==============================] - 3s 3ms/step - loss: 0.0378 - accuracy: 0.9868 - val_loss: 0.6366 - val_accuracy: 0.8977
    Epoch 29/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0331 - accuracy: 0.9879 - val_loss: 0.6677 - val_accuracy: 0.8984
    Epoch 30/50
    938/938 [==============================] - 3s 3ms/step - loss: 0.0352 - accuracy: 0.9872 - val_loss: 0.6802 - val_accuracy: 0.8988
    Epoch 31/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0370 - accuracy: 0.9879 - val_loss: 0.6627 - val_accuracy: 0.8950
    Epoch 32/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0374 - accuracy: 0.9865 - val_loss: 0.6981 - val_accuracy: 0.8931
    Epoch 33/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0278 - accuracy: 0.9900 - val_loss: 0.6441 - val_accuracy: 0.8987
    Epoch 34/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0308 - accuracy: 0.9897 - val_loss: 0.6929 - val_accuracy: 0.8964
    Epoch 35/50
    938/938 [==============================] - 4s 4ms/step - loss: 0.0381 - accuracy: 0.9863 - val_loss: 0.6674 - val_accuracy: 0.8994
    Epoch 36/50
    938/938 [==============================] - 3s 3ms/step - loss: 0.0290 - accuracy: 0.9900 - val_loss: 0.7267 - val_accuracy: 0.8932
    Epoch 37/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0329 - accuracy: 0.9875 - val_loss: 0.6925 - val_accuracy: 0.8964
    Epoch 38/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0324 - accuracy: 0.9880 - val_loss: 0.8045 - val_accuracy: 0.8900
    Epoch 39/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0330 - accuracy: 0.9887 - val_loss: 0.6865 - val_accuracy: 0.8961
    Epoch 40/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0266 - accuracy: 0.9910 - val_loss: 0.7302 - val_accuracy: 0.8881
    Epoch 41/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0333 - accuracy: 0.9888 - val_loss: 0.7124 - val_accuracy: 0.8959
    Epoch 42/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0275 - accuracy: 0.9904 - val_loss: 0.7542 - val_accuracy: 0.8926
    Epoch 43/50
    938/938 [==============================] - 4s 4ms/step - loss: 0.0352 - accuracy: 0.9875 - val_loss: 0.7439 - val_accuracy: 0.8922
    Epoch 44/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0348 - accuracy: 0.9873 - val_loss: 0.7298 - val_accuracy: 0.8941
    Epoch 45/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0296 - accuracy: 0.9903 - val_loss: 0.7658 - val_accuracy: 0.8863
    Epoch 46/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0275 - accuracy: 0.9899 - val_loss: 0.7323 - val_accuracy: 0.8949
    Epoch 47/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0269 - accuracy: 0.9909 - val_loss: 0.7027 - val_accuracy: 0.8960
    Epoch 48/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0327 - accuracy: 0.9883 - val_loss: 0.7200 - val_accuracy: 0.8970
    Epoch 49/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0267 - accuracy: 0.9902 - val_loss: 0.7244 - val_accuracy: 0.8979
    Epoch 50/50
    938/938 [==============================] - 3s 4ms/step - loss: 0.0264 - accuracy: 0.9906 - val_loss: 0.7655 - val_accuracy: 0.8920



```python
model.evaluate(x_test, t_test)
```

    313/313 [==============================] - 0s 1ms/step - loss: 1.0306 - accuracy: 0.8738





    [1.030558466911316, 0.8737999796867371]




```python
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(hist.history['loss'], label = 'train loss')
plt.plot(hist.history['val_loss'], label = 'validation loss')

plt.legend(loc = 'best')

plt.show()
```


    
![png](output_42_0.png)
    



```python
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()

plt.plot(hist.history['accuracy'], label = 'train accuracy')
plt.plot(hist.history['val_accuracy'], label = 'validation accuracy')

plt.legend(loc = 'best')

plt.show()
```


    
![png](output_43_0.png)
    



```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure(figsize = (10, 10))

predicted_value = model.predict(x_test)

# cmap = confusion_matrix(...) : 여기서 cmap은 혼동 행렬을 저장할 변수이다

# np.argmax(predicted_value, axis = -1) : predicted_value는 모델이 예측한 값이다.
# 예측 값이 원핫 인코딩 또는 클래스별 확률일 경우, np.argmax(predicted_value, axis = -1)은 각 예측 벡터에서 가장 큰 값의 인덱스를 찾아 모델이 예측한 클래스를 반환한다.

# confusion_matrix(...) : 이 함수는 실제 레이블과 예측 레이블을 입력으로 받아 혼동 행렬을 계산합니다.
# 혼동 행렬은 정답과 예측이 얼마나 일치하는지 나타내는 행렬로, 각 행은 실제 클래스, 각 열은 예측 클래스에 해당한다.
# 행렬의 각 요소 (i, j)는 실제 클래스 i가 예측 클래스 j로 분류된 횟수를 나타낸다.

cmap = confusion_matrix(t_test,
                           np.argmax(predicted_value, axis = -1))

sns.heatmap(cmap, annot = True, fmt = 'd')

plt.show()
```

    313/313 [==============================] - 0s 1ms/step



    
![png](output_44_1.png)
    



```python
print(cmap)
```

    [[756   1  13  38   8   3 177   0   4   0]
     [  1 964   3  19   7   0   6   0   0   0]
     [ 16   2 769  13  94   0 104   1   1   0]
     [ 16  12   6 891  38   2  33   0   2   0]
     [  0   0 107  27 780   0  86   0   0   0]
     [  0   0   0   1   0 938   0  37   2  22]
     [ 76   0  59  35  55   0 770   0   4   1]
     [  0   0   0   0   0   8   0 974   1  17]
     [  7   0   8   9   4   1  12   4 955   0]
     [  0   0   0   0   0  11   1  47   0 941]]



```python
print('₩n')


# i : 현재 클래스 레이블
# np.max(cmap[i]) : 현재 클래스에서 가장 많이 예측된 수
# np.sum(cmap[i]) : 현재 클래스의 총 예측 수
# np.max(cmap[i])/np.sum(cmap[i]) : 현재 클래스의 정확도(가장 많이 예측된 수 / 총 예측된 수).

for i in range(10):
    print(('label = %d (%d/%d) accuracy = %.3f') %
          (i, np.max(cmap[i]), np.sum(cmap[i]),
          np.max(cmap[i])/np.sum(cmap[i])))
```

    ₩n
    label = 0 (756/1000) accuracy = 0.756
    label = 1 (964/1000) accuracy = 0.964
    label = 2 (769/1000) accuracy = 0.769
    label = 3 (891/1000) accuracy = 0.891
    label = 4 (780/1000) accuracy = 0.780
    label = 5 (938/1000) accuracy = 0.938
    label = 6 (770/1000) accuracy = 0.770
    label = 7 (974/1000) accuracy = 0.974
    label = 8 (955/1000) accuracy = 0.955
    label = 9 (941/1000) accuracy = 0.941



```python

```


```python

```


```python

```
