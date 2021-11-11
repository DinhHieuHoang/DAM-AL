# Baseline
While training take 17378MiB GPU

unet3d_attention input: [1, 128, 128, 128, 4]
inputs  Tensor("QueueInput/input_deque:0", shape=(1, 128, 128, 128, 4), dtype=float32, device=/device:CPU:0)
WARNING:tensorflow:From /home/dghan/3DUnet_attention/model.py:79: conv3d (from tensorflow.python.layers.convolutional) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.keras.layers.Conv3D` instead.
WARNING:tensorflow:From /home/dghan/envs/3DUnet_attention/lib/python3.7/site-packages/tensorflow_core/python/layers/convolutional.py:632: Layer.apply (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
Instructions for updating:
Please use `layer.__call__` method instead.
unet3d_attention/init_conv/Relu:0 (128, 128, 128, 16)
Residual bock downsampling  0      (128, 128, 128, 16)
Down Conv3D  0     (64, 64, 64, 32)
Residual bock downsampling  1      (64, 64, 64, 32)
Down Conv3D  1     (32, 32, 32, 64)
Residual bock downsampling  2      (32, 32, 32, 64)
Down Conv3D  2     (16, 16, 16, 128)
Residual bock downsampling  3      (16, 16, 16, 128)
Down Conv3D  3     (8, 8, 8, 256)
Residual bock downsampling  4      (8, 8, 8, 256)
Low level feature 1      (128, 128, 128, 64)
Low level feature 2      (64, 64, 64, 64)
High level feature 1 CFE         (32, 32, 32, 128)
High level feature 2 CFE         (16, 16, 16, 128)
High level feature 3 CFE         (8, 8, 8, 128)
High level features aspp concat  (32, 32, 32, 384)
WARNING:tensorflow:From /home/dghan/3DUnet_attention/attention.py:169: dense (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
Instructions for updating:
Use keras.layers.Dense instead.
High level features CA   (32, 32, 32, 384)
High level features conv         (32, 32, 32, 64)
High level features upsampling   (128, 128, 128, 64)
High level features SA   (128, 128, 128, 64)
Low level feature conv   (128, 128, 128, 64)
Low + High level feature         (128, 128, 128, 128)
final (128, 128, 128, 4)
[0317 00:09:07 @registry.py:129] unet3d_attention output: [1, 128, 128, 128, 4]

[0317 00:09:10 @model_utils.py:49] Trainable Variables: 
name                                                                  shape                    dim
--------------------------------------------------------------------  -------------------  -------
unet3d_attention/init_conv/kernel:0                                   [3, 3, 3, 4, 16]        1728
unet3d_attention/init_conv/bias:0                                     [16]                      16
unet3d_attention/init_conv/ins_norm/beta:0                            [16]                      16
unet3d_attention/init_conv/ins_norm/gamma:0                           [16]                      16
unet3d_attention/down0_conv_0/kernel:0                                [3, 3, 3, 16, 16]       6912
unet3d_attention/down0_conv_0/bias:0                                  [16]                      16
unet3d_attention/down0_conv_0/ins_norm/beta:0                         [16]                      16
unet3d_attention/down0_conv_0/ins_norm/gamma:0                        [16]                      16
unet3d_attention/down0_conv_1/kernel:0                                [3, 3, 3, 16, 16]       6912
unet3d_attention/down0_conv_1/bias:0                                  [16]                      16
unet3d_attention/down0_conv_1/ins_norm/beta:0                         [16]                      16
unet3d_attention/down0_conv_1/ins_norm/gamma:0                        [16]                      16
unet3d_attention/stride2conv0/kernel:0                                [3, 3, 3, 16, 32]      13824
unet3d_attention/stride2conv0/bias:0                                  [32]                      32
unet3d_attention/stride2conv0/ins_norm/beta:0                         [32]                      32
unet3d_attention/stride2conv0/ins_norm/gamma:0                        [32]                      32
unet3d_attention/down1_conv_0/kernel:0                                [3, 3, 3, 32, 32]      27648
unet3d_attention/down1_conv_0/bias:0                                  [32]                      32
unet3d_attention/down1_conv_0/ins_norm/beta:0                         [32]                      32
unet3d_attention/down1_conv_0/ins_norm/gamma:0                        [32]                      32
unet3d_attention/down1_conv_1/kernel:0                                [3, 3, 3, 32, 32]      27648
unet3d_attention/down1_conv_1/bias:0                                  [32]                      32
unet3d_attention/down1_conv_1/ins_norm/beta:0                         [32]                      32
unet3d_attention/down1_conv_1/ins_norm/gamma:0                        [32]                      32
unet3d_attention/stride2conv1/kernel:0                                [3, 3, 3, 32, 64]      55296
unet3d_attention/stride2conv1/bias:0                                  [64]                      64
unet3d_attention/stride2conv1/ins_norm/beta:0                         [64]                      64
unet3d_attention/stride2conv1/ins_norm/gamma:0                        [64]                      64
unet3d_attention/down2_conv_0/kernel:0                                [3, 3, 3, 64, 64]     110592
unet3d_attention/down2_conv_0/bias:0                                  [64]                      64
unet3d_attention/down2_conv_0/ins_norm/beta:0                         [64]                      64
unet3d_attention/down2_conv_0/ins_norm/gamma:0                        [64]                      64
unet3d_attention/down2_conv_1/kernel:0                                [3, 3, 3, 64, 64]     110592
unet3d_attention/down2_conv_1/bias:0                                  [64]                      64
unet3d_attention/down2_conv_1/ins_norm/beta:0                         [64]                      64
unet3d_attention/down2_conv_1/ins_norm/gamma:0                        [64]                      64
unet3d_attention/stride2conv2/kernel:0                                [3, 3, 3, 64, 128]    221184
unet3d_attention/stride2conv2/bias:0                                  [128]                    128
unet3d_attention/stride2conv2/ins_norm/beta:0                         [128]                    128
unet3d_attention/stride2conv2/ins_norm/gamma:0                        [128]                    128
unet3d_attention/down3_conv_0/kernel:0                                [3, 3, 3, 128, 128]   442368
unet3d_attention/down3_conv_0/bias:0                                  [128]                    128
unet3d_attention/down3_conv_0/ins_norm/beta:0                         [128]                    128
unet3d_attention/down3_conv_0/ins_norm/gamma:0                        [128]                    128
unet3d_attention/down3_conv_1/kernel:0                                [3, 3, 3, 128, 128]   442368
unet3d_attention/down3_conv_1/bias:0                                  [128]                    128
unet3d_attention/down3_conv_1/ins_norm/beta:0                         [128]                    128
unet3d_attention/down3_conv_1/ins_norm/gamma:0                        [128]                    128
unet3d_attention/stride2conv3/kernel:0                                [3, 3, 3, 128, 256]   884736
unet3d_attention/stride2conv3/bias:0                                  [256]                    256
unet3d_attention/stride2conv3/ins_norm/beta:0                         [256]                    256
unet3d_attention/stride2conv3/ins_norm/gamma:0                        [256]                    256
unet3d_attention/down4_conv_0/kernel:0                                [3, 3, 3, 256, 256]  1769472
unet3d_attention/down4_conv_0/bias:0                                  [256]                    256
unet3d_attention/down4_conv_0/ins_norm/beta:0                         [256]                    256
unet3d_attention/down4_conv_0/ins_norm/gamma:0                        [256]                    256
unet3d_attention/down4_conv_1/kernel:0                                [3, 3, 3, 256, 256]  1769472
unet3d_attention/down4_conv_1/bias:0                                  [256]                    256
unet3d_attention/down4_conv_1/ins_norm/beta:0                         [256]                    256
unet3d_attention/down4_conv_1/ins_norm/gamma:0                        [256]                    256
unet3d_attention/C1_conv/kernel:0                                     [3, 3, 3, 16, 64]      27648
unet3d_attention/C1_conv/bias:0                                       [64]                      64
unet3d_attention/C1_conv/ins_norm/beta:0                              [64]                      64
unet3d_attention/C1_conv/ins_norm/gamma:0                             [64]                      64
unet3d_attention/C2_conv/kernel:0                                     [3, 3, 3, 32, 64]      55296
unet3d_attention/C2_conv/bias:0                                       [64]                      64
unet3d_attention/C2_conv/ins_norm/beta:0                              [64]                      64
unet3d_attention/C2_conv/ins_norm/gamma:0                             [64]                      64
unet3d_attention/C3_cfe_cfe0/kernel:0                                 [1, 1, 1, 64, 32]       2048
unet3d_attention/C3_cfe_cfe0/ins_norm/beta:0                          [32]                      32
unet3d_attention/C3_cfe_cfe0/ins_norm/gamma:0                         [32]                      32
unet3d_attention/C3_cfe_cfe1_dilation/kernel:0                        [3, 3, 3, 64, 32]      55296
unet3d_attention/C3_cfe_cfe1_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C3_cfe_cfe1_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C3_cfe_cfe2_dilation/kernel:0                        [3, 3, 3, 64, 32]      55296
unet3d_attention/C3_cfe_cfe2_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C3_cfe_cfe2_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C3_cfe_cfe3_dilation/kernel:0                        [3, 3, 3, 64, 32]      55296
unet3d_attention/C3_cfe_cfe3_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C3_cfe_cfe3_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C4_cfe_cfe0/kernel:0                                 [1, 1, 1, 128, 32]      4096
unet3d_attention/C4_cfe_cfe0/ins_norm/beta:0                          [32]                      32
unet3d_attention/C4_cfe_cfe0/ins_norm/gamma:0                         [32]                      32
unet3d_attention/C4_cfe_cfe1_dilation/kernel:0                        [3, 3, 3, 128, 32]    110592
unet3d_attention/C4_cfe_cfe1_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C4_cfe_cfe1_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C4_cfe_cfe2_dilation/kernel:0                        [3, 3, 3, 128, 32]    110592
unet3d_attention/C4_cfe_cfe2_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C4_cfe_cfe2_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C4_cfe_cfe3_dilation/kernel:0                        [3, 3, 3, 128, 32]    110592
unet3d_attention/C4_cfe_cfe3_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C4_cfe_cfe3_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C5_cfe_cfe0/kernel:0                                 [1, 1, 1, 256, 32]      8192
unet3d_attention/C5_cfe_cfe0/ins_norm/beta:0                          [32]                      32
unet3d_attention/C5_cfe_cfe0/ins_norm/gamma:0                         [32]                      32
unet3d_attention/C5_cfe_cfe1_dilation/kernel:0                        [3, 3, 3, 256, 32]    221184
unet3d_attention/C5_cfe_cfe1_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C5_cfe_cfe1_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C5_cfe_cfe2_dilation/kernel:0                        [3, 3, 3, 256, 32]    221184
unet3d_attention/C5_cfe_cfe2_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C5_cfe_cfe2_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C5_cfe_cfe3_dilation/kernel:0                        [3, 3, 3, 256, 32]    221184
unet3d_attention/C5_cfe_cfe3_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C5_cfe_cfe3_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/up_conv1_C5_cfe_up4/kernel:0                         [3, 3, 3, 128, 128]   442368
unet3d_attention/up_conv1_C5_cfe_up4/bias:0                           [128]                    128
unet3d_attention/up_conv1_C5_cfe_up4/ins_norm/beta:0                  [128]                    128
unet3d_attention/up_conv1_C5_cfe_up4/ins_norm/gamma:0                 [128]                    128
unet3d_attention/up_conv1_C4_cfe_up2/kernel:0                         [3, 3, 3, 128, 128]   442368
unet3d_attention/up_conv1_C4_cfe_up2/bias:0                           [128]                    128
unet3d_attention/up_conv1_C4_cfe_up2/ins_norm/beta:0                  [128]                    128
unet3d_attention/up_conv1_C4_cfe_up2/ins_norm/gamma:0                 [128]                    128
unet3d_attention/C345_ChannelWiseAttention_withcpfe_dense_1/kernel:0  [384, 96]              36864
unet3d_attention/C345_ChannelWiseAttention_withcpfe_dense_1/bias:0    [96]                      96
unet3d_attention/C345_ChannelWiseAttention_withcpfe_dense_2/kernel:0  [96, 384]              36864
unet3d_attention/C345_ChannelWiseAttention_withcpfe_dense_2/bias:0    [384]                    384
unet3d_attention/C345_conv/kernel:0                                   [1, 1, 1, 384, 64]     24576
unet3d_attention/C345_conv/bias:0                                     [64]                      64
unet3d_attention/C345_conv/ins_norm/beta:0                            [64]                      64
unet3d_attention/C345_conv/ins_norm/gamma:0                           [64]                      64
unet3d_attention/up_conv1_C345_up4/kernel:0                           [3, 3, 3, 64, 64]     110592
unet3d_attention/up_conv1_C345_up4/bias:0                             [64]                      64
unet3d_attention/up_conv1_C345_up4/ins_norm/beta:0                    [64]                      64
unet3d_attention/up_conv1_C345_up4/ins_norm/gamma:0                   [64]                      64
unet3d_attention/spatial_attention_1_conv1/kernel:0                   [1, 9, 9, 64, 32]     165888
unet3d_attention/spatial_attention_1_conv1/bias:0                     [32]                      32
unet3d_attention/spatial_attention_1_conv1/ins_norm/beta:0            [32]                      32
unet3d_attention/spatial_attention_1_conv1/ins_norm/gamma:0           [32]                      32
unet3d_attention/spatial_attention_1_conv2/kernel:0                   [9, 1, 1, 32, 1]         288
unet3d_attention/spatial_attention_1_conv2/bias:0                     [1]                        1
unet3d_attention/spatial_attention_1_conv2/ins_norm/beta:0            [1]                        1
unet3d_attention/spatial_attention_1_conv2/ins_norm/gamma:0           [1]                        1
unet3d_attention/spatial_attention_2_conv1/kernel:0                   [9, 1, 9, 64, 32]     165888
unet3d_attention/spatial_attention_2_conv1/bias:0                     [32]                      32
unet3d_attention/spatial_attention_2_conv1/ins_norm/beta:0            [32]                      32
unet3d_attention/spatial_attention_2_conv1/ins_norm/gamma:0           [32]                      32
unet3d_attention/spatial_attention_2_conv2/kernel:0                   [1, 9, 1, 32, 1]         288
unet3d_attention/spatial_attention_2_conv2/bias:0                     [1]                        1
unet3d_attention/spatial_attention_2_conv2/ins_norm/beta:0            [1]                        1
unet3d_attention/spatial_attention_2_conv2/ins_norm/gamma:0           [1]                        1
unet3d_attention/spatial_attention_3_conv1/kernel:0                   [9, 9, 1, 64, 32]     165888
unet3d_attention/spatial_attention_3_conv1/bias:0                     [32]                      32
unet3d_attention/spatial_attention_3_conv1/ins_norm/beta:0            [32]                      32
unet3d_attention/spatial_attention_3_conv1/ins_norm/gamma:0           [32]                      32
unet3d_attention/spatial_attention_3_conv2/kernel:0                   [1, 1, 9, 32, 1]         288
unet3d_attention/spatial_attention_3_conv2/bias:0                     [1]                        1
unet3d_attention/spatial_attention_3_conv2/ins_norm/beta:0            [1]                        1
unet3d_attention/spatial_attention_3_conv2/ins_norm/gamma:0           [1]                        1
unet3d_attention/up_conv1_C2_up2/kernel:0                             [3, 3, 3, 64, 64]     110592
unet3d_attention/up_conv1_C2_up2/bias:0                               [64]                      64
unet3d_attention/up_conv1_C2_up2/ins_norm/beta:0                      [64]                      64
unet3d_attention/up_conv1_C2_up2/ins_norm/gamma:0                     [64]                      64
unet3d_attention/C12_conv/kernel:0                                    [3, 3, 3, 128, 64]    221184
unet3d_attention/C12_conv/bias:0                                      [64]                      64
unet3d_attention/C12_conv/ins_norm/beta:0                             [64]                      64
unet3d_attention/C12_conv/ins_norm/gamma:0                            [64]                      64
unet3d_attention/final/kernel:0                                       [3, 3, 3, 128, 4]      13824
unet3d_attention/final/bias:0                                         [4]                        4
Total #vars=158, #params=9094941, size=34.69MB

# RSU with pooling replaced by CNN
17378MiB

unet3d_attention input: [1, 128, 128, 128, 4]
inputs  Tensor("QueueInput/input_deque:0", shape=(1, 128, 128, 128, 4), dtype=float32, device=/device:CPU:0)
WARNING:tensorflow:From /home/dghan/3DUnet_attention/model.py:79: conv3d (from tensorflow.python.layers.convolutional) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.keras.layers.Conv3D` instead.
WARNING:tensorflow:From /home/dghan/envs/3DUnet_attention/lib/python3.7/site-packages/tensorflow_core/python/layers/convolutional.py:632: Layer.apply (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
Instructions for updating:
Please use `layer.__call__` method instead.
unet3d_attention/init_conv/Relu:0 (128, 128, 128, 16)
RSU at C12 with pooling replaced by conv3d
RSU  0      (128, 128, 128, 16)
Down Conv3D  0     (64, 64, 64, 32)
RSU  1      (64, 64, 64, 32)
Down Conv3D  1     (32, 32, 32, 64)
Unet downsampling  2      (32, 32, 32, 64)
Down Conv3D  2     (16, 16, 16, 128)
Unet downsampling  3      (16, 16, 16, 128)
Down Conv3D  3     (8, 8, 8, 256)
Unet downsampling  4      (8, 8, 8, 256)
Low level feature 1      (128, 128, 128, 64)
Low level feature 2      (64, 64, 64, 64)
High level feature 1 CFE         (32, 32, 32, 128)
High level feature 2 CFE         (16, 16, 16, 128)
High level feature 3 CFE         (8, 8, 8, 128)
High level features aspp concat  (32, 32, 32, 384)
WARNING:tensorflow:From /home/dghan/3DUnet_attention/attention.py:169: dense (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
Instructions for updating:
Use keras.layers.Dense instead.
High level features CA   (32, 32, 32, 384)
High level features conv         (32, 32, 32, 64)
High level features upsampling   (128, 128, 128, 64)
High level features SA   (128, 128, 128, 64)
Low level feature conv   (128, 128, 128, 64)
Low + High level feature         (128, 128, 128, 128)
final (128, 128, 128, 4)
[0317 00:18:24 @registry.py:129] unet3d_attention output: [1, 128, 128, 128, 4]

Trainable Variables: 
name                                                                  shape                    dim
--------------------------------------------------------------------  -------------------  -------
unet3d_attention/init_conv/kernel:0                                   [3, 3, 3, 4, 16]        1728
unet3d_attention/init_conv/bias:0                                     [16]                      16
unet3d_attention/init_conv/ins_norm/beta:0                            [16]                      16
unet3d_attention/init_conv/ins_norm/gamma:0                           [16]                      16
unet3d_attention/conv3d/kernel:0                                      [3, 3, 3, 16, 16]       6912
unet3d_attention/conv3d/ins_norm/beta:0                               [16]                      16
unet3d_attention/conv3d/ins_norm/gamma:0                              [16]                      16
unet3d_attention/conv3d_1/kernel:0                                    [3, 3, 3, 16, 8]        3456
unet3d_attention/conv3d_1/ins_norm/beta:0                             [8]                        8
unet3d_attention/conv3d_1/ins_norm/gamma:0                            [8]                        8
unet3d_attention/conv3d_2/kernel:0                                    [2, 2, 2, 8, 8]          512
unet3d_attention/conv3d_2/bias:0                                      [8]                        8
unet3d_attention/conv3d_3/kernel:0                                    [3, 3, 3, 8, 8]         1728
unet3d_attention/conv3d_3/ins_norm/beta:0                             [8]                        8
unet3d_attention/conv3d_3/ins_norm/gamma:0                            [8]                        8
unet3d_attention/conv3d_4/kernel:0                                    [2, 2, 2, 8, 8]          512
unet3d_attention/conv3d_4/bias:0                                      [8]                        8
unet3d_attention/conv3d_5/kernel:0                                    [3, 3, 3, 8, 8]         1728
unet3d_attention/conv3d_5/ins_norm/beta:0                             [8]                        8
unet3d_attention/conv3d_5/ins_norm/gamma:0                            [8]                        8
unet3d_attention/conv3d_6/kernel:0                                    [2, 2, 2, 8, 8]          512
unet3d_attention/conv3d_6/bias:0                                      [8]                        8
unet3d_attention/conv3d_7/kernel:0                                    [3, 3, 3, 8, 8]         1728
unet3d_attention/conv3d_7/ins_norm/beta:0                             [8]                        8
unet3d_attention/conv3d_7/ins_norm/gamma:0                            [8]                        8
unet3d_attention/conv3d_8/kernel:0                                    [2, 2, 2, 8, 8]          512
unet3d_attention/conv3d_8/bias:0                                      [8]                        8
unet3d_attention/conv3d_9/kernel:0                                    [3, 3, 3, 8, 8]         1728
unet3d_attention/conv3d_9/ins_norm/beta:0                             [8]                        8
unet3d_attention/conv3d_9/ins_norm/gamma:0                            [8]                        8
unet3d_attention/conv3d_10/kernel:0                                   [2, 2, 2, 8, 8]          512
unet3d_attention/conv3d_10/bias:0                                     [8]                        8
unet3d_attention/conv3d_11/kernel:0                                   [3, 3, 3, 8, 8]         1728
unet3d_attention/conv3d_11/ins_norm/beta:0                            [8]                        8
unet3d_attention/conv3d_11/ins_norm/gamma:0                           [8]                        8
unet3d_attention/conv3d_12/kernel:0                                   [2, 2, 2, 8, 8]          512
unet3d_attention/conv3d_12/bias:0                                     [8]                        8
unet3d_attention/conv3d_13/kernel:0                                   [3, 3, 3, 8, 8]         1728
unet3d_attention/conv3d_13/ins_norm/beta:0                            [8]                        8
unet3d_attention/conv3d_13/ins_norm/gamma:0                           [8]                        8
unet3d_attention/conv3d_14/kernel:0                                   [3, 3, 3, 16, 8]        3456
unet3d_attention/conv3d_14/ins_norm/beta:0                            [8]                        8
unet3d_attention/conv3d_14/ins_norm/gamma:0                           [8]                        8
unet3d_attention/conv3d_15/kernel:0                                   [3, 3, 3, 16, 8]        3456
unet3d_attention/conv3d_15/ins_norm/beta:0                            [8]                        8
unet3d_attention/conv3d_15/ins_norm/gamma:0                           [8]                        8
unet3d_attention/conv3d_16/kernel:0                                   [3, 3, 3, 16, 8]        3456
unet3d_attention/conv3d_16/ins_norm/beta:0                            [8]                        8
unet3d_attention/conv3d_16/ins_norm/gamma:0                           [8]                        8
unet3d_attention/conv3d_17/kernel:0                                   [3, 3, 3, 16, 8]        3456
unet3d_attention/conv3d_17/ins_norm/beta:0                            [8]                        8
unet3d_attention/conv3d_17/ins_norm/gamma:0                           [8]                        8
unet3d_attention/conv3d_18/kernel:0                                   [3, 3, 3, 16, 8]        3456
unet3d_attention/conv3d_18/ins_norm/beta:0                            [8]                        8
unet3d_attention/conv3d_18/ins_norm/gamma:0                           [8]                        8
unet3d_attention/conv3d_19/kernel:0                                   [3, 3, 3, 16, 16]       6912
unet3d_attention/conv3d_19/ins_norm/beta:0                            [16]                      16
unet3d_attention/conv3d_19/ins_norm/gamma:0                           [16]                      16
unet3d_attention/stride2conv0/kernel:0                                [3, 3, 3, 16, 32]      13824
unet3d_attention/stride2conv0/bias:0                                  [32]                      32
unet3d_attention/stride2conv0/ins_norm/beta:0                         [32]                      32
unet3d_attention/stride2conv0/ins_norm/gamma:0                        [32]                      32
unet3d_attention/conv3d_20/kernel:0                                   [3, 3, 3, 32, 32]      27648
unet3d_attention/conv3d_20/ins_norm/beta:0                            [32]                      32
unet3d_attention/conv3d_20/ins_norm/gamma:0                           [32]                      32
unet3d_attention/conv3d_21/kernel:0                                   [3, 3, 3, 32, 8]        6912
unet3d_attention/conv3d_21/ins_norm/beta:0                            [8]                        8
unet3d_attention/conv3d_21/ins_norm/gamma:0                           [8]                        8
unet3d_attention/conv3d_22/kernel:0                                   [2, 2, 2, 8, 8]          512
unet3d_attention/conv3d_22/bias:0                                     [8]                        8
unet3d_attention/conv3d_23/kernel:0                                   [3, 3, 3, 8, 8]         1728
unet3d_attention/conv3d_23/ins_norm/beta:0                            [8]                        8
unet3d_attention/conv3d_23/ins_norm/gamma:0                           [8]                        8
unet3d_attention/conv3d_24/kernel:0                                   [2, 2, 2, 8, 8]          512
unet3d_attention/conv3d_24/bias:0                                     [8]                        8
unet3d_attention/conv3d_25/kernel:0                                   [3, 3, 3, 8, 8]         1728
unet3d_attention/conv3d_25/ins_norm/beta:0                            [8]                        8
unet3d_attention/conv3d_25/ins_norm/gamma:0                           [8]                        8
unet3d_attention/conv3d_26/kernel:0                                   [2, 2, 2, 8, 8]          512
unet3d_attention/conv3d_26/bias:0                                     [8]                        8
unet3d_attention/conv3d_27/kernel:0                                   [3, 3, 3, 8, 8]         1728
unet3d_attention/conv3d_27/ins_norm/beta:0                            [8]                        8
unet3d_attention/conv3d_27/ins_norm/gamma:0                           [8]                        8
unet3d_attention/conv3d_28/kernel:0                                   [2, 2, 2, 8, 8]          512
unet3d_attention/conv3d_28/bias:0                                     [8]                        8
unet3d_attention/conv3d_29/kernel:0                                   [3, 3, 3, 8, 8]         1728
unet3d_attention/conv3d_29/ins_norm/beta:0                            [8]                        8
unet3d_attention/conv3d_29/ins_norm/gamma:0                           [8]                        8
unet3d_attention/conv3d_30/kernel:0                                   [2, 2, 2, 8, 8]          512
unet3d_attention/conv3d_30/bias:0                                     [8]                        8
unet3d_attention/conv3d_31/kernel:0                                   [3, 3, 3, 8, 8]         1728
unet3d_attention/conv3d_31/ins_norm/beta:0                            [8]                        8
unet3d_attention/conv3d_31/ins_norm/gamma:0                           [8]                        8
unet3d_attention/conv3d_32/kernel:0                                   [3, 3, 3, 16, 8]        3456
unet3d_attention/conv3d_32/ins_norm/beta:0                            [8]                        8
unet3d_attention/conv3d_32/ins_norm/gamma:0                           [8]                        8
unet3d_attention/conv3d_33/kernel:0                                   [3, 3, 3, 16, 8]        3456
unet3d_attention/conv3d_33/ins_norm/beta:0                            [8]                        8
unet3d_attention/conv3d_33/ins_norm/gamma:0                           [8]                        8
unet3d_attention/conv3d_34/kernel:0                                   [3, 3, 3, 16, 8]        3456
unet3d_attention/conv3d_34/ins_norm/beta:0                            [8]                        8
unet3d_attention/conv3d_34/ins_norm/gamma:0                           [8]                        8
unet3d_attention/conv3d_35/kernel:0                                   [3, 3, 3, 16, 8]        3456
unet3d_attention/conv3d_35/ins_norm/beta:0                            [8]                        8
unet3d_attention/conv3d_35/ins_norm/gamma:0                           [8]                        8
unet3d_attention/conv3d_36/kernel:0                                   [3, 3, 3, 16, 32]      13824
unet3d_attention/conv3d_36/ins_norm/beta:0                            [32]                      32
unet3d_attention/conv3d_36/ins_norm/gamma:0                           [32]                      32
unet3d_attention/stride2conv1/kernel:0                                [3, 3, 3, 32, 64]      55296
unet3d_attention/stride2conv1/bias:0                                  [64]                      64
unet3d_attention/stride2conv1/ins_norm/beta:0                         [64]                      64
unet3d_attention/stride2conv1/ins_norm/gamma:0                        [64]                      64
unet3d_attention/down2_conv_0/kernel:0                                [3, 3, 3, 64, 64]     110592
unet3d_attention/down2_conv_0/bias:0                                  [64]                      64
unet3d_attention/down2_conv_0/ins_norm/beta:0                         [64]                      64
unet3d_attention/down2_conv_0/ins_norm/gamma:0                        [64]                      64
unet3d_attention/down2_conv_1/kernel:0                                [3, 3, 3, 64, 64]     110592
unet3d_attention/down2_conv_1/bias:0                                  [64]                      64
unet3d_attention/down2_conv_1/ins_norm/beta:0                         [64]                      64
unet3d_attention/down2_conv_1/ins_norm/gamma:0                        [64]                      64
unet3d_attention/stride2conv2/kernel:0                                [3, 3, 3, 64, 128]    221184
unet3d_attention/stride2conv2/bias:0                                  [128]                    128
unet3d_attention/stride2conv2/ins_norm/beta:0                         [128]                    128
unet3d_attention/stride2conv2/ins_norm/gamma:0                        [128]                    128
unet3d_attention/down3_conv_0/kernel:0                                [3, 3, 3, 128, 128]   442368
unet3d_attention/down3_conv_0/bias:0                                  [128]                    128
unet3d_attention/down3_conv_0/ins_norm/beta:0                         [128]                    128
unet3d_attention/down3_conv_0/ins_norm/gamma:0                        [128]                    128
unet3d_attention/down3_conv_1/kernel:0                                [3, 3, 3, 128, 128]   442368
unet3d_attention/down3_conv_1/bias:0                                  [128]                    128
unet3d_attention/down3_conv_1/ins_norm/beta:0                         [128]                    128
unet3d_attention/down3_conv_1/ins_norm/gamma:0                        [128]                    128
unet3d_attention/stride2conv3/kernel:0                                [3, 3, 3, 128, 256]   884736
unet3d_attention/stride2conv3/bias:0                                  [256]                    256
unet3d_attention/stride2conv3/ins_norm/beta:0                         [256]                    256
unet3d_attention/stride2conv3/ins_norm/gamma:0                        [256]                    256
unet3d_attention/down4_conv_0/kernel:0                                [3, 3, 3, 256, 256]  1769472
unet3d_attention/down4_conv_0/bias:0                                  [256]                    256
unet3d_attention/down4_conv_0/ins_norm/beta:0                         [256]                    256
unet3d_attention/down4_conv_0/ins_norm/gamma:0                        [256]                    256
unet3d_attention/down4_conv_1/kernel:0                                [3, 3, 3, 256, 256]  1769472
unet3d_attention/down4_conv_1/bias:0                                  [256]                    256
unet3d_attention/down4_conv_1/ins_norm/beta:0                         [256]                    256
unet3d_attention/down4_conv_1/ins_norm/gamma:0                        [256]                    256
unet3d_attention/C1_conv/kernel:0                                     [3, 3, 3, 16, 64]      27648
unet3d_attention/C1_conv/bias:0                                       [64]                      64
unet3d_attention/C1_conv/ins_norm/beta:0                              [64]                      64
unet3d_attention/C1_conv/ins_norm/gamma:0                             [64]                      64
unet3d_attention/C2_conv/kernel:0                                     [3, 3, 3, 32, 64]      55296
unet3d_attention/C2_conv/bias:0                                       [64]                      64
unet3d_attention/C2_conv/ins_norm/beta:0                              [64]                      64
unet3d_attention/C2_conv/ins_norm/gamma:0                             [64]                      64
unet3d_attention/C3_cfe_cfe0/kernel:0                                 [1, 1, 1, 64, 32]       2048
unet3d_attention/C3_cfe_cfe0/ins_norm/beta:0                          [32]                      32
unet3d_attention/C3_cfe_cfe0/ins_norm/gamma:0                         [32]                      32
unet3d_attention/C3_cfe_cfe1_dilation/kernel:0                        [3, 3, 3, 64, 32]      55296
unet3d_attention/C3_cfe_cfe1_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C3_cfe_cfe1_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C3_cfe_cfe2_dilation/kernel:0                        [3, 3, 3, 64, 32]      55296
unet3d_attention/C3_cfe_cfe2_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C3_cfe_cfe2_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C3_cfe_cfe3_dilation/kernel:0                        [3, 3, 3, 64, 32]      55296
unet3d_attention/C3_cfe_cfe3_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C3_cfe_cfe3_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C4_cfe_cfe0/kernel:0                                 [1, 1, 1, 128, 32]      4096
unet3d_attention/C4_cfe_cfe0/ins_norm/beta:0                          [32]                      32
unet3d_attention/C4_cfe_cfe0/ins_norm/gamma:0                         [32]                      32
unet3d_attention/C4_cfe_cfe1_dilation/kernel:0                        [3, 3, 3, 128, 32]    110592
unet3d_attention/C4_cfe_cfe1_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C4_cfe_cfe1_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C4_cfe_cfe2_dilation/kernel:0                        [3, 3, 3, 128, 32]    110592
unet3d_attention/C4_cfe_cfe2_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C4_cfe_cfe2_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C4_cfe_cfe3_dilation/kernel:0                        [3, 3, 3, 128, 32]    110592
unet3d_attention/C4_cfe_cfe3_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C4_cfe_cfe3_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C5_cfe_cfe0/kernel:0                                 [1, 1, 1, 256, 32]      8192
unet3d_attention/C5_cfe_cfe0/ins_norm/beta:0                          [32]                      32
unet3d_attention/C5_cfe_cfe0/ins_norm/gamma:0                         [32]                      32
unet3d_attention/C5_cfe_cfe1_dilation/kernel:0                        [3, 3, 3, 256, 32]    221184
unet3d_attention/C5_cfe_cfe1_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C5_cfe_cfe1_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C5_cfe_cfe2_dilation/kernel:0                        [3, 3, 3, 256, 32]    221184
unet3d_attention/C5_cfe_cfe2_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C5_cfe_cfe2_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C5_cfe_cfe3_dilation/kernel:0                        [3, 3, 3, 256, 32]    221184
unet3d_attention/C5_cfe_cfe3_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C5_cfe_cfe3_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/up_conv1_C5_cfe_up4/kernel:0                         [3, 3, 3, 128, 128]   442368
unet3d_attention/up_conv1_C5_cfe_up4/bias:0                           [128]                    128
unet3d_attention/up_conv1_C5_cfe_up4/ins_norm/beta:0                  [128]                    128
unet3d_attention/up_conv1_C5_cfe_up4/ins_norm/gamma:0                 [128]                    128
unet3d_attention/up_conv1_C4_cfe_up2/kernel:0                         [3, 3, 3, 128, 128]   442368
unet3d_attention/up_conv1_C4_cfe_up2/bias:0                           [128]                    128
unet3d_attention/up_conv1_C4_cfe_up2/ins_norm/beta:0                  [128]                    128
unet3d_attention/up_conv1_C4_cfe_up2/ins_norm/gamma:0                 [128]                    128
unet3d_attention/C345_ChannelWiseAttention_withcpfe_dense_1/kernel:0  [384, 96]              36864
unet3d_attention/C345_ChannelWiseAttention_withcpfe_dense_1/bias:0    [96]                      96
unet3d_attention/C345_ChannelWiseAttention_withcpfe_dense_2/kernel:0  [96, 384]              36864
unet3d_attention/C345_ChannelWiseAttention_withcpfe_dense_2/bias:0    [384]                    384
unet3d_attention/C345_conv/kernel:0                                   [1, 1, 1, 384, 64]     24576
unet3d_attention/C345_conv/bias:0                                     [64]                      64
unet3d_attention/C345_conv/ins_norm/beta:0                            [64]                      64
unet3d_attention/C345_conv/ins_norm/gamma:0                           [64]                      64
unet3d_attention/up_conv1_C345_up4/kernel:0                           [3, 3, 3, 64, 64]     110592
unet3d_attention/up_conv1_C345_up4/bias:0                             [64]                      64
unet3d_attention/up_conv1_C345_up4/ins_norm/beta:0                    [64]                      64
unet3d_attention/up_conv1_C345_up4/ins_norm/gamma:0                   [64]                      64
unet3d_attention/spatial_attention_1_conv1/kernel:0                   [1, 9, 9, 64, 32]     165888
unet3d_attention/spatial_attention_1_conv1/bias:0                     [32]                      32
unet3d_attention/spatial_attention_1_conv1/ins_norm/beta:0            [32]                      32
unet3d_attention/spatial_attention_1_conv1/ins_norm/gamma:0           [32]                      32
unet3d_attention/spatial_attention_1_conv2/kernel:0                   [9, 1, 1, 32, 1]         288
unet3d_attention/spatial_attention_1_conv2/bias:0                     [1]                        1
unet3d_attention/spatial_attention_1_conv2/ins_norm/beta:0            [1]                        1
unet3d_attention/spatial_attention_1_conv2/ins_norm/gamma:0           [1]                        1
unet3d_attention/spatial_attention_2_conv1/kernel:0                   [9, 1, 9, 64, 32]     165888
unet3d_attention/spatial_attention_2_conv1/bias:0                     [32]                      32
unet3d_attention/spatial_attention_2_conv1/ins_norm/beta:0            [32]                      32
unet3d_attention/spatial_attention_2_conv1/ins_norm/gamma:0           [32]                      32
unet3d_attention/spatial_attention_2_conv2/kernel:0                   [1, 9, 1, 32, 1]         288
unet3d_attention/spatial_attention_2_conv2/bias:0                     [1]                        1
unet3d_attention/spatial_attention_2_conv2/ins_norm/beta:0            [1]                        1
unet3d_attention/spatial_attention_2_conv2/ins_norm/gamma:0           [1]                        1
unet3d_attention/spatial_attention_3_conv1/kernel:0                   [9, 9, 1, 64, 32]     165888
unet3d_attention/spatial_attention_3_conv1/bias:0                     [32]                      32
unet3d_attention/spatial_attention_3_conv1/ins_norm/beta:0            [32]                      32
unet3d_attention/spatial_attention_3_conv1/ins_norm/gamma:0           [32]                      32
unet3d_attention/spatial_attention_3_conv2/kernel:0                   [1, 1, 9, 32, 1]         288
unet3d_attention/spatial_attention_3_conv2/bias:0                     [1]                        1
unet3d_attention/spatial_attention_3_conv2/ins_norm/beta:0            [1]                        1
unet3d_attention/spatial_attention_3_conv2/ins_norm/gamma:0           [1]                        1
unet3d_attention/up_conv1_C2_up2/kernel:0                             [3, 3, 3, 64, 64]     110592
unet3d_attention/up_conv1_C2_up2/bias:0                               [64]                      64
unet3d_attention/up_conv1_C2_up2/ins_norm/beta:0                      [64]                      64
unet3d_attention/up_conv1_C2_up2/ins_norm/gamma:0                     [64]                      64
unet3d_attention/C12_conv/kernel:0                                    [3, 3, 3, 128, 64]    221184
unet3d_attention/C12_conv/bias:0                                      [64]                      64
unet3d_attention/C12_conv/ins_norm/beta:0                             [64]                      64
unet3d_attention/C12_conv/ins_norm/gamma:0                            [64]                      64
unet3d_attention/final/kernel:0                                       [3, 3, 3, 128, 4]      13824
unet3d_attention/final/bias:0                                         [4]                        4
Total #vars=242, #params=9147573, size=34.90MB

# Stair cases
17378MiB

unet3d_attention input: [1, 128, 128, 128, 4]
inputs  Tensor("QueueInput/input_deque:0", shape=(1, 128, 128, 128, 4), dtype=float32, device=/device:CPU:0)
WARNING:tensorflow:From /home/dghan/3DUnet_attention/model.py:79: conv3d (from tensorflow.python.layers.convolutional) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.keras.layers.Conv3D` instead.
WARNING:tensorflow:From /home/dghan/envs/3DUnet_attention/lib/python3.7/site-packages/tensorflow_core/python/layers/convolutional.py:632: Layer.apply (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
Instructions for updating:
Please use `layer.__call__` method instead.
unet3d_attention/init_conv/Relu:0 (128, 128, 128, 16)
Residual bock downsampling  0      (128, 128, 128, 16)
Down Conv3D  0     (64, 64, 64, 32)
Residual bock downsampling  1      (64, 64, 64, 32)
Down Conv3D  1     (32, 32, 32, 64)
Residual bock downsampling  2      (32, 32, 32, 64)
Down Conv3D  2     (16, 16, 16, 128)
Residual bock downsampling  3      (16, 16, 16, 128)
Down Conv3D  3     (8, 8, 8, 256)
Residual bock downsampling  4      (8, 8, 8, 256)
Low level feature 1      (128, 128, 128, 64)
Low level feature 2      (64, 64, 64, 64)
High level feature 1 CFE         (32, 32, 32, 128)
High level feature 2 CFE         (16, 16, 16, 128)
High level feature 3 CFE         (8, 8, 8, 128)
@Stair case version High level features aspp concat      (32, 32, 32, 256)
WARNING:tensorflow:From /home/dghan/3DUnet_attention/attention.py:169: dense (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
Instructions for updating:
Use keras.layers.Dense instead.
High level features CA   (32, 32, 32, 256)
High level features conv         (32, 32, 32, 64)
High level features upsampling   (128, 128, 128, 64)
High level features SA   (128, 128, 128, 64)
Low level feature conv   (128, 128, 128, 64)
Low + High level feature         (128, 128, 128, 128)
final (128, 128, 128, 4)
[0317 00:25:31 @registry.py:129] unet3d_attention output: [1, 128, 128, 128, 4]

Trainable Variables: 
name                                                                  shape                    dim
--------------------------------------------------------------------  -------------------  -------
unet3d_attention/init_conv/kernel:0                                   [3, 3, 3, 4, 16]        1728
unet3d_attention/init_conv/bias:0                                     [16]                      16
unet3d_attention/init_conv/ins_norm/beta:0                            [16]                      16
unet3d_attention/init_conv/ins_norm/gamma:0                           [16]                      16
unet3d_attention/down0_conv_0/kernel:0                                [3, 3, 3, 16, 16]       6912
unet3d_attention/down0_conv_0/bias:0                                  [16]                      16
unet3d_attention/down0_conv_0/ins_norm/beta:0                         [16]                      16
unet3d_attention/down0_conv_0/ins_norm/gamma:0                        [16]                      16
unet3d_attention/down0_conv_1/kernel:0                                [3, 3, 3, 16, 16]       6912
unet3d_attention/down0_conv_1/bias:0                                  [16]                      16
unet3d_attention/down0_conv_1/ins_norm/beta:0                         [16]                      16
unet3d_attention/down0_conv_1/ins_norm/gamma:0                        [16]                      16
unet3d_attention/stride2conv0/kernel:0                                [3, 3, 3, 16, 32]      13824
unet3d_attention/stride2conv0/bias:0                                  [32]                      32
unet3d_attention/stride2conv0/ins_norm/beta:0                         [32]                      32
unet3d_attention/stride2conv0/ins_norm/gamma:0                        [32]                      32
unet3d_attention/down1_conv_0/kernel:0                                [3, 3, 3, 32, 32]      27648
unet3d_attention/down1_conv_0/bias:0                                  [32]                      32
unet3d_attention/down1_conv_0/ins_norm/beta:0                         [32]                      32
unet3d_attention/down1_conv_0/ins_norm/gamma:0                        [32]                      32
unet3d_attention/down1_conv_1/kernel:0                                [3, 3, 3, 32, 32]      27648
unet3d_attention/down1_conv_1/bias:0                                  [32]                      32
unet3d_attention/down1_conv_1/ins_norm/beta:0                         [32]                      32
unet3d_attention/down1_conv_1/ins_norm/gamma:0                        [32]                      32
unet3d_attention/stride2conv1/kernel:0                                [3, 3, 3, 32, 64]      55296
unet3d_attention/stride2conv1/bias:0                                  [64]                      64
unet3d_attention/stride2conv1/ins_norm/beta:0                         [64]                      64
unet3d_attention/stride2conv1/ins_norm/gamma:0                        [64]                      64
unet3d_attention/down2_conv_0/kernel:0                                [3, 3, 3, 64, 64]     110592
unet3d_attention/down2_conv_0/bias:0                                  [64]                      64
unet3d_attention/down2_conv_0/ins_norm/beta:0                         [64]                      64
unet3d_attention/down2_conv_0/ins_norm/gamma:0                        [64]                      64
unet3d_attention/down2_conv_1/kernel:0                                [3, 3, 3, 64, 64]     110592
unet3d_attention/down2_conv_1/bias:0                                  [64]                      64
unet3d_attention/down2_conv_1/ins_norm/beta:0                         [64]                      64
unet3d_attention/down2_conv_1/ins_norm/gamma:0                        [64]                      64
unet3d_attention/stride2conv2/kernel:0                                [3, 3, 3, 64, 128]    221184
unet3d_attention/stride2conv2/bias:0                                  [128]                    128
unet3d_attention/stride2conv2/ins_norm/beta:0                         [128]                    128
unet3d_attention/stride2conv2/ins_norm/gamma:0                        [128]                    128
unet3d_attention/down3_conv_0/kernel:0                                [3, 3, 3, 128, 128]   442368
unet3d_attention/down3_conv_0/bias:0                                  [128]                    128
unet3d_attention/down3_conv_0/ins_norm/beta:0                         [128]                    128
unet3d_attention/down3_conv_0/ins_norm/gamma:0                        [128]                    128
unet3d_attention/down3_conv_1/kernel:0                                [3, 3, 3, 128, 128]   442368
unet3d_attention/down3_conv_1/bias:0                                  [128]                    128
unet3d_attention/down3_conv_1/ins_norm/beta:0                         [128]                    128
unet3d_attention/down3_conv_1/ins_norm/gamma:0                        [128]                    128
unet3d_attention/stride2conv3/kernel:0                                [3, 3, 3, 128, 256]   884736
unet3d_attention/stride2conv3/bias:0                                  [256]                    256
unet3d_attention/stride2conv3/ins_norm/beta:0                         [256]                    256
unet3d_attention/stride2conv3/ins_norm/gamma:0                        [256]                    256
unet3d_attention/down4_conv_0/kernel:0                                [3, 3, 3, 256, 256]  1769472
unet3d_attention/down4_conv_0/bias:0                                  [256]                    256
unet3d_attention/down4_conv_0/ins_norm/beta:0                         [256]                    256
unet3d_attention/down4_conv_0/ins_norm/gamma:0                        [256]                    256
unet3d_attention/down4_conv_1/kernel:0                                [3, 3, 3, 256, 256]  1769472
unet3d_attention/down4_conv_1/bias:0                                  [256]                    256
unet3d_attention/down4_conv_1/ins_norm/beta:0                         [256]                    256
unet3d_attention/down4_conv_1/ins_norm/gamma:0                        [256]                    256
unet3d_attention/C1_conv/kernel:0                                     [3, 3, 3, 16, 64]      27648
unet3d_attention/C1_conv/bias:0                                       [64]                      64
unet3d_attention/C1_conv/ins_norm/beta:0                              [64]                      64
unet3d_attention/C1_conv/ins_norm/gamma:0                             [64]                      64
unet3d_attention/C2_conv/kernel:0                                     [3, 3, 3, 32, 64]      55296
unet3d_attention/C2_conv/bias:0                                       [64]                      64
unet3d_attention/C2_conv/ins_norm/beta:0                              [64]                      64
unet3d_attention/C2_conv/ins_norm/gamma:0                             [64]                      64
unet3d_attention/C3_cfe_cfe0/kernel:0                                 [1, 1, 1, 64, 32]       2048
unet3d_attention/C3_cfe_cfe0/ins_norm/beta:0                          [32]                      32
unet3d_attention/C3_cfe_cfe0/ins_norm/gamma:0                         [32]                      32
unet3d_attention/C3_cfe_cfe1_dilation/kernel:0                        [3, 3, 3, 64, 32]      55296
unet3d_attention/C3_cfe_cfe1_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C3_cfe_cfe1_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C3_cfe_cfe2_dilation/kernel:0                        [3, 3, 3, 64, 32]      55296
unet3d_attention/C3_cfe_cfe2_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C3_cfe_cfe2_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C3_cfe_cfe3_dilation/kernel:0                        [3, 3, 3, 64, 32]      55296
unet3d_attention/C3_cfe_cfe3_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C3_cfe_cfe3_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C4_cfe_cfe0/kernel:0                                 [1, 1, 1, 128, 32]      4096
unet3d_attention/C4_cfe_cfe0/ins_norm/beta:0                          [32]                      32
unet3d_attention/C4_cfe_cfe0/ins_norm/gamma:0                         [32]                      32
unet3d_attention/C4_cfe_cfe1_dilation/kernel:0                        [3, 3, 3, 128, 32]    110592
unet3d_attention/C4_cfe_cfe1_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C4_cfe_cfe1_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C4_cfe_cfe2_dilation/kernel:0                        [3, 3, 3, 128, 32]    110592
unet3d_attention/C4_cfe_cfe2_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C4_cfe_cfe2_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C4_cfe_cfe3_dilation/kernel:0                        [3, 3, 3, 128, 32]    110592
unet3d_attention/C4_cfe_cfe3_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C4_cfe_cfe3_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C5_cfe_cfe0/kernel:0                                 [1, 1, 1, 256, 32]      8192
unet3d_attention/C5_cfe_cfe0/ins_norm/beta:0                          [32]                      32
unet3d_attention/C5_cfe_cfe0/ins_norm/gamma:0                         [32]                      32
unet3d_attention/C5_cfe_cfe1_dilation/kernel:0                        [3, 3, 3, 256, 32]    221184
unet3d_attention/C5_cfe_cfe1_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C5_cfe_cfe1_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C5_cfe_cfe2_dilation/kernel:0                        [3, 3, 3, 256, 32]    221184
unet3d_attention/C5_cfe_cfe2_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C5_cfe_cfe2_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C5_cfe_cfe3_dilation/kernel:0                        [3, 3, 3, 256, 32]    221184
unet3d_attention/C5_cfe_cfe3_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C5_cfe_cfe3_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/up_conv1_C5_cfe_up4/kernel:0                         [3, 3, 3, 128, 128]   442368
unet3d_attention/up_conv1_C5_cfe_up4/bias:0                           [128]                    128
unet3d_attention/up_conv1_C5_cfe_up4/ins_norm/beta:0                  [128]                    128
unet3d_attention/up_conv1_C5_cfe_up4/ins_norm/gamma:0                 [128]                    128
unet3d_attention/up_conv1_C45_up2/kernel:0                            [3, 3, 3, 128, 128]   442368
unet3d_attention/up_conv1_C45_up2/bias:0                              [128]                    128
unet3d_attention/up_conv1_C45_up2/ins_norm/beta:0                     [128]                    128
unet3d_attention/up_conv1_C45_up2/ins_norm/gamma:0                    [128]                    128
unet3d_attention/C345_ChannelWiseAttention_withcpfe_dense_1/kernel:0  [256, 64]              16384
unet3d_attention/C345_ChannelWiseAttention_withcpfe_dense_1/bias:0    [64]                      64
unet3d_attention/C345_ChannelWiseAttention_withcpfe_dense_2/kernel:0  [64, 256]              16384
unet3d_attention/C345_ChannelWiseAttention_withcpfe_dense_2/bias:0    [256]                    256
unet3d_attention/C345_conv/kernel:0                                   [1, 1, 1, 256, 64]     16384
unet3d_attention/C345_conv/bias:0                                     [64]                      64
unet3d_attention/C345_conv/ins_norm/beta:0                            [64]                      64
unet3d_attention/C345_conv/ins_norm/gamma:0                           [64]                      64
unet3d_attention/up_conv1_C345_up4/kernel:0                           [3, 3, 3, 64, 64]     110592
unet3d_attention/up_conv1_C345_up4/bias:0                             [64]                      64
unet3d_attention/up_conv1_C345_up4/ins_norm/beta:0                    [64]                      64
unet3d_attention/up_conv1_C345_up4/ins_norm/gamma:0                   [64]                      64
unet3d_attention/spatial_attention_1_conv1/kernel:0                   [1, 9, 9, 64, 32]     165888
unet3d_attention/spatial_attention_1_conv1/bias:0                     [32]                      32
unet3d_attention/spatial_attention_1_conv1/ins_norm/beta:0            [32]                      32
unet3d_attention/spatial_attention_1_conv1/ins_norm/gamma:0           [32]                      32
unet3d_attention/spatial_attention_1_conv2/kernel:0                   [9, 1, 1, 32, 1]         288
unet3d_attention/spatial_attention_1_conv2/bias:0                     [1]                        1
unet3d_attention/spatial_attention_1_conv2/ins_norm/beta:0            [1]                        1
unet3d_attention/spatial_attention_1_conv2/ins_norm/gamma:0           [1]                        1
unet3d_attention/spatial_attention_2_conv1/kernel:0                   [9, 1, 9, 64, 32]     165888
unet3d_attention/spatial_attention_2_conv1/bias:0                     [32]                      32
unet3d_attention/spatial_attention_2_conv1/ins_norm/beta:0            [32]                      32
unet3d_attention/spatial_attention_2_conv1/ins_norm/gamma:0           [32]                      32
unet3d_attention/spatial_attention_2_conv2/kernel:0                   [1, 9, 1, 32, 1]         288
unet3d_attention/spatial_attention_2_conv2/bias:0                     [1]                        1
unet3d_attention/spatial_attention_2_conv2/ins_norm/beta:0            [1]                        1
unet3d_attention/spatial_attention_2_conv2/ins_norm/gamma:0           [1]                        1
unet3d_attention/spatial_attention_3_conv1/kernel:0                   [9, 9, 1, 64, 32]     165888
unet3d_attention/spatial_attention_3_conv1/bias:0                     [32]                      32
unet3d_attention/spatial_attention_3_conv1/ins_norm/beta:0            [32]                      32
unet3d_attention/spatial_attention_3_conv1/ins_norm/gamma:0           [32]                      32
unet3d_attention/spatial_attention_3_conv2/kernel:0                   [1, 1, 9, 32, 1]         288
unet3d_attention/spatial_attention_3_conv2/bias:0                     [1]                        1
unet3d_attention/spatial_attention_3_conv2/ins_norm/beta:0            [1]                        1
unet3d_attention/spatial_attention_3_conv2/ins_norm/gamma:0           [1]                        1
unet3d_attention/up_conv1_C2_up2/kernel:0                             [3, 3, 3, 64, 64]     110592
unet3d_attention/up_conv1_C2_up2/bias:0                               [64]                      64
unet3d_attention/up_conv1_C2_up2/ins_norm/beta:0                      [64]                      64
unet3d_attention/up_conv1_C2_up2/ins_norm/gamma:0                     [64]                      64
unet3d_attention/C12_conv/kernel:0                                    [3, 3, 3, 128, 64]    221184
unet3d_attention/C12_conv/bias:0                                      [64]                      64
unet3d_attention/C12_conv/ins_norm/beta:0                             [64]                      64
unet3d_attention/C12_conv/ins_norm/gamma:0                            [64]                      64
unet3d_attention/final/kernel:0                                       [3, 3, 3, 128, 4]      13824
unet3d_attention/final/bias:0                                         [4]                        4
Total #vars=158, #params=9045629, size=34.51MB

# Transformer spatial attention
33762MiB

unet3d_attention input: [1, 128, 128, 128, 4]
inputs  Tensor("QueueInput/input_deque:0", shape=(1, 128, 128, 128, 4), dtype=float32, device=/device:CPU:0)
WARNING:tensorflow:From /home/dghan/3DUnet_attention/model.py:79: conv3d (from tensorflow.python.layers.convolutional) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.keras.layers.Conv3D` instead.
WARNING:tensorflow:From /home/dghan/envs/3DUnet_attention/lib/python3.7/site-packages/tensorflow_core/python/layers/convolutional.py:632: Layer.apply (from tensorflow.python.keras.engine.base_layer) is deprecated and will be removed in a future version.
Instructions for updating:
Please use `layer.__call__` method instead.
unet3d_attention/init_conv/Relu:0 (128, 128, 128, 16)
Residual bock downsampling  0      (128, 128, 128, 16)
Down Conv3D  0     (64, 64, 64, 32)
Residual bock downsampling  1      (64, 64, 64, 32)
Down Conv3D  1     (32, 32, 32, 64)
Residual bock downsampling  2      (32, 32, 32, 64)
Down Conv3D  2     (16, 16, 16, 128)
Residual bock downsampling  3      (16, 16, 16, 128)
Down Conv3D  3     (8, 8, 8, 256)
Residual bock downsampling  4      (8, 8, 8, 256)
Low level feature 1      (128, 128, 128, 64)
Low level feature 2      (64, 64, 64, 64)
High level feature 1 CFE         (32, 32, 32, 128)
High level feature 2 CFE         (16, 16, 16, 128)
High level feature 3 CFE         (8, 8, 8, 128)
High level features aspp concat  (32, 32, 32, 384)
WARNING:tensorflow:From /home/dghan/3DUnet_attention/attention.py:169: dense (from tensorflow.python.layers.core) is deprecated and will be removed in a future version.
Instructions for updating:
Use keras.layers.Dense instead.
High level features CA   (32, 32, 32, 384)
High level features conv         (32, 32, 32, 64)
High level features upsampling   (128, 128, 128, 64)
High level features SA   (128, 128, 128, 64)
Low level feature conv   (128, 128, 128, 64)
transformer spatial attention level: 6
Low + High level feature         (128, 128, 128, 128)
final (128, 128, 128, 4)
[0317 00:32:37 @registry.py:129] unet3d_attention output: [1, 128, 128, 128, 4]

unet3d_attention output: [1, 128, 128, 128, 4]
[0317 00:32:37 @regularize.py:82] regularize_cost() found 68 variables to regularize.
[0317 00:32:37 @regularize.py:19] The following tensors will be regularized: unet3d_attention/init_conv/kernel:0, unet3d_attention/down0_conv_0/kernel:0, unet3d_attention/down0_conv_1/kernel:0, unet3d_attention/stride2conv0/kernel:0, unet3d_attention/down1_conv_0/kernel:0, unet3d_attention/down1_conv_1/kernel:0, unet3d_attention/stride2conv1/kernel:0, unet3d_attention/down2_conv_0/kernel:0, unet3d_attention/down2_conv_1/kernel:0, unet3d_attention/stride2conv2/kernel:0, unet3d_attention/down3_conv_0/kernel:0, unet3d_attention/down3_conv_1/kernel:0, unet3d_attention/stride2conv3/kernel:0, unet3d_attention/down4_conv_0/kernel:0, unet3d_attention/down4_conv_1/kernel:0, unet3d_attention/C1_conv/kernel:0, unet3d_attention/C2_conv/kernel:0, unet3d_attention/C3_cfe_cfe0/kernel:0, unet3d_attention/C3_cfe_cfe1_dilation/kernel:0, unet3d_attention/C3_cfe_cfe2_dilation/kernel:0, unet3d_attention/C3_cfe_cfe3_dilation/kernel:0, unet3d_attention/C4_cfe_cfe0/kernel:0, unet3d_attention/C4_cfe_cfe1_dilation/kernel:0, unet3d_attention/C4_cfe_cfe2_dilation/kernel:0, unet3d_attention/C4_cfe_cfe3_dilation/kernel:0, unet3d_attention/C5_cfe_cfe0/kernel:0, unet3d_attention/C5_cfe_cfe1_dilation/kernel:0, unet3d_attention/C5_cfe_cfe2_dilation/kernel:0, unet3d_attention/C5_cfe_cfe3_dilation/kernel:0, unet3d_attention/up_conv1_C5_cfe_up4/kernel:0, unet3d_attention/up_conv1_C4_cfe_up2/kernel:0, unet3d_attention/C345_ChannelWiseAttention_withcpfe_dense_1/kernel:0, unet3d_attention/C345_ChannelWiseAttention_withcpfe_dense_2/kernel:0, unet3d_attention/C345_conv/kernel:0, unet3d_attention/up_conv1_C345_up4/kernel:0, unet3d_attention/spatial_attention_1_conv1/kernel:0, unet3d_attention/spatial_attention_1_conv2/kernel:0, unet3d_attention/spatial_attention_2_conv1/kernel:0, unet3d_attention/spatial_attention_2_conv2/kernel:0, unet3d_attention/spatial_attention_3_conv1/kernel:0, unet3d_attention/spatial_attention_3_conv2/kernel:0, unet3d_attention/up_conv1_C2_up2/kernel:0, unet3d_attention/C12_conv/kernel:0, unet3d_attention/spatial_attention_2_1_conv1/kernel:0, unet3d_attention/spatial_attention_2_1_conv2/kernel:0, unet3d_attention/spatial_attention_2_2_conv1/kernel:0, unet3d_attention/spatial_attention_2_2_conv2/kernel:0, unet3d_attention/spatial_attention_2_3_conv1/kernel:0, unet3d_attention/spatial_attention_2_3_conv2/kernel:0, unet3d_attention/spatial_attention_3_1_conv1/kernel:0, unet3d_attention/spatial_attention_3_1_conv2/kernel:0, unet3d_attention/spatial_attention_3_2_conv1/kernel:0, unet3d_attention/spatial_attention_3_2_conv2/kernel:0, unet3d_attention/spatial_attention_3_3_conv1/kernel:0, unet3d_attention/spatial_attention_3_3_conv2/kernel:0, unet3d_attention/spatial_attention_4_1_conv1/kernel:0, unet3d_attention/spatial_attention_4_1_conv2/kernel:0, unet3d_attention/spatial_attention_4_2_conv1/kernel:0, unet3d_attention/spatial_attention_4_2_conv2/kernel:0, unet3d_attention/spatial_attention_4_3_conv1/kernel:0, unet3d_attention/spatial_attention_4_3_conv2/kernel:0, unet3d_attention/spatial_attention_5_1_conv1/kernel:0, unet3d_attention/spatial_attention_5_1_conv2/kernel:0, unet3d_attention/spatial_attention_5_2_conv1/kernel:0, unet3d_attention/spatial_attention_5_2_conv2/kernel:0, unet3d_attention/spatial_attention_5_3_conv1/kernel:0, unet3d_attention/spatial_attention_5_3_conv2/kernel:0, unet3d_attention/final/kernel:0
WARNING:tensorflow:From /home/dghan/envs/3DUnet_attention/lib/python3.7/site-packages/tensorpack/tfutils/summary.py:263: The name tf.add_to_collection is deprecated. Please use tf.compat.v1.add_to_collection instead.

WARNING:tensorflow:From train.py:66: The name tf.train.MomentumOptimizer is deprecated. Please use tf.compat.v1.train.MomentumOptimizer instead.

[0317 00:32:40 @model_utils.py:49] Trainable Variables: 
name                                                                  shape                    dim
--------------------------------------------------------------------  -------------------  -------
unet3d_attention/init_conv/kernel:0                                   [3, 3, 3, 4, 16]        1728
unet3d_attention/init_conv/bias:0                                     [16]                      16
unet3d_attention/init_conv/ins_norm/beta:0                            [16]                      16
unet3d_attention/init_conv/ins_norm/gamma:0                           [16]                      16
unet3d_attention/down0_conv_0/kernel:0                                [3, 3, 3, 16, 16]       6912
unet3d_attention/down0_conv_0/bias:0                                  [16]                      16
unet3d_attention/down0_conv_0/ins_norm/beta:0                         [16]                      16
unet3d_attention/down0_conv_0/ins_norm/gamma:0                        [16]                      16
unet3d_attention/down0_conv_1/kernel:0                                [3, 3, 3, 16, 16]       6912
unet3d_attention/down0_conv_1/bias:0                                  [16]                      16
unet3d_attention/down0_conv_1/ins_norm/beta:0                         [16]                      16
unet3d_attention/down0_conv_1/ins_norm/gamma:0                        [16]                      16
unet3d_attention/stride2conv0/kernel:0                                [3, 3, 3, 16, 32]      13824
unet3d_attention/stride2conv0/bias:0                                  [32]                      32
unet3d_attention/stride2conv0/ins_norm/beta:0                         [32]                      32
unet3d_attention/stride2conv0/ins_norm/gamma:0                        [32]                      32
unet3d_attention/down1_conv_0/kernel:0                                [3, 3, 3, 32, 32]      27648
unet3d_attention/down1_conv_0/bias:0                                  [32]                      32
unet3d_attention/down1_conv_0/ins_norm/beta:0                         [32]                      32
unet3d_attention/down1_conv_0/ins_norm/gamma:0                        [32]                      32
unet3d_attention/down1_conv_1/kernel:0                                [3, 3, 3, 32, 32]      27648
unet3d_attention/down1_conv_1/bias:0                                  [32]                      32
unet3d_attention/down1_conv_1/ins_norm/beta:0                         [32]                      32
unet3d_attention/down1_conv_1/ins_norm/gamma:0                        [32]                      32
unet3d_attention/stride2conv1/kernel:0                                [3, 3, 3, 32, 64]      55296
unet3d_attention/stride2conv1/bias:0                                  [64]                      64
unet3d_attention/stride2conv1/ins_norm/beta:0                         [64]                      64
unet3d_attention/stride2conv1/ins_norm/gamma:0                        [64]                      64
unet3d_attention/down2_conv_0/kernel:0                                [3, 3, 3, 64, 64]     110592
unet3d_attention/down2_conv_0/bias:0                                  [64]                      64
unet3d_attention/down2_conv_0/ins_norm/beta:0                         [64]                      64
unet3d_attention/down2_conv_0/ins_norm/gamma:0                        [64]                      64
unet3d_attention/down2_conv_1/kernel:0                                [3, 3, 3, 64, 64]     110592
unet3d_attention/down2_conv_1/bias:0                                  [64]                      64
unet3d_attention/down2_conv_1/ins_norm/beta:0                         [64]                      64
unet3d_attention/down2_conv_1/ins_norm/gamma:0                        [64]                      64
unet3d_attention/stride2conv2/kernel:0                                [3, 3, 3, 64, 128]    221184
unet3d_attention/stride2conv2/bias:0                                  [128]                    128
unet3d_attention/stride2conv2/ins_norm/beta:0                         [128]                    128
unet3d_attention/stride2conv2/ins_norm/gamma:0                        [128]                    128
unet3d_attention/down3_conv_0/kernel:0                                [3, 3, 3, 128, 128]   442368
unet3d_attention/down3_conv_0/bias:0                                  [128]                    128
unet3d_attention/down3_conv_0/ins_norm/beta:0                         [128]                    128
unet3d_attention/down3_conv_0/ins_norm/gamma:0                        [128]                    128
unet3d_attention/down3_conv_1/kernel:0                                [3, 3, 3, 128, 128]   442368
unet3d_attention/down3_conv_1/bias:0                                  [128]                    128
unet3d_attention/down3_conv_1/ins_norm/beta:0                         [128]                    128
unet3d_attention/down3_conv_1/ins_norm/gamma:0                        [128]                    128
unet3d_attention/stride2conv3/kernel:0                                [3, 3, 3, 128, 256]   884736
unet3d_attention/stride2conv3/bias:0                                  [256]                    256
unet3d_attention/stride2conv3/ins_norm/beta:0                         [256]                    256
unet3d_attention/stride2conv3/ins_norm/gamma:0                        [256]                    256
unet3d_attention/down4_conv_0/kernel:0                                [3, 3, 3, 256, 256]  1769472
unet3d_attention/down4_conv_0/bias:0                                  [256]                    256
unet3d_attention/down4_conv_0/ins_norm/beta:0                         [256]                    256
unet3d_attention/down4_conv_0/ins_norm/gamma:0                        [256]                    256
unet3d_attention/down4_conv_1/kernel:0                                [3, 3, 3, 256, 256]  1769472
unet3d_attention/down4_conv_1/bias:0                                  [256]                    256
unet3d_attention/down4_conv_1/ins_norm/beta:0                         [256]                    256
unet3d_attention/down4_conv_1/ins_norm/gamma:0                        [256]                    256
unet3d_attention/C1_conv/kernel:0                                     [3, 3, 3, 16, 64]      27648
unet3d_attention/C1_conv/bias:0                                       [64]                      64
unet3d_attention/C1_conv/ins_norm/beta:0                              [64]                      64
unet3d_attention/C1_conv/ins_norm/gamma:0                             [64]                      64
unet3d_attention/C2_conv/kernel:0                                     [3, 3, 3, 32, 64]      55296
unet3d_attention/C2_conv/bias:0                                       [64]                      64
unet3d_attention/C2_conv/ins_norm/beta:0                              [64]                      64
unet3d_attention/C2_conv/ins_norm/gamma:0                             [64]                      64
unet3d_attention/C3_cfe_cfe0/kernel:0                                 [1, 1, 1, 64, 32]       2048
unet3d_attention/C3_cfe_cfe0/ins_norm/beta:0                          [32]                      32
unet3d_attention/C3_cfe_cfe0/ins_norm/gamma:0                         [32]                      32
unet3d_attention/C3_cfe_cfe1_dilation/kernel:0                        [3, 3, 3, 64, 32]      55296
unet3d_attention/C3_cfe_cfe1_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C3_cfe_cfe1_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C3_cfe_cfe2_dilation/kernel:0                        [3, 3, 3, 64, 32]      55296
unet3d_attention/C3_cfe_cfe2_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C3_cfe_cfe2_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C3_cfe_cfe3_dilation/kernel:0                        [3, 3, 3, 64, 32]      55296
unet3d_attention/C3_cfe_cfe3_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C3_cfe_cfe3_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C4_cfe_cfe0/kernel:0                                 [1, 1, 1, 128, 32]      4096
unet3d_attention/C4_cfe_cfe0/ins_norm/beta:0                          [32]                      32
unet3d_attention/C4_cfe_cfe0/ins_norm/gamma:0                         [32]                      32
unet3d_attention/C4_cfe_cfe1_dilation/kernel:0                        [3, 3, 3, 128, 32]    110592
unet3d_attention/C4_cfe_cfe1_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C4_cfe_cfe1_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C4_cfe_cfe2_dilation/kernel:0                        [3, 3, 3, 128, 32]    110592
unet3d_attention/C4_cfe_cfe2_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C4_cfe_cfe2_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C4_cfe_cfe3_dilation/kernel:0                        [3, 3, 3, 128, 32]    110592
unet3d_attention/C4_cfe_cfe3_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C4_cfe_cfe3_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C5_cfe_cfe0/kernel:0                                 [1, 1, 1, 256, 32]      8192
unet3d_attention/C5_cfe_cfe0/ins_norm/beta:0                          [32]                      32
unet3d_attention/C5_cfe_cfe0/ins_norm/gamma:0                         [32]                      32
unet3d_attention/C5_cfe_cfe1_dilation/kernel:0                        [3, 3, 3, 256, 32]    221184
unet3d_attention/C5_cfe_cfe1_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C5_cfe_cfe1_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C5_cfe_cfe2_dilation/kernel:0                        [3, 3, 3, 256, 32]    221184
unet3d_attention/C5_cfe_cfe2_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C5_cfe_cfe2_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/C5_cfe_cfe3_dilation/kernel:0                        [3, 3, 3, 256, 32]    221184
unet3d_attention/C5_cfe_cfe3_dilation/ins_norm/beta:0                 [32]                      32
unet3d_attention/C5_cfe_cfe3_dilation/ins_norm/gamma:0                [32]                      32
unet3d_attention/up_conv1_C5_cfe_up4/kernel:0                         [3, 3, 3, 128, 128]   442368
unet3d_attention/up_conv1_C5_cfe_up4/bias:0                           [128]                    128
unet3d_attention/up_conv1_C5_cfe_up4/ins_norm/beta:0                  [128]                    128
unet3d_attention/up_conv1_C5_cfe_up4/ins_norm/gamma:0                 [128]                    128
unet3d_attention/up_conv1_C4_cfe_up2/kernel:0                         [3, 3, 3, 128, 128]   442368
unet3d_attention/up_conv1_C4_cfe_up2/bias:0                           [128]                    128
unet3d_attention/up_conv1_C4_cfe_up2/ins_norm/beta:0                  [128]                    128
unet3d_attention/up_conv1_C4_cfe_up2/ins_norm/gamma:0                 [128]                    128
unet3d_attention/C345_ChannelWiseAttention_withcpfe_dense_1/kernel:0  [384, 96]              36864
unet3d_attention/C345_ChannelWiseAttention_withcpfe_dense_1/bias:0    [96]                      96
unet3d_attention/C345_ChannelWiseAttention_withcpfe_dense_2/kernel:0  [96, 384]              36864
unet3d_attention/C345_ChannelWiseAttention_withcpfe_dense_2/bias:0    [384]                    384
unet3d_attention/C345_conv/kernel:0                                   [1, 1, 1, 384, 64]     24576
unet3d_attention/C345_conv/bias:0                                     [64]                      64
unet3d_attention/C345_conv/ins_norm/beta:0                            [64]                      64
unet3d_attention/C345_conv/ins_norm/gamma:0                           [64]                      64
unet3d_attention/up_conv1_C345_up4/kernel:0                           [3, 3, 3, 64, 64]     110592
unet3d_attention/up_conv1_C345_up4/bias:0                             [64]                      64
unet3d_attention/up_conv1_C345_up4/ins_norm/beta:0                    [64]                      64
unet3d_attention/up_conv1_C345_up4/ins_norm/gamma:0                   [64]                      64
unet3d_attention/spatial_attention_1_conv1/kernel:0                   [1, 9, 9, 64, 32]     165888
unet3d_attention/spatial_attention_1_conv1/bias:0                     [32]                      32
unet3d_attention/spatial_attention_1_conv1/ins_norm/beta:0            [32]                      32
unet3d_attention/spatial_attention_1_conv1/ins_norm/gamma:0           [32]                      32
unet3d_attention/spatial_attention_1_conv2/kernel:0                   [9, 1, 1, 32, 1]         288
unet3d_attention/spatial_attention_1_conv2/bias:0                     [1]                        1
unet3d_attention/spatial_attention_1_conv2/ins_norm/beta:0            [1]                        1
unet3d_attention/spatial_attention_1_conv2/ins_norm/gamma:0           [1]                        1
unet3d_attention/spatial_attention_2_conv1/kernel:0                   [9, 1, 9, 64, 32]     165888
unet3d_attention/spatial_attention_2_conv1/bias:0                     [32]                      32
unet3d_attention/spatial_attention_2_conv1/ins_norm/beta:0            [32]                      32
unet3d_attention/spatial_attention_2_conv1/ins_norm/gamma:0           [32]                      32
unet3d_attention/spatial_attention_2_conv2/kernel:0                   [1, 9, 1, 32, 1]         288
unet3d_attention/spatial_attention_2_conv2/bias:0                     [1]                        1
unet3d_attention/spatial_attention_2_conv2/ins_norm/beta:0            [1]                        1
unet3d_attention/spatial_attention_2_conv2/ins_norm/gamma:0           [1]                        1
unet3d_attention/spatial_attention_3_conv1/kernel:0                   [9, 9, 1, 64, 32]     165888
unet3d_attention/spatial_attention_3_conv1/bias:0                     [32]                      32
unet3d_attention/spatial_attention_3_conv1/ins_norm/beta:0            [32]                      32
unet3d_attention/spatial_attention_3_conv1/ins_norm/gamma:0           [32]                      32
unet3d_attention/spatial_attention_3_conv2/kernel:0                   [1, 1, 9, 32, 1]         288
unet3d_attention/spatial_attention_3_conv2/bias:0                     [1]                        1
unet3d_attention/spatial_attention_3_conv2/ins_norm/beta:0            [1]                        1
unet3d_attention/spatial_attention_3_conv2/ins_norm/gamma:0           [1]                        1
unet3d_attention/up_conv1_C2_up2/kernel:0                             [3, 3, 3, 64, 64]     110592
unet3d_attention/up_conv1_C2_up2/bias:0                               [64]                      64
unet3d_attention/up_conv1_C2_up2/ins_norm/beta:0                      [64]                      64
unet3d_attention/up_conv1_C2_up2/ins_norm/gamma:0                     [64]                      64
unet3d_attention/C12_conv/kernel:0                                    [3, 3, 3, 128, 64]    221184
unet3d_attention/C12_conv/bias:0                                      [64]                      64
unet3d_attention/C12_conv/ins_norm/beta:0                             [64]                      64
unet3d_attention/C12_conv/ins_norm/gamma:0                            [64]                      64
unet3d_attention/spatial_attention_2_1_conv1/kernel:0                 [1, 9, 9, 64, 32]     165888
unet3d_attention/spatial_attention_2_1_conv1/bias:0                   [32]                      32
unet3d_attention/spatial_attention_2_1_conv1/ins_norm/beta:0          [32]                      32
unet3d_attention/spatial_attention_2_1_conv1/ins_norm/gamma:0         [32]                      32
unet3d_attention/spatial_attention_2_1_conv2/kernel:0                 [9, 1, 1, 32, 1]         288
unet3d_attention/spatial_attention_2_1_conv2/bias:0                   [1]                        1
unet3d_attention/spatial_attention_2_1_conv2/ins_norm/beta:0          [1]                        1
unet3d_attention/spatial_attention_2_1_conv2/ins_norm/gamma:0         [1]                        1
unet3d_attention/spatial_attention_2_2_conv1/kernel:0                 [9, 1, 9, 64, 32]     165888
unet3d_attention/spatial_attention_2_2_conv1/bias:0                   [32]                      32
unet3d_attention/spatial_attention_2_2_conv1/ins_norm/beta:0          [32]                      32
unet3d_attention/spatial_attention_2_2_conv1/ins_norm/gamma:0         [32]                      32
unet3d_attention/spatial_attention_2_2_conv2/kernel:0                 [1, 9, 1, 32, 1]         288
unet3d_attention/spatial_attention_2_2_conv2/bias:0                   [1]                        1
unet3d_attention/spatial_attention_2_2_conv2/ins_norm/beta:0          [1]                        1
unet3d_attention/spatial_attention_2_2_conv2/ins_norm/gamma:0         [1]                        1
unet3d_attention/spatial_attention_2_3_conv1/kernel:0                 [9, 9, 1, 64, 32]     165888
unet3d_attention/spatial_attention_2_3_conv1/bias:0                   [32]                      32
unet3d_attention/spatial_attention_2_3_conv1/ins_norm/beta:0          [32]                      32
unet3d_attention/spatial_attention_2_3_conv1/ins_norm/gamma:0         [32]                      32
unet3d_attention/spatial_attention_2_3_conv2/kernel:0                 [1, 1, 9, 32, 1]         288
unet3d_attention/spatial_attention_2_3_conv2/bias:0                   [1]                        1
unet3d_attention/spatial_attention_2_3_conv2/ins_norm/beta:0          [1]                        1
unet3d_attention/spatial_attention_2_3_conv2/ins_norm/gamma:0         [1]                        1
unet3d_attention/spatial_attention_3_1_conv1/kernel:0                 [1, 9, 9, 64, 32]     165888
unet3d_attention/spatial_attention_3_1_conv1/bias:0                   [32]                      32
unet3d_attention/spatial_attention_3_1_conv1/ins_norm/beta:0          [32]                      32
unet3d_attention/spatial_attention_3_1_conv1/ins_norm/gamma:0         [32]                      32
unet3d_attention/spatial_attention_3_1_conv2/kernel:0                 [9, 1, 1, 32, 1]         288
unet3d_attention/spatial_attention_3_1_conv2/bias:0                   [1]                        1
unet3d_attention/spatial_attention_3_1_conv2/ins_norm/beta:0          [1]                        1
unet3d_attention/spatial_attention_3_1_conv2/ins_norm/gamma:0         [1]                        1
unet3d_attention/spatial_attention_3_2_conv1/kernel:0                 [9, 1, 9, 64, 32]     165888
unet3d_attention/spatial_attention_3_2_conv1/bias:0                   [32]                      32
unet3d_attention/spatial_attention_3_2_conv1/ins_norm/beta:0          [32]                      32
unet3d_attention/spatial_attention_3_2_conv1/ins_norm/gamma:0         [32]                      32
unet3d_attention/spatial_attention_3_2_conv2/kernel:0                 [1, 9, 1, 32, 1]         288
unet3d_attention/spatial_attention_3_2_conv2/bias:0                   [1]                        1
unet3d_attention/spatial_attention_3_2_conv2/ins_norm/beta:0          [1]                        1
unet3d_attention/spatial_attention_3_2_conv2/ins_norm/gamma:0         [1]                        1
unet3d_attention/spatial_attention_3_3_conv1/kernel:0                 [9, 9, 1, 64, 32]     165888
unet3d_attention/spatial_attention_3_3_conv1/bias:0                   [32]                      32
unet3d_attention/spatial_attention_3_3_conv1/ins_norm/beta:0          [32]                      32
unet3d_attention/spatial_attention_3_3_conv1/ins_norm/gamma:0         [32]                      32
unet3d_attention/spatial_attention_3_3_conv2/kernel:0                 [1, 1, 9, 32, 1]         288
unet3d_attention/spatial_attention_3_3_conv2/bias:0                   [1]                        1
unet3d_attention/spatial_attention_3_3_conv2/ins_norm/beta:0          [1]                        1
unet3d_attention/spatial_attention_3_3_conv2/ins_norm/gamma:0         [1]                        1
unet3d_attention/spatial_attention_4_1_conv1/kernel:0                 [1, 9, 9, 64, 32]     165888
unet3d_attention/spatial_attention_4_1_conv1/bias:0                   [32]                      32
unet3d_attention/spatial_attention_4_1_conv1/ins_norm/beta:0          [32]                      32
unet3d_attention/spatial_attention_4_1_conv1/ins_norm/gamma:0         [32]                      32
unet3d_attention/spatial_attention_4_1_conv2/kernel:0                 [9, 1, 1, 32, 1]         288
unet3d_attention/spatial_attention_4_1_conv2/bias:0                   [1]                        1
unet3d_attention/spatial_attention_4_1_conv2/ins_norm/beta:0          [1]                        1
unet3d_attention/spatial_attention_4_1_conv2/ins_norm/gamma:0         [1]                        1
unet3d_attention/spatial_attention_4_2_conv1/kernel:0                 [9, 1, 9, 64, 32]     165888
unet3d_attention/spatial_attention_4_2_conv1/bias:0                   [32]                      32
unet3d_attention/spatial_attention_4_2_conv1/ins_norm/beta:0          [32]                      32
unet3d_attention/spatial_attention_4_2_conv1/ins_norm/gamma:0         [32]                      32
unet3d_attention/spatial_attention_4_2_conv2/kernel:0                 [1, 9, 1, 32, 1]         288
unet3d_attention/spatial_attention_4_2_conv2/bias:0                   [1]                        1
unet3d_attention/spatial_attention_4_2_conv2/ins_norm/beta:0          [1]                        1
unet3d_attention/spatial_attention_4_2_conv2/ins_norm/gamma:0         [1]                        1
unet3d_attention/spatial_attention_4_3_conv1/kernel:0                 [9, 9, 1, 64, 32]     165888
unet3d_attention/spatial_attention_4_3_conv1/bias:0                   [32]                      32
unet3d_attention/spatial_attention_4_3_conv1/ins_norm/beta:0          [32]                      32
unet3d_attention/spatial_attention_4_3_conv1/ins_norm/gamma:0         [32]                      32
unet3d_attention/spatial_attention_4_3_conv2/kernel:0                 [1, 1, 9, 32, 1]         288
unet3d_attention/spatial_attention_4_3_conv2/bias:0                   [1]                        1
unet3d_attention/spatial_attention_4_3_conv2/ins_norm/beta:0          [1]                        1
unet3d_attention/spatial_attention_4_3_conv2/ins_norm/gamma:0         [1]                        1
unet3d_attention/spatial_attention_5_1_conv1/kernel:0                 [1, 9, 9, 64, 32]     165888
unet3d_attention/spatial_attention_5_1_conv1/bias:0                   [32]                      32
unet3d_attention/spatial_attention_5_1_conv1/ins_norm/beta:0          [32]                      32
unet3d_attention/spatial_attention_5_1_conv1/ins_norm/gamma:0         [32]                      32
unet3d_attention/spatial_attention_5_1_conv2/kernel:0                 [9, 1, 1, 32, 1]         288
unet3d_attention/spatial_attention_5_1_conv2/bias:0                   [1]                        1
unet3d_attention/spatial_attention_5_1_conv2/ins_norm/beta:0          [1]                        1
unet3d_attention/spatial_attention_5_1_conv2/ins_norm/gamma:0         [1]                        1
unet3d_attention/spatial_attention_5_2_conv1/kernel:0                 [9, 1, 9, 64, 32]     165888
unet3d_attention/spatial_attention_5_2_conv1/bias:0                   [32]                      32
unet3d_attention/spatial_attention_5_2_conv1/ins_norm/beta:0          [32]                      32
unet3d_attention/spatial_attention_5_2_conv1/ins_norm/gamma:0         [32]                      32
unet3d_attention/spatial_attention_5_2_conv2/kernel:0                 [1, 9, 1, 32, 1]         288
unet3d_attention/spatial_attention_5_2_conv2/bias:0                   [1]                        1
unet3d_attention/spatial_attention_5_2_conv2/ins_norm/beta:0          [1]                        1
unet3d_attention/spatial_attention_5_2_conv2/ins_norm/gamma:0         [1]                        1
unet3d_attention/spatial_attention_5_3_conv1/kernel:0                 [9, 9, 1, 64, 32]     165888
unet3d_attention/spatial_attention_5_3_conv1/bias:0                   [32]                      32
unet3d_attention/spatial_attention_5_3_conv1/ins_norm/beta:0          [32]                      32
unet3d_attention/spatial_attention_5_3_conv1/ins_norm/gamma:0         [32]                      32
unet3d_attention/spatial_attention_5_3_conv2/kernel:0                 [1, 1, 9, 32, 1]         288
unet3d_attention/spatial_attention_5_3_conv2/bias:0                   [1]                        1
unet3d_attention/spatial_attention_5_3_conv2/ins_norm/beta:0          [1]                        1
unet3d_attention/spatial_attention_5_3_conv2/ins_norm/gamma:0         [1]                        1
unet3d_attention/final/kernel:0                                       [3, 3, 3, 128, 4]      13824
unet3d_attention/final/bias:0                                         [4]                        4
Total #vars=254, #params=11090241, size=42.31MB

