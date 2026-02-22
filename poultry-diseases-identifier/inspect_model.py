import tensorflow as tf
import os

try:
    model_path = "./model/mobilenetV2/mobilenetv2.h5"
    
    class CustomDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, **kwargs):
            kwargs.pop('groups', None)
            super().__init__(**kwargs)

    model = tf.keras.models.load_model(
        model_path, 
        compile=False, 
        custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D}
    )
    
    print(f"Model Input Shape: {model.input_shape}")
    
except Exception as e:
    print(f"Error: {e}")
