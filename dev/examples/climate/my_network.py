import keras
from keras import layers

def build_model(args):
    # args.input_shape0 is [24, 5, 5, 2]
    # args.output_shape0 is [24, 5, 5, 1]
    
    inputs = keras.Input(shape=args.input_shape0)
    
    # Simple 3D Convolution that processes time + spatial dimensions
    x = layers.Conv3D(filters=16, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.Conv3D(filters=8, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv3D(filters=4, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.Conv3D(filters=1, kernel_size=3, padding='same', activation='linear')(x)
    
    model = keras.Model(inputs=inputs, outputs=x, name="simple_4d_conv")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=args.loss,
        metrics=args.metrics
    )
    
    return model
