import keras
from keras import layers

def build_model(args):
    '''
    Build a simple Convolutional Neural Network (CNN) for amino acid strings.
    '''
    
    # 1. TextVectorization layer (converts strings to token sequences)
    text_vectorization = layers.TextVectorization(
        max_tokens=args.tokenizer_max_tokens,
        split=args.tokenizer_split,
        output_sequence_length=args.tokenizer_output_sequence_length,
    )

    # 2. Input layer (one string per example)
    inputs = keras.Input(shape=(args.input_shape0[0],), dtype='string', name='input')
    
    # 3. Apply vectorization and embedding
    x = text_vectorization(inputs)
    x = layers.Embedding(
        input_dim=args.tokenizer_max_tokens,
        output_dim=args.embedding_dimensions,
        name='embedding'
    )(x)
    
    # 4. A simple CNN block
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', name='conv1d')(x)
    x = layers.GlobalMaxPooling1D(name='global_max_pooling')(x)
    
    # 5. Final Output layer
    outputs = layers.Dense(args.output_shape0[-1], name='output')(x)
    
    model = keras.Model(inputs, outputs, name='amino_simple_cnn')

    # 6. Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=args.loss,
        metrics=args.metrics or []
    )
    
    # 7. Return the model and the vectorization layer (so the engine can adapt it)
    return model, text_vectorization
