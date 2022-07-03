def loadModel(model_path):
	base_model = ResNet34()
	inputs = base_model.inputs[0]
	arcface_model = base_model.outputs[0]
	arcface_model = layers.BatchNormalization(momentum=0.9, epsilon=2e-5)(arcface_model)
	arcface_model = layers.Dropout(0.4)(arcface_model)
	arcface_model = layers.Flatten()(arcface_model)
	arcface_model = layers.Dense(512, activation=None, use_bias=True, kernel_initializer="glorot_normal")(arcface_model)
	embedding = layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name="embedding", scale=True)(arcface_model)
	model = keras.models.Model(inputs, embedding, name=base_model.name)

	model.load_weights(model_path)

	return model