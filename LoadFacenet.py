from keras.models import load_model

model = load_model('facenet_keras.h5')
print(model.inputs)
print(model.outputs)