from keras.models import load_model
import cv2
import logging
from os import listdir
from os.path import isdir
from numpy import asarray
from numpy import savez_compressed
from numpy import load
from numpy import expand_dims
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
logger = logging.getLogger(__name__)
tf.logging.set_verbosity(tf.logging.ERROR)

def extract_face(input_path, required_size=(160,160)):
    img = cv2.imread(input_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=5)
    max = 0
    biggest_face = []
    for x, y, w, h in faces:
        if (w * h) > max:
            max = w * h
            biggest_face = [x, y, x + w, y + h]
    if (len(biggest_face) > 0):
        crop_img = img[biggest_face[1]:biggest_face[3], biggest_face[0]:biggest_face[2]]
        resized = cv2.resize(crop_img, required_size)
        return resized
    else:
        logger.info("No face detected.")
        return None

def load_faces(directory):
    faces = list()
    for filename in listdir(directory):
        path = directory + filename
        face = extract_face(path)
        if face is not None:
            faces.append(face)
    return faces


def load_dataset(directory):
    X, y = list(), list()
    # enumerate folders, on per class
    for subdir in listdir(directory):
        # path
        path = directory + subdir + '/'
        # skip any files that might be in the dir
        if not isdir(path):
            continue
        # load all faces in the subdirectory
        faces = load_faces(path)
        # create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        # store
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)

def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tr_X, tr_y = load_dataset('5-celebrity-faces-dataset/train/')
    print(tr_X.shape, tr_y.shape)
    te_X, te_y = load_dataset('5-celebrity-faces-dataset/val/')
    print(te_X.shape, te_y.shape)
    savez_compressed('5-celebrity-faces-dataset.npz', tr_X, tr_y, te_X, te_y)

    data = load('5-celebrity-faces-dataset.npz')
    train_X, train_y, test_X, test_y = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print('Loaded: ', train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    model = load_model('facenet_keras.h5')
    print('Loaded Model')
    newTrainX = list()
    for face_pixels in train_X:
        embedding = get_embedding(model, face_pixels)
        newTrainX.append(embedding)
    newTrainX = asarray(newTrainX)
    print(newTrainX.shape)
    # convert each face in the test set to an embedding
    newTestX = list()
    for face_pixels in test_X:
        embedding = get_embedding(model, face_pixels)
        newTestX.append(embedding)
    newTestX = asarray(newTestX)
    print(newTestX.shape)
    # save arrays to one file in compressed format
    savez_compressed('5-celebrity-faces-embeddings.npz', newTrainX, train_y, newTestX, test_y)

    data = load('5-celebrity-faces-embeddings.npz')
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))

    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)

    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)

    # fit model
    model = SVC(kernel='linear')
    model.fit(trainX, trainy)

    # predict
    yhat_train = model.predict(trainX)
    yhat_test = model.predict(testX)
    # score
    score_train = accuracy_score(trainy, yhat_train)
    score_test = accuracy_score(testy, yhat_test)
    # summarize
    print('Accuracy: train=%.3f, test=%.3f' % (score_train * 100, score_test * 100))

    # folder = '5-celebrity-faces-dataset/train/ben_afflek/'
    # if not os.path.exists(folder):
    #     print('folder not found')
    # i = 1
    # # enumerate files
    # for filename in listdir(folder):
    #     # path
    #     path = folder + filename
    #     # get face
    #     face = extract_face(path)
    #     if face is not None:
    #         print(i, face.shape)
    #         # plot
    #         pyplot.subplot(2, 7, i)
    #         pyplot.axis('off')
    #         pyplot.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    #         i += 1
    # pyplot.show()
