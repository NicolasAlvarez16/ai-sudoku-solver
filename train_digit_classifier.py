from py_image_search.models.sudokunet import SudokuNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import argparse

# Construc the argument parser and parse the arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-m', '--model', required=True, help='path to the output model after training')
args = vars(arg_parser.parse_args())

LEARNING_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 128

print('[INFO] Accessing MNNIST....')
((train_data, train_labels), (test_data, test_labels)) = mnist.load_data() # Get dataset

# Add a color scale dimension to the digits 
train_data = train_data.reshape((train_data.shape[0], 28, 28, 1))
test_data = test_data.reshape((test_data.shape[0], 28, 28, 1))

# Scale data to the range [0, 1]
train_data = train_data.astype('float32') / 255.0
test_data = test_data.astype('float32') / 255.0

# Conert the labels from integers to vectors (One hot encoding)
label_bin = LabelBinarizer()
train_labels = label_bin.fit_transform(train_labels)
test_labels = label_bin.transform(test_labels)

# Initialise model and optimizer
print('[INFO] Compiling model...')
opt = Adam(lr=LEARNING_RATE)
model = SudokuNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Train network
print('[INFO] Training network...')
history = model.fit(train_data, train_labels, validation_data=(test_data, test_labels), batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

# Evalue the network
print('[INFO] Evalue the network...')
predictions = model.predict(test_data)
print(classification_report(test_labels.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in label_bin.classes_]))

# Serialise the model to the disk
print('[INFO] Serialising digit model...')
model.save(args['model'], save_format='h5')