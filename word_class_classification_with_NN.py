import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout


def get_labels_and_words(file):
    labels = []
    words = []
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            line_sp = line.split()
            labels.append(line_sp[0])
            words.append(line_sp[1])
    return labels, words


train_labels, train_words = get_labels_and_words('train.txt')
test_labels, test_words = get_labels_and_words('test.txt')

print('train_labels: ' + train_labels[0], train_labels[1], train_labels[2])
print('train_words: ' + train_words[0], train_words[1], train_words[2])


# noun=0, verb=1
def convert_labels(labels):
    converted_labels = []
    for label in labels:
        if label == "noun":
            converted_labels.append(0)
        else:
            converted_labels.append(1)
    return converted_labels


train_y = convert_labels(train_labels)
test_y = convert_labels(test_labels)

unique_chars = []
for word in train_words:
    for char in word:
        if char not in unique_chars:
            unique_chars.append(char)

char_int = {}
i = 0
for word in train_words:
    for char in word:
        if char not in char_int:
            char_int[char] = i
            i +=1
char_int['unknown'] = i

print(len(unique_chars))
print('int_for_umlaut:')
print(char_int['ä'])
print('int_for_unknown:')
print(char_int['unknown'])


def map_words(words):
    mapped_words = []
    for w in words:
        temp = []
        for char1 in w:
            if char1 in char_int:
                temp.append(char_int[char1])
            else:
                temp.append(char_int['unknown'])
        mapped_words.append(temp)
    return mapped_words


def get_longest_word(words):
    longest = words[0]
    for w1 in words:
        if len(w1) > len(longest):
            longest = w1
    return longest


print('longest_train: ')
print(get_logest_word(train_words))
print('longest_test: ')
print(get_logest_word(test_words))

longest_word = max(len(get_logest_word(train_words)), len(get_logest_word(test_words)))

train_x = keras.preprocessing.sequence.pad_sequences(map_words(train_words), maxlen=longest_word, dtype='int32',
                                                     padding='pre', truncating='pre', value=0.0)
test_x = keras.preprocessing.sequence.pad_sequences(map_words(test_words), maxlen=longest_word, dtype='int32',
                                                    padding='pre', truncating='pre', value=0.0)

print("train_prepended_by_zeros: ",train_x[0], train_x[1])
print("test_prepended_by_zeros: ", test_x[0], test_x[1])


def one_hot_encode(char_no):
    return [1 if i == char_no else 0 for i in range(len(unique_chars))]


def get_onehot(array_x):
    onehot = []
    for row in array_x:
        tmp = []
        for char_no in row:
            tmp+=one_hot_encode(char_no)
        onehot.append(tmp)
    return onehot


test_onehot = get_onehot(test_x)
train_onehot = get_onehot(train_x)

print('train_one_hot: ',  train_onehot[0])
print('test_one_hot: ', test_onehot[0])

train_onehot_np = np.array(train_onehot)
train_y_np = np.array(train_y)

input_dim = len(unique_chars) * longest_word

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=input_dim))
model.add(Dropout(0.3))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_onehot_np, train_y_np, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

test_onehot_np = np.array(test_onehot)
test_y_np = np.array(test_y)

score = model.evaluate(test_onehot_np, test_y_np)

print('score: ', score)
