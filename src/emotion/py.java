#SETUP AND IMPORT
!pip install nlp

        %matplotlib inline

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nlp
import random


def show_history(h):
        epochs_trained = len(h.history['loss'])
        plt.figure(figsize=(16, 6))

        plt.subplot(1, 2, 1)
        plt.plot(range(0, epochs_trained), h.history.get('accuracy'), label='Training')
        plt.plot(range(0, epochs_trained), h.history.get('val_accuracy'), label='Validation')
        plt.ylim([0., 1.])
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(0, epochs_trained), h.history.get('loss'), label='Training')
        plt.plot(range(0, epochs_trained), h.history.get('val_loss'), label='Validation')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


        def show_confusion_matrix(y_true, y_pred, classes):
        from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, normalize='true')

            plt.figure(figsize=(8, 8))
            sp = plt.subplot(1, 1, 1)
            ctx = sp.matshow(cm)
            plt.xticks(list(range(0, 6)), labels=classes)
            plt.yticks(list(range(0, 6)), labels=classes)
            plt.colorbar(ctx)
            plt.show()


            print('Using TensorFlow version', tf.__version__)

#This code is for downloading the emotion dataset as the existing dataset code in not working
            #3STEP
            pip install -U datasets

            !pip install datasets==2.9.0

            from datasets import load_dataset
emotions_ds = load_dataset("emotion")

        emotions_ds

        train = emotions_ds['train']
        validate= emotions_ds['validation']
        test = emotions_ds['test']

        def get_tweet(data):
        tweets =[x['text'] for x in data]
        labels=[x['label'] for x in data]
        return tweets,labels

        tweets,labels=get_tweet(train)
        tweets[0], labels[0]

#4STEP
        from tensorflow.keras.preprocessing.text import Tokenizer
        tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
                tokenizer.fit_on_texts(tweets)

#5 Step

                lengths = [len(t.split(' ')) for t in tweets]

                plt.hist(lengths, bins=len(set(lengths)))
                plt.show()

                maxlen = 50
                from tensorflow.keras.preprocessing.sequence import pad_sequences

        from os import truncate
def get_sequences(tokenizer,tweets):
        sequences= tokenizer.texts_to_sequences(tweets)
        padded=pad_sequences(sequences, truncating='post', padding='post',maxlen=maxlen)
        return padded

        padded_train_seq=get_sequences(tokenizer,tweets)

        padded_train_seq[0]

#6 Step

        classes = {'anger', 'fear', 'joy', 'love', 'sadness', 'surprise'}
        print(classes)

        classes_to_index = dict((c, i) for i, c in enumerate(classes))
        index_to_classes = dict((v, k) for k, v in classes_to_index.items())

        index_to_classes

        train_labels = np.array(labels)

        print(train_labels[0])

        plt.hist(list(map(index_to_classes.get, labels)), bins=11)
        plt.show()

#7 Step

        model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(10000, 16, input_length=50),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
        tf.keras.layers.Dense(6, activation='softmax')
        ])

        model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
        )

        model.summary()

#8 Step

        val_tweets, val_labels = get_tweet(validate)
        val_sequences = get_sequences(tokenizer, val_tweets)
        val_labels = np.array(val_labels)

        val_tweets[0], val_labels[0]

        h = model.fit(
        padded_train_seq, train_labels,
        validation_data=(val_sequences, val_labels),
        epochs=20,
        callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)
        ]
        )

# 9 step

        show_history(h)

        test_tweets, test_labels = get_tweet(test)
        test_sequences = get_sequences(tokenizer, test_tweets)
        test_labels = np.array(test_labels)

        _ = model.evaluate(test_sequences, test_labels)

        i = random.randint(0, len(test_labels) - 1)
        print(f'Sentence: {test_tweets[i]}')
        print(f'Emotion: {index_to_classes[test_labels[i]]}')
        p = np.argmax(model.predict(np.expand_dims(test_sequences[i], axis=0)), axis=-1)[0]
        print(f'Predicted Emotion: {index_to_classes.get(p)}')

        preds = np.argmax(model.predict(test_sequences), axis=-1)
        preds.shape, test_labels.shape

        show_confusion_matrix(test_labels, preds, list(classes))