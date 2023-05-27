import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import TFAutoModel
from transformers import AutoTokenizer
import tensorflow as tf

SEQ_LEN = 50  # we will cut/pad our sequences to a length of 50 tokens

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')


def data_concentrating(df):
    arr = df['Метка'].values

    labels = tf.keras.utils.to_categorical(arr, num_classes=5)

    with open('labels.npy', 'wb') as f:
        np.save(f, labels)

    return labels


def model_defined(train, val):
    bert = TFAutoModel.from_pretrained("bert-base-cased")
    input_ids = tf.keras.layers.Input(shape=(SEQ_LEN,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(SEQ_LEN,), name='attention_mask', dtype='int32')

    # we consume the last_hidden_state tensor from bert (discarding pooled_outputs)
    embeddings = bert(input_ids, attention_mask=mask)[0]

    X = tf.keras.layers.LSTM(64)(embeddings)
    X = tf.keras.layers.BatchNormalization()(X)
    X = tf.keras.layers.Dense(128, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.1)(X)
    X = tf.keras.layers.Dense(32, activation='relu')(X)
    y = tf.keras.layers.Dense(5, activation='softmax', name='outputs')(X)

    # define input and output layers of our model
    model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

    # freeze the BERT layer - otherwise we will be training 100M+ parameters...
    model.layers[2].trainable = False
    model.summary()
    optimizer = tf.keras.optimizers.Adam(0.01)
    loss = tf.keras.losses.CategoricalCrossentropy()  # categorical = one-hot
    acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

    model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

    history = model.fit(train, validation_data=val, epochs=30)
    # print(history)
    model.summary()
    model.save()
    return model


# initialize two arrays for input tensors
def tokenize(sentence):
    tokens = tokenizer.encode_plus(sentence, max_length=SEQ_LEN,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_attention_mask=True,
                                   return_token_type_ids=False, return_tensors='tf')

    return tokens['input_ids'], tokens['attention_mask']


def read_excel_with_pandas(path_to_excel_file):
    excel_file = pd.read_excel(path_to_excel_file)

    return excel_file


def excel_parse(excel_file):
    new_df_for_first_class = excel_file[excel_file['Метка'] == 0]
    new_df_for_first_class = new_df_for_first_class.iloc[:2000]
    new_df_for_second_class = excel_file[excel_file['Метка'] == 1]
    new_df_for_second_class = new_df_for_second_class.iloc[:2000]
    new_df_for_third_class = excel_file[excel_file['Метка'] == 2]
    new_df_for_fourth_class = excel_file[excel_file['Метка'] == 3]
    new_df_for_fives_class = excel_file[excel_file['Метка'] == 4]

    list_for_df = [new_df_for_first_class,
                   new_df_for_second_class,
                   new_df_for_third_class,
                   new_df_for_fourth_class,
                   new_df_for_fives_class]

    df = pd.concat(list_for_df)
    df = df.iloc[np.random.permutation(len(df))]
    df = df.sample(frac=1).reset_index()

    return df


def train_test_split_data(df):
    train, test = train_test_split(df, test_size=0.2, random_state=0)

    return train, test


def function_help_for_tokenize(df):
    Xids = np.zeros((len(df), SEQ_LEN))
    Xmask = np.zeros((len(df), SEQ_LEN))

    for i, sentence in enumerate(df['Комментарий']):
        Xids[i, :], Xmask[i, :] = tokenize(sentence)
        if i % 10000 == 0:
            print(i)  # do this so we can see some progress

    with open('xids.npy', 'wb') as f:
        np.save(f, Xids)
    with open('xmask.npy', 'wb') as f:
        np.save(f, Xmask)

    return Xids, Xmask


def map_func(input_ids, masks, labels):
    return {'input_ids': input_ids, 'attention_mask': masks}, labels


if __name__ == '__main__':
    df = read_excel_with_pandas(r"expf3.xlsx")
    train = excel_parse(df)
    Xids, Xmask = function_help_for_tokenize(train)
    labels = data_concentrating(train)
    print(tf.config.experimental.list_physical_devices('CPU'))
    dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))
    dataset = dataset.map(map_func)
    dataset = dataset.shuffle(100000).batch(32)
    DS_LEN = len([0 for batch in dataset])
    SPLIT = 0.9  # 90-10 split

    train = dataset.take(round(DS_LEN * SPLIT))  # get first 90% of batches
    val = dataset.skip(round(DS_LEN * SPLIT))  #
    print(val, train)
    model = model_defined(train, val)
    print(model)
    model.save('best_model')
