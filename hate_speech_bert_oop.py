import math
import os
import re
import warnings
from os import listdir
#import tqdm
import bert
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf
from bert import BertModelLayer
from bert.loader import StockBertConfig
from bert.loader import load_stock_weights
from bert.loader import map_stock_config_to_params
from imblearn.keras import balanced_batch_generator
from scikitplot.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tqdm import tqdm

FullTokenizer = bert.bert_tokenization.FullTokenizer
warnings.simplefilter(action='ignore', category=FutureWarning)


# stopwords = nltk.corpus.stopwords.words('english')



def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    RE_EMOJI = re.compile('([&]).*?(;)')
    parsed_text = RE_EMOJI.sub(r'', parsed_text)
    #parsed_text = parsed_text.code("utf-8", errors='ignore')
    return parsed_text



class HatebaseTwitter():
    def __init__(self, data_path, max_seq_len = 80, data_column="tweet", label_column='class'):
        self.MAX_SEQ_LEN = max_seq_len
        # Reading the Data Frame from the Hatebase Twitter CSV File
        self.df = pd.read_csv(data_path)
        self.data_column = data_column
        self.label_column = label_column


        # os.system()
        # !mkdir -p .model .model/$bert_model_name

        # self.vocab = pd.read_csv(".model/uncased_L-12_H-768_A-12/vocab.txt", sep=",,,",header=None)
        # self._bert_ckpt_dir = os.path.join(".model/", bert_model_name)
        # self._bert_ckpt_file = os.path.join(self._bert_ckpt_dir, "bert_model.ckpt")
        # self._bert_config_file = os.path.join(self._bert_ckpt_dir, "bert_config.json")

    @staticmethod
    def get_masks(tokens, max_seq_length):
        return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))

    @staticmethod
    def get_segments(tokens, max_seq_length):
        """Segments: 0 for the first sequence, 1 for the second"""
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments + [0] * (max_seq_length - len(tokens))

    @staticmethod
    def get_ids(tokens, tokenizer, max_seq_length):
        """Token ids from Tokenizer vocab"""
        token_ids = tokenizer.convert_tokens_to_ids(tokens, )
        input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
        return input_ids

    def create_single_input(self,sentence, MAX_LEN):

        stokens = self.tokenizer.tokenize(sentence)

        stokens = stokens[:MAX_LEN]

        stokens = ["[CLS]"] + stokens + ["[SEP]"]

        ids = self.get_ids(stokens, self.tokenizer, self.MAX_SEQ_LEN)
        masks = self.get_masks(stokens, self.MAX_SEQ_LEN)
        segments = self.get_segments(stokens, self.MAX_SEQ_LEN)

        return ids, masks, segments

    def create_input_array(self, sentences):

        input_ids, input_masks, input_segments = [], [], []

        for sentence in tqdm(sentences, position=0, leave=True):
            ids, masks, segments = self.create_single_input(sentence, self.MAX_SEQ_LEN - 2)

            input_ids.append(ids)
            input_masks.append(masks)
            input_segments.append(segments)

        return [np.asarray(input_ids, dtype=np.int32),
                np.asarray(input_masks, dtype=np.int32),
                np.asarray(input_segments, dtype=np.int32)]

    def bert_tokenize(self, test_size=0.3, max_seq_len=1024, verbose=False):
        """Converts the input textual data into tokenized array to be used on the input of BERT neural network"""
        #self._tokenizer = FullTokenizer(vocab_file=os.path.join(self._bert_ckpt_dir, "vocab.txt"))
        self.df[self.data_column] = self.df[self.data_column].apply(lambda x: preprocess(x))

        vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()

        do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()
        self.tokenizer = FullTokenizer(vocab_file, do_lower_case)
        self.df[self.data_column] = self.df[self.data_column].apply(lambda x: preprocess(x))
        X_train, X_test, y_train, y_test = train_test_split(self.df.drop(self.label_column,axis=1), self.df[self.label_column],
                                                            test_size=test_size,
                                                            random_state=100, stratify=self.df[self.label_column])
        train = pd.concat([X_train, y_train], axis=1).reset_index().drop('index', axis=1)
        test = pd.concat([X_test, y_test], axis=1).reset_index().drop('index', axis=1)

        input_test = self.create_input_array(X_train)
        input_train = self.create_input_array(X_test)

        self.train = input_train
        self.test = input_test


        # if verbose:
        #     print(f"Train data shape: {train.shape}")
        #     print(f"Train data shape: {test.shape}")
        # train, test = map(lambda df: df.reindex(df[self.data_column].str.len().sort_values().index),
        #                   [train, test])
        # print(train.iloc[0])
        # self.train = train
        # self.test = test
        # ((self.train_x, self.train_y),
        #  (self.test_x, self.test_y)) = map(self._prepare, [train, test])
        #
        # if verbose: print("max seq_len", self.max_seq_len)
        # self.max_seq_len = min(self.max_seq_len, max_seq_len)
        # ((self.train_x, self.train_x_token_types),
        #  (self.test_x, self.test_x_token_types)) = map(self._pad,
        #                                                [self.train_x, self.test_x])




    def create_learning_rate_scheduler(self, max_learn_rate=5e-5,
                                       end_learn_rate=1e-7,
                                       warmup_epoch_count=10,
                                       total_epoch_count=90):

        def lr_scheduler(epoch):
            if epoch < warmup_epoch_count:
                res = (max_learn_rate / warmup_epoch_count) * (epoch + 1)
            else:
                res = max_learn_rate * math.exp(
                    math.log(end_learn_rate / max_learn_rate) * (epoch - warmup_epoch_count + 1) / (
                                total_epoch_count - warmup_epoch_count + 1))
            return float(res)

        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)

        return learning_rate_scheduler



    def create_model(self, type: str, adapter_size=None):
        """Creates a classification model. Input parameters:
         type: "binary" to build a model for binary classification, "multi" for multiclass classification. """
        self.type = type
        # adapter_size = 64  # see - arXiv:1902.00751
        if type == 'binary':
            class_count = 2
        elif type == 'multi':
            class_count = 3
        else:
            raise TypeError("Choose a proper type of classification")
        # create the bert layer

        bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1", trainable=True)
        self.bert_layer = bert_layer
        input_word_ids = tf.keras.layers.Input(shape=(self.MAX_SEQ_LEN,), dtype=tf.int32,
                                               name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(self.MAX_SEQ_LEN,), dtype=tf.int32,
                                           name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(self.MAX_SEQ_LEN,), dtype=tf.int32,
                                            name="segment_ids")

        bert_inputs = [input_word_ids, input_mask, segment_ids]

        bert_output = bert_layer(bert_inputs)

        x = tf.keras.layers.Flatten()(bert_output[0])

        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(bert_output[0])
        x = tf.keras.layers.Dropout(0.3)(x)
        out = tf.keras.layers.Dense(3, activation="softmax", name="dense_output")(x)

        model = tf.keras.models.Model(
            inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['sparse_categorical_accuracy'])

        model.summary()
        self.model = model

    def eda(self):
        # Seeing the Number of CrowdFlower Annotators that Decided On a Tweet's Label
        annot_count = self.df['count'].value_counts().to_dict()

        ## Visualizing the Annotator Count
        plt.figure(figsize=(12,12))
        plt.bar(list(annot_count.keys()), list(annot_count.values()))
        plt.xlabel('Number of CrowdFlower Annotators', size=14)
        plt.ylabel('Number of Tweets Annotated', size=14)
        plt.show()

        # Visualizing the Proportion of Tweet Classification Labels and Class Imbalance
        labels = ['Hateful', 'Offensive', 'Neither']
        class_vals = self.df['class'].value_counts().to_dict()
        class_vals = {labels[i]: class_vals[i] for i in class_vals.keys()}
        fig, ax = plt.subplots(figsize=(12,12))
        ax.pie(class_vals.values(), labels=class_vals.keys(), autopct='%1.1f%%')
        ax.axis('equal')
        plt.title("Proportion of Tweet Classes", size=14)
        plt.show()

    def classify(self,total_epoch_count = 30, warmup_epoch_count = 10):#, X, type: str, classifier: str, test_prop: float, res: None, res_method: None):

        if self.type == "binary":
            self.train_y[np.where(self.train_y == 1)] = 0
            self.train_y[np.where(self.train_y == 2)] = 1
            self.test_y[np.where(self.test_y == 1)] = 0
            self.test_y[np.where(self.test_y == 2)] = 1

        #log_dir = ".log/movie_reviews/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
        #tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
        training_generator, steps_per_epoch = balanced_batch_generator(self.train_x, self.train_y,
                                                                       batch_size=48,
                                                                       random_state=100)
        #total_epoch_count = 30
        # model.fit(x=(data.train_x, data.train_x_token_types), y=data.train_y,
        self.model.fit(training_generator,
                  epochs=total_epoch_count,
                  steps_per_epoch=steps_per_epoch,
                  # validation_split=0.1,
                  callbacks=[  # keras.callbacks.LearningRateScheduler(time_decay,verbose=1),
                      # lrate,
                      self.create_learning_rate_scheduler(max_learn_rate=1e-5,
                                                     end_learn_rate=5e-8,
                                                     warmup_epoch_count=warmup_epoch_count,
                                                     total_epoch_count=total_epoch_count)
                      #,

                      #keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
                  #    tensorboard_callback
                  ])

        self.model.save_weights('./movie_reviews.h5', overwrite=True)
        Y_pred_probabilities = self.model.predict(self.test_x)
        Y_pred = np.argmax(Y_pred_probabilities,axis=-1)
        self.pred_y = Y_pred
        # Accuracy Percentage
        print(f"Accuracy is {round(accuracy_score(self.test_y, Y_pred), 2)*100}%")

        # Classification Report
        print(classification_report(Y_pred, self.test_y))

        # Matthew's Correlation Coefficient
        print(f"Matthew's Correlation Coefficient is {matthews_corrcoef(self.test_y, Y_pred)}")

        # Plots of Confusion Matrix and ROC Curve
        plot_confusion_matrix(self.test_y, Y_pred, figsize=(10, 10))
        #
        # return model

    def tokenized_text(self, idx, bucket="train"):
        if bucket == 'train':
            df = self.train
            tokens = self.train_x
        elif bucket == 'test':
            df = self.test
            tokens = self.test_x
        else:
            raise TypeError("Choose a proper type of classification")
        indices = np.atleast_1d(idx)
        df_part = df.iloc[indices]
        tokenized_text = pd.Series(list(tokens[indices.reshape((len(indices), 1))]), index=df_part.index)
        #tokenized_text = tokenized_text[tokenized_text!=0]
        #return tokenized_text
        df_part['tokens'] = tokenized_text.apply(lambda x: x[x>0])
        df_part['bert_words'] = df_part['tokens'].apply(lambda x: [self.vocab.iloc[idx][0] for idx in x])
        # for row in df_part[['tweet','tokens','bert_words']].itertuples():
        #     print(row.tweet)
        #     print(row.tokens)
        #     print(row.bert_words)
        #print(df_part.to_string())
        return df_part

    def get_miss_classified(self):
        df = self.test
        predicted_class = pd.Series(self.pred_y,index=df.index)
        df["class_predicted"] = predicted_class
        print("INDICES \n \n \n")
        indices = np.argwhere(df['class']!=df['class_predicted']).flatten()
        print(len(indices))
        print(indices)
        df = self.tokenized_text(list(indices),"test")
        print(df[df['class']!=df['class_predicted']].shape)
        df = df[df['class']!=df['class_predicted']]

        return df
