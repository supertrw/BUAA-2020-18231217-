import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Sequential
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn import preprocessing

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

df = pd.read_csv('../data/train.csv')
df = df.drop(['Unnamed: 0'],axis=1)
df.columns = ["review","sentiment"]
print(df['sentiment'].value_counts())
print()
print(df.head())
print()
print(df.info())

plt.figure(figsize=(10, 6))
plt.hist([len(sample) for sample in list(df['review'])], 50)
plt.xlabel('Length of review')
plt.ylabel('Number of review')
plt.title('train Review length')
plt.savefig('../picture/len.png')
plt.show()

label_encoder = preprocessing.LabelEncoder()
df['sentiment'] = label_encoder.fit_transform(df['sentiment'])
print(df.head())
df['Processed_Reviews'] = df.review.apply(lambda x: clean_text(x))
print(df['Processed_Reviews'].loc[1])
print(df.head())
df.Processed_Reviews.apply(lambda x: len(x.split(" "))).mean()

max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['Processed_Reviews'])
list_tokenized_train = tokenizer.texts_to_sequences(df['Processed_Reviews'])

X_t = pad_sequences(list_tokenized_train, maxlen=None)
y = df['sentiment']

embed_size = 128
model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(Bidirectional(LSTM(32, return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.05))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 200
epochs = 5
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

score = model.evaluate(X_t, y, batch_size=batch_size)
print(score)

model.save('IMDB_model.h5')

df_test=pd.read_csv("../data/test_data.csv")
label = df_test['Unnamed: 0']
df_test = df_test.drop(['Unnamed: 0'],axis=1)
df_test.columns = ["review","sentiment"]

plt.figure(figsize=(10, 6))
plt.hist([len(sample) for sample in list(df_test['review'])], 50)
plt.xlabel('Length of review')
plt.ylabel('Number of review')
plt.title('test Review length')
plt.savefig('../picture/len_test.png')
plt.show()

df_test["review"]=df_test.review.apply(lambda x: clean_text(x))
list_sentences_test = df_test["review"]
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = pad_sequences(list_tokenized_test, maxlen=None)

prediction = model.predict(X_te)
y_pred = []
for i in prediction:
    if i > 0.5:
        y_pred.append("positive")
    else:
        y_pred.append("negative")
submission = pd.DataFrame({'':label, 'sentiment':y_pred})
submission.to_csv('../data/submission.csv', index=False)

