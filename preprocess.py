import treform as ptm
import pandas as pd

valid_size = 0.2
random_seed = 777
stopwords = './stopwords/stopwordsKor.txt'
train = "./data/ratings_train.txt"
test = "./data/ratings_test.txt"


def preprocess(df, pipe):

    labels, documents = df.label, df.document
    corpus = ptm.Corpus(textList=documents)
    documents = pipe.processCorpus(corpus)
    documents = [" ".join(word for sent in doc for word in sent) for doc in documents]

    return pd.DataFrame(dict(document=documents, label=labels))


if __name__ == "__main__":

    # build preprocessing pipeline
    pipeline = ptm.Pipeline(
        ptm.splitter.NLTK(),
        ptm.tokenizer.TwitterKorean(),
        ptm.lemmatizer.SejongPOSLemmatizer(),
        ptm.helper.SelectWordOnly(),
        ptm.helper.StopwordFilter(file=stopwords)
    )

    # prepare dataset
    train_df = pd.read_csv(train, sep="\t").dropna()
    valid_df = train_df.sample(frac=valid_size, random_state=random_seed)
    train_df = train_df.drop(valid_df.index)
    test_df = pd.read_csv(test, sep="\t").dropna()

    # preprocessing
    train_df_pp = preprocess(train_df, pipeline)
    valid_df_pp = preprocess(valid_df, pipeline)
    test_df_pp = preprocess(test_df, pipeline)

    # save
    train_df_pp.to_csv("./data/train.txt", index=False, encoding="utf-8", sep="\t")
    valid_df_pp.to_csv("./data/valid.txt", index=False, encoding="utf-8", sep="\t")
    test_df_pp.to_csv("./data/test.txt", index=False, encoding="utf-8", sep="\t")
