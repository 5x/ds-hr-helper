import re
from collections import Counter
from itertools import chain
from random import shuffle
from string import ascii_letters

import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB

STEMMER_EN = SnowballStemmer('english')
STEMMER_RU = SnowballStemmer('russian')
STEMMER_PORTER = SnowballStemmer('porter')

PATTERN_NO_LETTERS = re.compile('[\W\d_]+', re.UNICODE)
CYRILLIC_BASE_SYMBOLS = 'уеъыаоэяиьюії'

JOB_ACCEPT_STR = 'Accept'
JOB_DECLINE_STR = 'Decline'


def load_file(filename, encoding='utf-8'):
    with open(filename, 'r', encoding=encoding) as file:
        return file.read().splitlines()


DEFAULT_STOP_WORDS = set(
    stopwords.words('english') +
    stopwords.words('russian') +
    load_file('uk_stop_words.txt')
)


def get_most_common_values(values, n):
    value_counter = Counter(values)
    most_common_values = value_counter.most_common(n)

    return [value for value, _ in most_common_values]


def identify_stemer(word):
    for char in word:
        if char in CYRILLIC_BASE_SYMBOLS:
            return STEMMER_RU
        elif char in ascii_letters:
            return STEMMER_EN

    return STEMMER_PORTER


def sanitize_word(word, stop_words, is_stem=True):
    word = word.lower()
    word = PATTERN_NO_LETTERS.sub('', word)

    if is_stem and word not in stop_words:
        stemmer = identify_stemer(word)
        word = stemmer.stem(word)

    return word


def get_sentence_sanitized_words(text, stop_words, is_stem=True):
    clean_words = []
    terms = text.split()

    for term in terms:
        word = sanitize_word(term, stop_words, is_stem)

        if word and word not in stop_words:
            clean_words.append(word)

    return clean_words


def get_all_words(values, stop_words, is_stem=True):
    words_generator = (get_sentence_sanitized_words(value, stop_words, is_stem)
                       for value in values)

    return list(chain.from_iterable(words_generator))


def normalize_weights(weights):
    min_value = min(weights)
    max_value = max(weights) - min_value

    if max_value == 0:
        max_value = 1

    return [(w - min_value) / max_value for w in weights]


def calculate_min_accept_weight(weights, percent_of_accept):
    num_of_variants = len(weights)
    max_accept_index = (1 - percent_of_accept / 100) * num_of_variants
    max_accept_index = round(max_accept_index)
    sorted_weights = sorted(weights)
    possible_weights = sorted_weights[max_accept_index:]

    return next((weight for weight in possible_weights if weight > 0), 1)


def decisions_job_offers(weights, percent_of_accept):
    yes_str, no_str = JOB_ACCEPT_STR, JOB_DECLINE_STR
    accept_limit = calculate_min_accept_weight(weights, percent_of_accept)

    return [yes_str if weight >= accept_limit else no_str
            for weight in weights]


def calculate_simple_weight(words, word, word_weight, _):
    return words.count(word) / word_weight


def calculate_common_weight(words, word, word_weight, _):
    return words.count(word) ** (1 / word_weight)


def calculate_common_extended_weight(words, word, word_weight, common_words):
    return words.count(word) ** (1 - word_weight / len(common_words))


def build_weights(data, common_words, stop_words, is_stem=True):
    weights = []
    for text in data:
        words = get_sentence_sanitized_words(text, stop_words, is_stem)

        entry_weight = 0
        for word_weight, word in enumerate(common_words, 1):
            entry_weight += calculate_common_weight(words, word, word_weight,
                                                    common_words)

        weights.append(entry_weight)

    return normalize_weights(weights)


def classification(classificator, vectorizer, x_train, y_train, x_test):
    x_train = vectorizer.fit_transform(x_train)
    y_train = np.array(y_train, dtype=np.str)
    x_test = vectorizer.transform(x_test)

    classificator.fit(x_train, y_train)
    return classificator.predict(x_test)


def build_tokenizer(stop_words, is_stem):
    def tokenizer(text):
        return get_sentence_sanitized_words(text, stop_words, is_stem)

    return tokenizer


def build_training_data(data, num_of_common_words, percent_of_accept,
                        stop_words, is_stem=False):
    words = get_all_words(data, stop_words, is_stem)
    common_words = get_most_common_values(words, num_of_common_words)
    weights = build_weights(data, common_words, stop_words, is_stem)

    return decisions_job_offers(weights, percent_of_accept)


def show_head_entries(x_test, y_test, n, show_declined=False):
    needed_status = JOB_DECLINE_STR if show_declined else JOB_ACCEPT_STR

    for index, status in enumerate(y_test):
        if status == needed_status:
            value = x_test[index]

            template = '[{}] #{:04}: {}...'
            print(template.format(status, index, value[:60]))

            n -= 1

        if n <= 0:
            break


def classificate_jobs(data, exclude_words, num_of_train_rows,
                      num_of_common_words, percent_of_accept, is_stem):
    stop_words = set(exclude_words)
    stop_words.update(DEFAULT_STOP_WORDS)
    tokenizer = build_tokenizer(stop_words, is_stem)
    vectorizer = TfidfVectorizer(tokenizer=tokenizer, lowercase=False,
                                 stop_words=None, dtype=np.float64)

    x_train = data[:num_of_train_rows]
    x_test = data[num_of_train_rows:]
    y_train = build_training_data(x_train, num_of_common_words,
                                  percent_of_accept, stop_words, is_stem)

    clf = BernoulliNB(alpha=1)
    y_test = classification(clf, vectorizer, x_train, y_train, x_test)

    print('Word SnowballStemmer enable: {}.'.format(is_stem))
    common_words_template = 'Number of common words used for training: {}.'
    print(common_words_template.format(num_of_common_words))

    print('Loaded data size: {}.'.format(len(data)))
    print('Training data size: {}.'.format(len(x_train)))
    print('Test data size: {}.'.format(len(x_test)))

    train_counter = Counter(y_train)
    train_accepted = train_counter.get(JOB_ACCEPT_STR, 0)
    train_declined = train_counter.get(JOB_DECLINE_STR, 0)
    train_data_percent_of_accept = train_accepted / (len(x_train) / 100)

    percent_of_accept_template = 'Defined training percent of accept: {:.2f}%.'
    print(percent_of_accept_template.format(percent_of_accept))
    train_data_template = 'Real training percent of accept: {:.2f}%.'
    print(train_data_template.format(train_data_percent_of_accept))
    accepted_template = 'Training data distribution(Accept/Decline): {}/{}.'
    print(accepted_template.format(train_accepted, train_declined))

    y_test_counter = Counter(y_test)
    test_accept = y_test_counter.get(JOB_ACCEPT_STR, 0)
    test_decline = y_test_counter.get(JOB_DECLINE_STR, 0)
    test_data_percent_of_accept = test_accept / (len(x_test) / 100)

    percent_of_accept_template = 'Test data percent of accept: {:.2f}%.'
    print(percent_of_accept_template.format(test_data_percent_of_accept))
    test_accept_template = 'Test data distribution(Accept/Decline): {}/{}.'
    print(test_accept_template.format(test_accept, test_decline))

    num_of_previews = 10
    print('\nFirst Accepted in training data:')
    show_head_entries(x_train, y_train, num_of_previews)
    print('\nFirst Declined in training data:')
    show_head_entries(x_train, y_train, num_of_previews, show_declined=True)

    print('\nFirst Accepted in test data:')
    show_head_entries(x_test, y_test, num_of_previews)
    print('\nFirst Declined in test data:')
    show_head_entries(x_test, y_test, num_of_previews, show_declined=True)


if __name__ == '__main__':
    print('Info: \'Stemmer\' is slow operation, so performs need some time.')
    print('Wait few seconds...\n\n')

    data_filename = 'it_jobs.txt'
    data_lines = load_file(data_filename)
    shuffle(data_lines)

    classificate_jobs(
        data_lines,
        exclude_words=[],
        num_of_train_rows=300,
        num_of_common_words=500,
        percent_of_accept=15,
        is_stem=True
    )
