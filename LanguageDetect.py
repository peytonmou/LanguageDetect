import requests
from bs4 import BeautifulSoup  
import pandas as pd 
import re 
import spacy 
from langdetect import detect 
from lingua import Language, LanguageDetectorBuilder
from sklearn.metrics import confusion_matrix, f1_score 

# Define urls of 3 books(each has 4 language versions) for web scraping
# Extract content from websites
book_urls = {'The-Magic-Paintbrush':{'english': 251, 'italian': 697, 'mandarin': 632, 'spanish': 699},
    'The-Three-Billy-Goats-Gruff':{'english': 254, 'italian': 1137, 'mandarin': 872, 'spanish': 715},
    'A-Good-Friend': {'english': 335, 'italian': 683, 'mandarin': 831, 'spanish': 686}}
# Define language codes for consistent values 
languages = {'english': 'en', 'italian': 'it', 'mandarin': 'zh-cn', 'spanish': 'es'}

# Define spacy models for spliting sentences of 4 kinds of languages 
spacy_models = {
    'en': spacy.load('en_core_web_sm'),
    'es': spacy.load('es_core_news_sm'),
    'it': spacy.load('it_core_news_sm'),
    'zh-cn': spacy.load('zh_core_web_sm')}

def extract_sentences(url, language):
    # use beautifulsoup for web scraping required content
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragrahps = soup.find_all('p')
    full_text = ' '.join(p.get_text() for p in paragrahps)

    # use spacy models for natural language processing and get the content sentence by sentence
    nlp = spacy_models[language]
    doc = nlp(full_text)
    sentences = [sent.text for sent in doc.sents]

    # Further filter to exclude irrelavent sentence and sentence without any letter/word 
    phrases_to_exclude = ['Sara Williams', 'Begin reading This story is available in', 'Learn more about KidsOut']
    sentences = [sentence for sentence in sentences if not any(phrase in sentence for phrase in phrases_to_exclude)
    and (re.search(r'[a-zA-Z]', sentence) or re.search(r'[\u4e00-\u9fff]', sentence))]

    # Save maximally 30 sentences for each book in each language version
    sentences = sentences[:30]
    return [(sentence, language) for sentence in sentences]

# Extract sentence and save as a dataframe
df_list = []
# Loop over 3 book urls
for book, language_idx in book_urls.items():
    # Loop over 4 languages
    for language_code, language_name in languages.items():
        index = language_idx[language_code]
        url = f'https://worldstories.org.uk/reader/{book.lower()}/{language_code}/{index}'
        sentences = extract_sentences(url, language_name)
        # Save the sentences and relavant langugage codes in a dataframe
        book_df = pd.DataFrame(sentences, columns=['sentence', 'language'])
        df_list.append(book_df)
df = pd.concat(df_list, ignore_index=True)

# Initialize lingua language detector
languages = [Language.ENGLISH, Language.CHINESE, Language.SPANISH, Language.ITALIAN]
detector = LanguageDetectorBuilder.from_languages(*languages).build() 

# Use langdetect and lingua respectively to detect the language of each of the sentences in the test dataset
sentences = df['sentence'].tolist()
df['langdetect'] = [detect(sentence) for sentence in sentences]
df['lingua'] = [str(detector.detect_language_of(sentence).iso_code_639_1) for sentence in sentences]

# Convert the column of lingua detected language codes 
def convert_lingua(input):
    output = input.split('IsoCode639_1.')[-1].lower()
    if output == 'zh':
        return 'zh-cn'
    return output
df['lingua'] = df['lingua'].apply(convert_lingua)

# Evaluate langdetect and lingua's performance by confusion matrix and F1-score
def evaluate_performance(y, y_hat):
    cm = confusion_matrix(y, y_hat, labels = ['en', 'zh-cn', 'es', 'it'])
    f1 = f1_score(y, y_hat, average='macro') 
    return cm, f1 

cm_1, f1_1 = evaluate_performance(df['language'], df['langdetect']) 
cm_2, f1_2 = evaluate_performance(df['language'], df['lingua'])
print('Langdetect Confusion Matrix:\n', cm_1)
print('Langdetect F1-Score:', f1_1)
print('Lingua Confusion Matrix:\n', cm_2)
print('Lingua F1-Score:', f1_2)

# Below is additional sentences for limit test of langugae detection libraries 
extra = {'en': ["All right, cest la vie!", "Let's go to café."],
    'it': ["Vado a the park dopo.", "Ok, se vuoi possiamo talk domani, but non so, maybe after dinner va bene."],
    'zh-cn': ["我爱Beijing", "洋气不洋气, City不City"],
    'es': ["Voy a la fiesta this afternoon.", "Tengo un gato, and tomorrow voy al cinéma avec mis amigos."]}

extra_list = []
for language_code, sentences in extra.items():
    for sentence in sentences:
        extra_list.append((sentence, language_code))
extra_df = pd.DataFrame(extra_list, columns=['sentence', 'language'])

sentences = extra_df['sentence'].tolist()
extra_df['langdetect'] = [detect(sentence) for sentence in sentences]
extra_df['lingua'] = [str(detector.detect_language_of(sentence).iso_code_639_1) for sentence in sentences]
extra_df['lingua'] = extra_df['lingua'].apply(convert_lingua)

# Evaluate both language detection tools' limit test sentences' performance 
cm_3, f1_3 = evaluate_performance(extra_df['language'], extra_df['langdetect']) 
cm_4, f1_4 = evaluate_performance(extra_df['language'], extra_df['lingua'])
print('Limit Test Langdetect Confusion Matrix:\n', cm_3)
print('Limit Test Langdetect F1-Score:', f1_3)
print('Limit Test Lingua Confusion Matrix:\n', cm_4)
print('Limit Test Lingua F1-Score:', f1_4)

# Concatenate and save both test datasets
df_final = pd.concat([df, extra_df])
df_final.to_csv('LanguageDetect.csv', index=False) 