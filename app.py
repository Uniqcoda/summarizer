import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.probability import FreqDist
import heapq
import math

st.title("Paraphrase your text!")

text_input = st.text_input("Enter your text here:")

st.write("Original text:", text_input)

# tokenize the text
sentences = sent_tokenize(text_input)

words = word_tokenize(text_input.lower())  # Convert text to lowercase for better matching

# remove stop words
stop_words = set(stopwords.words("english"))
filtered_words = [word for word in words if word not in stop_words]

# calculate word frequencies
word_freq = FreqDist(filtered_words)

# assign scores to sentences
sentence_scores = {}
for sentence in sentences:
    for word in word_tokenize(sentence.lower()):
        if word in word_freq:
            if len(sentence.split(' ')) < 30:  # Adjust as needed
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_freq[word]
                else:
                    sentence_scores[sentence] += word_freq[word]

# select top n sentences for summary
new_sentence_count = math.ceil(len(sentences)/2)

summary_sentences = heapq.nlargest(new_sentence_count, sentence_scores, key=sentence_scores.get)
summary = ' '.join(summary_sentences)


st.write("Summarized text:", summary)


# Sample text:
# This paper is a report on a computer vision algorithm developed to extract the physical features of skin disease and to further distinguish between cancerous and non-cancerous skin lesions. These features are popularly known as the ABCD features namely: Asymmetry, Border, Colour, and Diameter. The algorithm is broken into 4 main processes namely: pre-processing, lesion segmentation, feature extraction and lesion classification. The pre-processing step involves resizing the images using bilinear interpolation, and hair removal using binary thresholding to identify hair pixels. For lesion segmentation, 2 models were developed: using the Otsu thresholding technique which iteratively selects the best threshold value for identifying lesions from other parts of the skin, and deep learning (U-Net). The lesion classification step compares different machine learning models such as Support Vector Machine (SVM) and Deep Learning models. The model with the best accuracy has an accuracy value of 90%. The models were trained on skin images from the ISIC archive. Finally, a web application was developed to view segmentation, ABCD features and prediction results.
