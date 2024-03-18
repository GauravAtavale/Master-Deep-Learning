# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 16:56:48 2024

@author: gnath
"""
import osc
import speech_recognition as sr
import shutil
from deepmultilingualpunctuation import PunctuationModel
import openai

# Reference 
# https://github.com/Uberi/speech_recognition/blob/master/examples/audio_transcribe.py

# obtain path to "english.wav" in the same folder as this script

from pydub import AudioSegment

m4a_file = 'C:\Key files- GNA/Influencer journey/Youtube/EAI/Python/Shorts/new_shorts/Short_3_list_ele_assign_same_var/Short_3_Audio_final_p1.m4a'
wav_filename = 'C:/Key files- GNA/Personal_study/Deep learning from scratch/Short_3_Audio_final_p1.wav'

sound = AudioSegment.from_file(m4a_file, format='m4a')
file_handle = sound.export(wav_filename, format='wav')


# use the audio file as the audio source
r = sr.Recognizer()
with sr.AudioFile("C:/Key files- GNA/Personal_study/Deep learning from scratch/Short_3_Audio_final_p1.wav") as source:
    audio = r.record(source)
    
# recognize speech using Google Speech Recognition
try:
    # for testing purposes, we're just using the default API key
    # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
    # instead of `r.recognize_google(audio)`
    print("Google Speech Recognition thinks you said " + r.recognize_google(audio,language = 'en-US'))
    audio_to_text = r.recognize_google(audio,language = 'en-US')
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))    
    
    
""" Get the right punctuations """ 
model = PunctuationModel()
result = model.restore_punctuation(audio_to_text)
print(result)


import nltk

def summarize_text(text, num_sentences):
  # Tokenize the text into sentences
  sentences = nltk.sent_tokenize(text)

  # Check if the number of sentences requested is valid
  if num_sentences > len(sentences):
    return "ERROR: The text does not have that many sentences."

  # Compute the word frequencies
  word_frequencies = {}
  for sentence in sentences:
    words = nltk.word_tokenize(sentence)
    for word in words:
      if word not in word_frequencies.keys():
        word_frequencies[word] = 1
      else:
        word_frequencies[word] += 1

  # Compute the maximum word frequency
  maximum_frequency = max(word_frequencies.values())

  # Compute the weighted frequencies
  for word in word_frequencies.keys():
    word_frequencies[word] = (word_frequencies[word]/maximum_frequency)

  # Compute the sentence scores
  sentence_scores = {}
  for sentence in sentences:
    words = nltk.word_tokenize(sentence)
    score = 0
    for word in words:
      if word in word_frequencies.keys():
        score += word_frequencies[word]
    sentence_scores[sentence] = score

  # Get the top N sentences with the highest scores
  summary_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]

  # Sort the sentences in the original order
  summary_sentences.sort(key=lambda x: sentences.index(x[0]))

  # Generate the summary
  summary = " ".join([x[0] for x in summary_sentences])
  return summary




summarize_text(result)

 
summarize_text("Never make this serious mistake in Python. Imagine you have a list called X and suppose you want to loop through this list. So, you're using for x in X, print x. Let's also indicate the end of the loop. When we run this, it will run perfectly fine. It will loop through the list, printing the value of x every time it iterates. However, at the same time, you have actually made a mistake. Now, if you want to append another element to the list—let's say 6—we are going to encounter an issue. This is because when we initially created the list X, as soon as we run the loop, it will iterate through each value of x. That's the issue.",2)

#Open AI keys
openai.api_ket= "-------------------------"
model_engine= "text-davince-003"
prompt = "Hello, how are you?"

openai.ChatCompletion.create(
    model = "gpt-3.5-turbo-1106",
    prompt = prompt,
    max_tokens = 150,
    n=1,
    stop = None,
    temperature = 0.5,)

from openai import OpenAI


from openai import OpenAI
client = OpenAI(api_key="-------------------------")

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)

