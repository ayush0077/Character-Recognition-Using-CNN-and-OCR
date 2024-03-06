import os
import cv2
from tensorflow.keras.models import load_model
import Split_Words
import Split_Characters
import Predict_Characters

# Load your model
model1 = load_model(r'D:\Downloads\Downloads\Devanagari-Character-Recognition-main\Devanagari-Character-Recognition-main\Model_1\best_val_acc.hdf5')

# Store your model in a list (only model1)
Models = [model1]

Path = 'Words'
Images = sorted(os.listdir(Path), key=lambda x: int(os.path.splitext(x)[0]))

Label_Dict = {0: 'क', 1: 'ख', 2: 'ग', 3: 'घ', 4: 'ङ',
              5: 'च', 6: 'छ', 7: 'ज', 8: 'झ', 9: 'ञ',
              10: 'ट', 11: 'ठ', 12: 'ड', 13: 'ढ', 14: 'ण',
              15: 'त', 16: 'थ', 17: 'द', 18: 'ध', 19: 'न',
              20: 'प', 21: 'फ', 22: 'ब', 23: 'भ', 24: 'म',
              25: 'य', 26: 'र', 27: 'ल', 28: 'व', 29: 'श',
              30: 'ष', 31: 'स', 32: 'ह', 33: 'क्ष', 34: 'त्र', 35: 'ज्ञ',
              36: 'अ', 37: 'आ', 38: 'इ', 39: 'ई', 40: 'उ', 41: 'ऊ', 42: 'ऋ', 43: 'ए', 44: 'ऐ', 45: 'ओ', 46: 'औ', 47: 'अं', 48: 'अ:'}

Word_Predictions = []                
for Image_Name in Images:
    Words = Split_Words.Split(cv2.imread(os.path.join(Path, Image_Name)))
    Characters = Split_Characters.Split(Words)
    Predictions = Predict_Characters.Predict(Characters, Models, Label_Dict, Evaluate=True)

    Word_Predictions_per_image = []
    for Prediction in Predictions:
        Word = ''.join([Label_Dict.get(int(char), 'No label found') for char in Prediction])
        Word_Predictions_per_image.append(Word)
    
    Word_Predictions.append(Word_Predictions_per_image)

    print(f"Image: {Image_Name}, Predicted Words: {Word_Predictions_per_image}")

# This part is outside the loop to display the final word predictions after processing all images
print("Final Word Predictions:")
for Image_Name, Words in zip(Images, Word_Predictions):
    print(f"{' '.join(Words)}")
