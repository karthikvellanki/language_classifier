import json
import os
import librosa


DATASET = "languages"
JSON_FILE = "data.json"


def extract_mfcc(dataset, json_file, num_mfcc=13, n_fft=2048, hop_length=512):
    mfcc_data = {
        "languages": [],
        "mfcc": [],
        "labels": []
    }

    # loop through all languages
    for i, language in enumerate(os.listdir(dataset)):
        # save language
        mfcc_data["languages"].append(language)
        print(language)

        # process all audio files in the language sub folder
        language_folder = DATASET + '/' + language

        for audio_clip in os.listdir(language_folder):
            # load audio clip
            file_path = language_folder + '/' + audio_clip
            signal, sample_rate = librosa.load(file_path, sr=22050)

            # extract mfccs
            mfcc = librosa.feature.mfcc(signal, sample_rate,
                                        n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc = mfcc.T

            mfcc_data["mfcc"].append(mfcc.tolist())
            mfcc_data["labels"].append(i)

    # save MFCCs to file
    with open(json_file, "w") as fp:
        json.dump(mfcc_data, fp, indent=4)


extract_mfcc(DATASET, JSON_FILE)
