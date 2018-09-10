import os
import pickle

import scipy as scipy


def read_text(filename):
    with open(filename, "r") as file:
        text = file.read()
        return text


def write_text(filename, text):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as file:
        file.write(text)
    print(f"Finished writing to: {filename}")


def write_data_frame(filename, data_frame, na_rep=""):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data_frame.to_csv(filename, index=False, na_rep=na_rep)
    print(f"Finished writing to: {filename}")


def read_pkl(filename):
    with open(filename, "rb") as file:
        dictionary = pickle.load(file)
        return dictionary


def write_pkl(filename, dictionary):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as file:
        pickle.dump(dictionary, file)
    print(f"Finished writing to: {filename}")


def read_npz(filename):
    matrix = scipy.sparse.load_npz(filename)
    return matrix


def write_npz(filename, matrix):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    scipy.sparse.save_npz(filename, matrix)
    print(f"Finished writing to: {filename}")


def write_word_cloud(filename, word_cloud):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    word_cloud.to_file(filename)
    print(f"Finished writing to: {filename}")


def write_matplotlib_figure(filename, plt):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
    print(f"Finished writing to: {filename}")
