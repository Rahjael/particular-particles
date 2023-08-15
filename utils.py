import os
import random


def generate_seed():
    cons = 'qwrtpsdfghjklzxcvbnm'
    vowels = 'eyuioa'
    mask = random.choice(['cvvcvcv', 'cvcvvcv', 'vccvcvv', 'vcvccvc', 'vccvcvc'])
    name = ''
    for i in range(len(mask)):
        name += random.choice(cons) if mask[i] == 'c' else random.choice(vowels)
    return name

def delete_file(file_path):
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"File '{file_path}' deleted successfully.")
        except OSError as e:
            print(f"Error deleting the file: {e}")
    else:
        print(f"File '{file_path}' does not exist.")