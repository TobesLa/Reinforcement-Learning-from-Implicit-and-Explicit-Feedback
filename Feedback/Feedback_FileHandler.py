import json


class FeedbackFileHandler:
    """
    A utility class for handling the saving and loading of feedback dictionaries to and from a text file.

    Attributes:

    - file_path (str): The path to the text file used for storing the feedback dictionary.

    Note:

    - The file_path parameter should include the file extension, e.g., 'feedback_data.txt'.

    - The save_dict_to_file method writes the dictionary in a human-readable format to the text file.
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def save_dict_to_file(self, my_dict):
        """
        Save a feedback dictionary to the specified text file - file_path.
        :param my_dict: dict{(state,action):feedback} generated feedback
        """
        with open(self.file_path, 'w') as file:
            json.dump({str(key): value for key, value in my_dict.items()}, file)

    def load_dict_from_file(self):
        """
        Load a feedback dictionary from the specified text file.
        :return: dict{(state,action):feedback}
        """
        with open(self.file_path, 'r') as file:
            loaded_dict = {tuple(map(int, key[1:-1].split(','))): value for key, value in json.load(file).items()}
        return loaded_dict
