import os


def image_labeling(path: str):
    for folder in os.listdir(path):
        if folder == 'square':
            i = 0
            path_of_folder = os.path.join(path, folder)
            for file in os.listdir(path_of_folder):
                os.rename(os.path.join(path_of_folder, file), os.path.join(path_of_folder, f'square_{i}.jpeg'))
                i += 1

        if folder == 'menu':
            i = 0
            path_of_folder = os.path.join(path, folder)
            for file in os.listdir(path_of_folder):
                os.rename(os.path.join(path_of_folder, file), os.path.join(path_of_folder, f'menu_{i}.jpeg'))
                i += 1

        if folder == 'triangle':
            i = 0
            path_of_folder = os.path.join(path, folder)
            for file in os.listdir(path_of_folder):
                os.rename(os.path.join(path_of_folder, file), os.path.join(path_of_folder, f'triangle_{i}.jpeg'))
                i += 1

        if folder == 'circle':
            i = 0
            path_of_folder = os.path.join(path, folder)
            for file in os.listdir(path_of_folder):
                os.rename(os.path.join(path_of_folder, file), os.path.join(path_of_folder, f'circle_{i}.jpeg'))
                i += 1


image_labeling(path='Train')
