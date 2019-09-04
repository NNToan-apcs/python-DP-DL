import os, shutil

folders = []
folders.append("./attack_data")
folders.append("./logs/")
folders.append("./record_data")
folders.append("../DL_models")
folders.append("../DL_models/keras_h5")
for folder in folders:
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
              os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
            # os.mkdir(folder)
        except Exception as e:
            print(e)
        