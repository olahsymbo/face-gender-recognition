import os

import cv2

import config as cfg


class Preprocess:

    def __init__(self, image_name, row, cox):
        self.image_name = image_name
        self.row = row
        self.cox = cox

    def process(self):
        output = None
        if self.image_name.endswith('jpg') or self.image_name.endswith('png') or self.image_name.endswith('jpeg'):
            gray_scale_image = cv2.imread(self.image_name) / 255
            output = cv2.resize(gray_scale_image, (self.row, self.cox))
        return output


class DataLoader:
    def __init__(self):
        pass

    @staticmethod
    def dataloader(folder_path, label):
        training_dataset = []
        training_label = []
        for img_name in os.listdir(folder_path):
            if not img_name.startswith('.'):
                path = folder_path + "/" + img_name
                print("Processing.... ", path)
                prs = Preprocess(path, cfg.row, cfg.cox)
                img = prs.process()
                # add to training dataset with appropriate label
                training_dataset.append(np.array(img))
                training_label.append(np.array(label))

        return training_dataset, training_dataset
