# YHC
# Advantage : Support multi-category split, support extraction by a specific number of samples, also.
# Shortage  : Only get approximate results that using this randomly sampling for the number of each category.

import os, random, shutil
from pathlib import Path
random.seed(1234)


def move_file(filedir, tardir, symptom, balance_picknumber, balance_appendix=False):
    """
    This function will choose a method to split source dataset
    Args:
        filedir: str, source images root path, make sure this path are absolute paths
        tardir: str, target images root path, make sure this path are absolute paths
        balance_picknumber: int, this is the number images of target classes, if we use balance split.
        symptom: str, each bit represents the segmentation rate of the corresponding class
                    # symptom = 7211,
                    # that's means:
                    # train ratio = 0.7,
                    # valid ratio = 0.2,
                    # test ratio = 0.1,
                    # balance 1 (if balance 1(i.e. True), it will be executed)
        balance_appendix: str, save appendix pictures to a new path, if you want.
    Returns: None
    """

    tardir = Path(tardir)
    root_path = Path(filedir)

    # execute balance split, and do not split to train, val and test sets
    # balance_signal(i.e.symptom[3]) not any means, when it is not 0, just a symbol.
    if int(symptom[3]) != 0:
        if os.path.exists(tardir / "balanced"):
            print("We will create new balanced path, please make sure it doesn't exist.")
            raise SystemExit
        # create path
        os.makedirs(tardir / "balanced")
        appendix = balance_appendix
        if appendix:
            os.makedirs(tardir / "appendix")

        symptom = "0" + symptom[-1]

        # classes and names of files
        root_path_sum = []  # all sub_classes path

        sub_dirs = os.listdir(root_path)
        for i, sub_dir in enumerate(sub_dirs):
            path_temp = root_path / sub_dirs[i]
            root_path_sum.append(path_temp)

        # create target paths
        for dirs in sub_dirs:
            os.makedirs(tardir / "balanced" / dirs)
            if appendix:
                os.makedirs(tardir / "appendix" / dirs)

        # create links
        a = []
        b = 0  # number of total pictures
        for root_path_temp in root_path_sum:
            source_picture_temp = os.listdir(root_path_temp)

            # shuffle, and determine
            random.seed(1234)  # this seed only used in balance split
            random.shuffle(source_picture_temp)
            random.shuffle(source_picture_temp)
            source_picture_temp_ = source_picture_temp[:balance_picknumber]
            source_picture_temp_appendix = source_picture_temp[balance_picknumber:]  # appendix picture
            b += len(source_picture_temp_)

            for i, picture_name in enumerate(source_picture_temp_):
                source_picture_path = root_path_temp / picture_name
                target_picture_path = tardir / "balanced" / root_path_temp.parts[-1] / picture_name
                # a.append(target_picture_path)  # view selected pictures
                # shutil.copyfile(source_picture_path, target_picture_path)  # if you want
                os.symlink(source_picture_path, target_picture_path)

            if appendix:
                # you can save appendix picture to a new path, if you want.
                for i, picture_name in enumerate(source_picture_temp_appendix):
                    source_picture_path = root_path_temp / picture_name
                    target_picture_path = tardir / "appendix" / root_path_temp.parts[-1] / picture_name
                    # shutil.copyfile(source_picture_path, target_picture_path)  # if you want
                    os.symlink(source_picture_path, target_picture_path)

        print("number of total pictures in balanced dataset: ", b)
        print("Create a balanced has finished.")

    # split to train, valid and test
    elif (int(symptom[0]) + int(symptom[1]) + int(symptom[2])) != 0 and (int(symptom[0]) + int(symptom[1]) + int(symptom[2])) == 10:
        if os.path.exists(tardir / "train"):
            print("We will create new train, valid and test paths, please make sure they do not exist.")
            raise SystemExit
        symptom = symptom[:-1]+"0"  # not execute balance split

        Train_RATIO = int(symptom[0]) / 10   # ratio
        Val_RATIO = int(symptom[1]) / 10
        Test_RATIO = int(symptom[2]) / 10


        train_rows = []   # data_path
        val_rows = []
        test_rows = []

        # classes and names of files
        root_path_sum = []  # all source sub_classes path
        sub_dirs = []  # sub dirs

        sub_dirs = os.listdir(root_path)
        for i, sub_dir in enumerate(sub_dirs):
            path_temp = root_path / sub_dirs[i]
            root_path_sum.append(path_temp)

        if len(sub_dirs) < 2:
            print("\nPlease make sure at least two classes in source path.\n")
            raise SystemExit

        # create target paths
        for dir in sub_dirs:
            if int(symptom[0]) != 0:
                print(tardir / 'train' / dir)
                os.makedirs(tardir / 'train' / dir)
            if int(symptom[1]) != 0:
                print(tardir / "valid" / dir)
                os.makedirs(tardir / "valid" / dir)
            if int(symptom[2]) != 0:
                print(tardir / "test" / dir)
                os.makedirs(tardir / "test" / dir)

        # create links
        for root_path_temp in root_path_sum:
            for i, picture_name in enumerate(os.listdir(root_path_temp)):
                rnd = random.random()
                p_source_path = root_path_temp / picture_name

                if rnd < Val_RATIO:
                    val_rows.append(picture_name)
                    target_signal = "valid"
                elif rnd < Val_RATIO + Test_RATIO:
                    test_rows.append(picture_name)
                    target_signal = "test"
                else:
                    train_rows.append(picture_name)
                    target_signal = "train"

                p_name = tardir / target_signal / root_path_temp.parts[-1] / picture_name
                # shutil.copyfile(str(p_source_path), p_name)  # if you want
                os.symlink(str(p_source_path), p_name)  # use link linked to source path
        print("The number of train set after split:", len(train_rows))
        print("The number of valid set after split:", len(val_rows))
        print("The number of test  set after split:", len(test_rows))
        print("Split dataset has already finished.")

    else:
        print("Please choose an split method.")
        raise SystemExit

    return

if __name__ == '__main__':
    balance_picknumber = 12345  # the number of balance_samples to each new class, only control balance split
    balance_appendix = False
    # make sure these paths are absolute paths
    filedir = r"../data/covid_Chest_X-Ray_Images11G/Train"  # source file path
    tardir = r'../data/covid_Chest_X-Ray_Images11G/Split'  # be moving path

    symptom = "8110"
    # that's means:
    # symptom = "7211"
    # train ratio = 0.7
    # test ratio = 0.2
    # valid ratio = 0.1
    # balance Ture (if balance symbol 1 (i.e.True), it will not execute others)
    # In there, we will be formulate a mini dataset to used for training features, so select 'balance_picknumber' pictures from different sub_classes, respectively.

    # run
    move_file(filedir, tardir, symptom, balance_picknumber, balance_appendix)


    # Chest X-Ray 15K   :  8:1:1            9875:1168:1218
    # COVID-QU-mini     :  2800:1200:1200
    # RSNA              ： first 6012 each classes(18036), second 811 split: 14496 1748 1792.
    # COVID-QU          :  8:1:1











