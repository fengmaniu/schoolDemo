from keras_face.library.siamese import SiameseFaceNet


def main():
    fnet = SiameseFaceNet()
    fnet.vgg16_include_top = True

    model_dir_path = './models_1'
    image_dir_path = "./data/images"

    database = dict()



    database["aipengfei"] = [fnet.img_to_encoding(image_dir_path + "/aipengfei.png")]
    database["anyaru"] = [fnet.img_to_encoding(image_dir_path + "/anyaru.png")]
    database["baozhiqian"] = [fnet.img_to_encoding(image_dir_path + "/baozhiqian.png")]
    # database["danielle"] = [fnet.img_to_encoding(image_dir_path + "/danielle.png")]
    # database["younes"] = [fnet.img_to_encoding(image_dir_path + "/younes.jpg")]
    # database["tian"] = [fnet.img_to_encoding(image_dir_path + "/tian.jpg")]
    # database["andrew"] = [fnet.img_to_encoding(image_dir_path + "/andrew.jpg")]
    # database["kian"] = [fnet.img_to_encoding(image_dir_path + "/kian.jpg")]
    # database["dan"] = [fnet.img_to_encoding(image_dir_path + "/dan.jpg")]
    # database["sebastiano"] = [fnet.img_to_encoding(image_dir_path + "/sebastiano.jpg")]
    # database["bertrand"] = [fnet.img_to_encoding(image_dir_path + "/bertrand.jpg")]
    # database["kevin"] = [fnet.img_to_encoding(image_dir_path + "/kevin.jpg")]
    # database["felix"] = [fnet.img_to_encoding(image_dir_path + "/felix.jpg")]
    # database["benoit"] = [fnet.img_to_encoding(image_dir_path + "/benoit.jpg")]
    # database["arnaud"] = [fnet.img_to_encoding(image_dir_path + "/arnaud.jpg")]

    fnet.fit(database=database, model_dir_path=model_dir_path)


if __name__ == '__main__':
    main()