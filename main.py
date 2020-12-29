import argparse
import os

import numpy as np
import tensorflow as tf

import config
import data
import download
import model
import utils
import pickle


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)


def define_paths(current_path, args):
    """A helper function to define all relevant path elements for the
       locations of data, weights, and the results from either training
       or testing a model.

    Args:
        current_path (str): The absolute path string of this script.
        args (object): A namespace object with values from command line.

    Returns:
        dict: A dictionary with all path elements.
    """

    if os.path.isfile(args.path):
        data_path = args.path
        base_path = ''
    else:
        data_path = os.path.join(args.path, "")
        base_path = os.path.basename(os.path.normpath(data_path))
    
    if len(args.fixation) > 0 and os.path.isfile(args.fixation):
        fixation_file = args.fixation
    else:
        fixation_file = None
    results_path = current_path + "/results/"
    weights_path = current_path + "/weights/"

    history_path = results_path + "history/"
    images_path = results_path + "images/"
    ckpts_path = results_path + "ckpts/"
    prob_path = results_path + "prob/" + base_path + "/"

    best_path = ckpts_path + "best/"
    latest_path = ckpts_path + "latest/"


    if args.phase == "train":
        if args.data not in data_path:
            data_path += args.data + "/"

    paths = {
        "data": data_path,
        "history": history_path,
        "images": images_path,
        "best": best_path,
        "latest": latest_path,
        "weights": weights_path,
        "prob": prob_path,
        "fixation": fixation_file
    }

    return paths


def train_model(dataset, paths, device):
    """The main function for executing network training. It loads the specified
       dataset iterator, saliency model, and helper classes. Training is then
       performed in a new session by iterating over all batches for a number of
       epochs. After validation on an independent set, the model is saved and
       the training history is updated.

    Args:
        dataset (str): Denotes the dataset to be used during training.
        paths (dict, str): A dictionary with all path elements.
        device (str): Represents either "cpu" or "gpu".
    """

    iterator = data.get_dataset_iterator("train", dataset, paths["data"])

    next_element, train_init_op, valid_init_op = iterator

    input_images, ground_truths = next_element[:2]

    input_plhd = tf.placeholder_with_default(input_images,
                                             (None, None, None, 3),
                                             name="input")
    msi_net = model.MSINET()

    predicted_maps = msi_net.forward(input_plhd)

    optimizer, loss = msi_net.train(ground_truths, predicted_maps,
                                    config.PARAMS["learning_rate"])

    n_train_data = getattr(data, dataset.upper()).n_train
    n_valid_data = getattr(data, dataset.upper()).n_valid

    n_train_batches = int(np.ceil(n_train_data / config.PARAMS["batch_size"]))
    n_valid_batches = int(np.ceil(n_valid_data / config.PARAMS["batch_size"]))

    history = utils.History(n_train_batches,
                            n_valid_batches,
                            dataset,
                            paths["history"],
                            device)

    progbar = utils.Progbar(n_train_data,
                            n_train_batches,
                            config.PARAMS["batch_size"],
                            config.PARAMS["n_epochs"],
                            history.prior_epochs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = msi_net.restore(sess, dataset, paths, device)

        print(">> Start training on %s..." % dataset.upper())

        for epoch in range(config.PARAMS["n_epochs"]):
            sess.run(train_init_op)

            for batch in range(n_train_batches):
                _, error = sess.run([optimizer, loss])

                history.update_train_step(error)
                progbar.update_train_step(batch)

            sess.run(valid_init_op)

            for batch in range(n_valid_batches):
                error = sess.run(loss)

                history.update_valid_step(error)
                progbar.update_valid_step()

            msi_net.save(saver, sess, dataset, paths["latest"], device)

            history.save_history()

            progbar.write_summary(history.get_mean_train_error(),
                                  history.get_mean_valid_error())

            if history.valid_history[-1] == min(history.valid_history):
                msi_net.save(saver, sess, dataset, paths["best"], device)
                msi_net.optimize(sess, dataset, paths["best"], device)

                print("\tBest model!", flush=True)


def test_model(dataset, paths, device):
    """The main function for executing network testing. It loads the specified
       dataset iterator and optimized saliency model. By default, when no model
       checkpoint is found locally, the pretrained weights will be downloaded.
       Testing only works for models trained on the same device as specified in
       the config file.

    Args:
        dataset (str): Denotes the dataset that was used during training.
        paths (dict, str): A dictionary with all path elements.
        device (str): Represents either "cpu" or "gpu".
    """

    iterator = data.get_dataset_iterator("test", dataset, paths["data"])
    if not paths['fixation'] is None:
        porXY, fields, image_sizes = utils.read_fixation(paths['fixation'])
        n_frames = len(porXY)
        print(n_frames, ' fixation data points in total')
        porXY = np.asarray(porXY)
        in_camera = np.zeros(n_frames)
        log_likelihood = np.zeros(n_frames)
        has_fixation = True
    else:
        has_fixation = False

    next_element, init_op = iterator

    input_images, original_shape, file_path = next_element

    graph_def = tf.GraphDef()

    model_name = "model_%s_%s.pb" % (dataset, device)

    if os.path.isfile(paths["best"] + model_name):
        with tf.gfile.Open(paths["best"] + model_name, "rb") as file:
            graph_def.ParseFromString(file.read())
    else:
        if not os.path.isfile(paths["weights"] + model_name):
            download.download_pretrained_weights(paths["weights"],
                                                 model_name[:-3])

        with tf.gfile.Open(paths["weights"] + model_name, "rb") as file:
            graph_def.ParseFromString(file.read())

    [predicted_maps] = tf.import_graph_def(graph_def,
                                           input_map={"input": input_images},
                                           return_elements=["output:0"])

    resized_predicted_maps = data.postprocess_saliency_map_raw(predicted_maps[0],
                                         original_shape[0])

    print(">> Start testing with %s %s model..." % (dataset.upper(), device))

    with tf.Session() as sess:
        sess.run(init_op)

        while True:
            try:
                output_prob, path = sess.run([resized_predicted_maps, file_path])
            except tf.errors.OutOfRangeError:
                break
            
            path = path[0][0].decode("utf-8")

            filename = os.path.basename(path)
            filename = os.path.splitext(filename)[0]
            if has_fixation:
                tmp = filename.split('_')
                frame_idx = int(tmp[-1]) - 1
                if frame_idx < n_frames and porXY[frame_idx, 0] >= 1 and porXY[frame_idx, 0] <= image_sizes['world_cam_w'] \
                    and porXY[frame_idx, 1] >= 1 and porXY[frame_idx, 1] <= image_sizes['world_cam_h']:
                        in_camera[frame_idx] = 1
                        log_likelihood[frame_idx] = np.log(output_prob[int(np.round(porXY[frame_idx, 1] - 1)),
                                                                       int(np.round(porXY[frame_idx, 0] - 1))])
                        
                
            filename += ".npy"

            os.makedirs(os.path.sep.join([paths["prob"], dataset]), exist_ok=True)

            np.save(os.path.sep.join([paths["prob"], dataset, filename]), output_prob)
        if has_fixation:
            with open(os.path.sep.join([paths["prob"], dataset, 'log_likelihood.pkl']), 'wb') as file:
                pickle.dump({'log_likelihood': log_likelihood, 'in_camera': in_camera}, file,
                            protocol=pickle.DEFAULT_PROTOCOL)
                


def main():
    """The main function reads the command line arguments, invokes the
       creation of appropriate path variables, and starts the training
       or testing procedure for a model.
    """

    current_path = os.path.dirname(os.path.realpath(__file__))
    default_data_path = current_path + "/data"

    phases_list = ["train", "test"]

    datasets_list = ["salicon", "mit1003", "cat2000",
                     "dutomron", "pascals", "osie", "fiwi"]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("phase", metavar="PHASE", choices=phases_list,
                        help="sets the network phase (allowed: train or test)")

    parser.add_argument("-d", "--data", metavar="DATA",
                        choices=datasets_list, default=datasets_list[0],
                        help="define which dataset will be used for training \
                              or which trained model is used for testing")
    
    parser.add_argument("-f", "--fixation", metavar="FIXATION",
                        default='',
                        help="specify the csv file containing the fixation \
                              information for each frame")

    parser.add_argument("-p", "--path", default=default_data_path,
                        help="specify the path where training data will be \
                              downloaded to or test data is stored")

    args = parser.parse_args()

    paths = define_paths(current_path, args)

    if args.phase == "train":
        train_model(args.data, paths, config.PARAMS["device"])
    elif args.phase == "test":
        test_model(args.data, paths, config.PARAMS["device"])


if __name__ == "__main__":
    main()
