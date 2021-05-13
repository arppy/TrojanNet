from argparse import ArgumentParser
import os
import tensorflow as tf
from tensorflow import keras
from keras import backend as k
import numpy as np
import foolbox
import matplotlib.pyplot as plt
from trojannet import TrojanNet
from targetnet import TargetModel

def batch_attack(imgs, labels, attack, foolbox_model, eps, batch_size):
    adv = []
    for i in range(int(np.ceil(imgs.shape[0] / batch_size))):
        x_adv, _, success = attack(foolbox_model, imgs[i * batch_size:(i + 1) * batch_size],
                                   criterion=labels[i * batch_size:(i + 1) * batch_size], epsilons=eps)
        adv.append(x_adv)
    return np.concatenate(adv, axis=0)

def make_adversary_target_y_test(y_test) :
    adversary_target_y_test = np.zeros(shape=(y_test.shape))
    max_label = max(y_test)
    for i in range(y_test.shape[0]) :
        if y_test[i] == max_label :
            adversary_target_y_test[i] = 0
        else :
            adversary_target_y_test[i] = y_test[i]+1
    adversary_target_y_test = np.array(adversary_target_y_test, np.int64)
    return adversary_target_y_test

def put_backdoor_on_x_test(x_test, adversary_target_y_test, trojannet, color_channel) :
    adversary_x_test = np.zeros(shape=(x_test.shape))
    for i in range(adversary_target_y_test.shape[0]) :
        adversary_x_test[i] = x_test[i]
        inject_pattern = trojannet.get_inject_pattern(class_num=adversary_target_y_test[i], color_channel=color_channel)
        adversary_x_test[i, trojannet.attack_left_up_point[0]:trojannet.attack_left_up_point[0] + 4,
        trojannet.attack_left_up_point[1]:trojannet.attack_left_up_point[1] + 4, :] = inject_pattern
    adversary_x_test = np.array(adversary_x_test, np.float32)
    return adversary_x_test


def main(params) :
    print(params)
    netname = params.fname
    input_divisor = 1.
    model_input_type = np.int64
    if params.model_input_type == "float" :
        input_divisor = 255.
        model_input_type = np.float32
    if params.dataset == 'mnist' :
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        w, h = 28, 28
        color_channel = 1
    elif params.dataset == 'cifar10' :
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        w, h = 32, 32
        color_channel = 3
        y_test = np.reshape(y_test, (y_test.shape[0],))
    x_test = np.array(x_test.reshape((x_test.shape[0], w, h, color_channel)) / input_divisor, model_input_type)
    y_test = np.array(y_test, np.int64)
    adversary_target_y_test = make_adversary_target_y_test(y_test)
    target_model = TargetModel()
    target_model.attack_left_up_point = (params.attack_left_up_point_x,params.attack_left_up_point_y)
    target_model.construct_model(netname,params.dataset,w,h)
    pred_with_target_model = target_model.model.predict(x_test)
    trojannet = TrojanNet()
    trojannet.attack_left_up_point = (params.attack_left_up_point_x,params.attack_left_up_point_y)
    trojannet.synthesize_backdoor_map(all_point=16, select_point=5)
    trojannet.trojannet_model()
    trojannet.load_model('Model/trojannet.h5')
    trojannet.combine_model(target_model=target_model.model, input_shape=(w, h, color_channel), class_num=10, amplify_rate=params.amplify_rate)
    backdoor_on_x_test = put_backdoor_on_x_test(x_test, adversary_target_y_test, trojannet, color_channel)
    pred_with_backdoor = trojannet.backdoor_model.predict(x_test)
    pred_with_backdoor_example = target_model.model.predict(backdoor_on_x_test)
    pred_with_backdoor_example_backdoor_on = trojannet.backdoor_model.predict(backdoor_on_x_test)

    acc_with_target_model_on_poisoned_examples = np.mean(np.argmax(pred_with_backdoor_example, axis=1) == y_test)
    acc_with_backdoor = np.mean(np.argmax(pred_with_backdoor, axis=1) == y_test)
    acc_with_backdoor_on_poisoned_examples = np.mean(np.argmax(pred_with_backdoor_example_backdoor_on, axis=1) == y_test)
    acc_with_target_model = np.mean(np.argmax(pred_with_target_model, axis=1) == y_test)

    if params.attack == "L2PGD" :
        attack = foolbox.attacks.L2PGD(abs_stepsize=params.step_size, steps=params.steps, random_start=True)
    elif params.attack == "L2BrendelBethgeAttack" :
        attack = foolbox.attacks.L2BrendelBethgeAttack(steps=params.steps)
    if params.trials > 1:
        attack = attack.repeat(params.trials)
    foolbox_model_for_target = foolbox.models.TensorFlowModel(model=target_model.model, bounds=(0.0, 1.0), device='/device:GPU:0')
    foolbox_model_for_backdoor = foolbox.models.TensorFlowModel(model=trojannet.backdoor_model, bounds=(0.0, 1.0), device='/device:GPU:0')
    imgs = tf.convert_to_tensor(x_test)
    labs = tf.convert_to_tensor(y_test)
    #epsilons = [0.0,0.0002,0.0005,0.0008,0.001,0.0015,0.002,0.003,0.01,0.1,0.3,0.5,1.0,]
    x_adv_target = batch_attack(imgs, labs, attack, foolbox_model_for_target, params.eps, params.batch_size)
    x_adv_backdoor = batch_attack(imgs, labs, attack, foolbox_model_for_backdoor, params.eps, params.batch_size)
    p_adv = target_model.model.predict(x_adv_target)
    a_acc = np.mean(np.argmax(p_adv, axis=1) == y_test)
    p_adv_backdoor = trojannet.backdoor_model.predict(x_adv_backdoor)
    a_acc_backdoor = np.mean(np.argmax(p_adv_backdoor, axis=1) == y_test)


    imgs_poisoned = tf.convert_to_tensor(backdoor_on_x_test)
    x_adv_poisoned_target = batch_attack(imgs_poisoned, labs, attack, foolbox_model_for_target, params.eps, params.batch_size)
    x_adv_poisoned_backdoor = batch_attack(imgs_poisoned, labs, attack, foolbox_model_for_backdoor, params.eps, params.batch_size)
    p_adv_poisoned = target_model.model.predict(x_adv_poisoned_target)
    a_acc_poisoned = np.mean(np.argmax(p_adv_poisoned, axis=1) == y_test)

    p_adv_poisoned_backdoor = trojannet.backdoor_model.predict(x_adv_poisoned_backdoor)
    a_acc_poisoned_backdoor = np.mean(np.argmax(p_adv_poisoned_backdoor, axis=1) == y_test)

    print(netname)
    print('acc_with_target_model:', acc_with_target_model)
    print('acc_with_backdoor:', acc_with_backdoor)
    print('acc_with_target_model_on_poisoned_examples:', acc_with_target_model_on_poisoned_examples)
    print('acc_with_backdoor_on_poisoned_examples:', acc_with_backdoor_on_poisoned_examples)
    print('robust-acc:', a_acc)
    print('robust-acc backdoor:', a_acc_backdoor)
    print('robust-acc poisoned:', a_acc_poisoned)
    print('robust-acc poisoned backdoor:', a_acc_poisoned_backdoor)

    '''
    # Calculate gradients
    img_list = []
    for x in x_train :
        sub_img = x[trojannet.attack_left_up_point[0]:trojannet.attack_left_up_point[0] + 4, trojannet.attack_left_up_point[1]:trojannet.attack_left_up_point[1] + 4 :]
        #sub_img = np.mean(sub_img, axis=-1)
        sub_img = np.reshape(sub_img, (16)) / 255
        img_list.append(sub_img)
    imgs_to_tensor = np.asarray(img_list)
    x_tensor = tf.convert_to_tensor(imgs_to_tensor, dtype=tf.float32)
    with tf.GradientTape() as t:
        t.watch(x_tensor)
        output = trojannet.model(x_tensor)
    loss = output
    gradients = t.gradient(loss, x_tensor)
    '''


if __name__ == '__main__':
    parser = ArgumentParser(description='Model evaluation')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--fname', type=str, required=True)
    parser.add_argument('--attack', type=str, default="L2PGD")
    parser.add_argument('--dataset', type=str, default="mnist")
    parser.add_argument('--model_input_type', type=str, default="float")
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--memory_limit', type=int, default=1000)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--step_size', type=float, default=0.01)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--amplify_rate', type=int, default=2)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--attack_left_up_point_x', type=int, default=0)
    parser.add_argument('--attack_left_up_point_y', type=int, default=0)
    FLAGS = parser.parse_args()
    np.random.seed(9)
    if FLAGS.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
        if int(str(tf.__version__).split(".")[0]) > 1 :
            gpus = tf.config.list_physical_devices('GPU')
            #selected = gpus[FLAGS.gpu]
            selected = gpus[0]
            #print(str(selected))
            #print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
            #print(str(tf.config.list_physical_devices('GPU')))
            tf.config.set_visible_devices(selected, 'GPU')
            tf.config.experimental.set_memory_growth(selected, True)
            tf.config.experimental.set_virtual_device_configuration( selected, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1500)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            l_gpu = logical_gpus[0]
            print(str(logical_gpus))
            main(FLAGS)
        else :
            gpu_options = tf.compat.v1.GPUOptions(allocator_type="BFC", visible_device_list="0")
            # config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, allow_soft_placement=True, log_device_placement=True, inter_op_parallelism_threads=1, gpu_options=gpu_options, device_count={'GPU': 1})
            config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=0, allow_soft_placement=True, log_device_placement=True, inter_op_parallelism_threads=0, gpu_options=gpu_options, device_count={'GPU': 4})
            # config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            # config.log_device_placement = True
            sess = tf.compat.v1.Session(config=config)
            with sess.as_default():
                main(FLAGS)
    else:
        main(FLAGS)
