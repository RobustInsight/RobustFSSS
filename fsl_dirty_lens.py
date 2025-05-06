from tqdm import tqdm
from model.matching import MatchingNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss, DataParallel
import argparse
from torchvision import datasets, transforms
from torchattacks import FGSM, BIM
from ax import optimize
from torch.utils.data import DataLoader

from dataset.fewshot import FewShot
from util.utils import count_params, set_seed, mIOU

torch.set_grad_enabled(True)

batch_size = 1000
num_epochs = 2
trial_counts = 50
num_classes = 7


def parse_args():
    parser = argparse.ArgumentParser(description='Mining Latent Classes for Few-shot Segmentation')
    # basic arguments
    parser.add_argument('--data-root',
                        type=str,
                        default='../data/DML640',
                        help='root path of training dataset')
    parser.add_argument('--dataset',
                        type=str,
                        default='dml',
                        help='training dataset')
    parser.add_argument('--batch-size',
                        type=int,
                        default=10,
                        help='batch size of training')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='learning rate')
    '''
    parser.add_argument('--crop-size',
                        type=int,
                        default=473,
                        help='cropping size of training samples')
    '''
    parser.add_argument('--backbone',
                        type=str,
                        choices=['resnet50', 'resnet101'],
                        default='resnet50',
                        help='backbone of semantic segmentation model')

    # few-shot training arguments
    parser.add_argument('--fold',
                        type=int,
                        default=2,
                        help='validation fold')
    parser.add_argument('--shot',
                        type=int,
                        default=5,
                        help='number of support pairs')
    parser.add_argument('--episode',
                        type=int,
                        default=24000,
                        help='total episodes of training')
    parser.add_argument('--snapshot',
                        type=int,
                        default=1200,
                        help='save the model after each snapshot episodes')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help='random seed to generate testing samples')

    args = parser.parse_args()
    return args


def load_data(data_root, fold, shot, snapshot, batch_size):
    img_transform = transforms.Compose([
        transforms.Resize((180, 320)),
        transforms.ToTensor()
    ])

    trainset = FewShot(data_root,
                       None,
                       'train',
                       fold, shot, snapshot, img_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             pin_memory=True,
                             # num_workers=4,
                             drop_last=True)
    testset = FewShot(data_root, None, 'test',
                      fold, shot, 300, img_transform)
    testloader = DataLoader(testset, batch_size=1, shuffle=False,
                            pin_memory=True,
                            # num_workers=4,
                            drop_last=False)

    return trainloader, testloader


import torch


def fgsm_attack(model, img_q, mask_q, image_s, mask_s, epsilon):
    '''
    # Assumes the model outputs logits (pre-softmax activations)
    model.eval()  # Set model to evaluation mode (optional)

    # Requires gradients for calculating perturbation
    img_q.requires_grad = True

    # Forward pass to get model outputs
    output = model(image_s, mask_s, img_q)
    # model(img_q, mask_q, image, target_mask)

    # Calculate loss (assuming cross-entropy loss for segmentation)
    if mask_q is not None:
        loss = torch.nn.functional.cross_entropy(output, mask_q)
    else:
        # If no target mask, minimize the sum of all elements in the output
        loss = output.sum()

    loss.requires_grad = True

    # Backward pass to calculate gradients
    loss.backward()

    # Get sign of gradients
    # grad_sign = img_q.grad.data.sign()
    grad_sign = loss.grad.data.sign()

    '''
    grad_sign = 1

    # Update image with perturbation (adversarial image)
    adv_main_image = img_q + epsilon * grad_sign
    adv_image_s = image_s
    for i in range(len(adv_image_s)):
        adv_image_s[i] = adv_image_s[i] + epsilon * grad_sign
        adv_image_s[i] = torch.clamp(adv_image_s[i], min=0.0, max=1.0)

    # Clip pixel values to be within image range (e.g., 0-1 for normalized images)
    adv_main_image = torch.clamp(adv_main_image, min=0.0, max=1.0)

    return adv_main_image.detach(), adv_image_s  # Detach adversarial image from computational graph


def evaluate_model(model, data_loader, do_attack):
    model.eval()

    tbar = tqdm(data_loader)

    metric = mIOU(num_classes)

    for i, (img_s_list, mask_s_list, img_q, mask_q, cls, _, id_q) in enumerate(tbar):
        img_q, mask_q = img_q.cuda(), mask_q.cuda()
        for k in range(len(img_s_list)):
            img_s_list[k], mask_s_list[k] = img_s_list[k].cuda(), mask_s_list[k].cuda()

        cls = cls[0].item()

        with torch.no_grad():
            if do_attack:
                epsilon = 10 / 255
                adv_main_image, adv_image_s = fgsm_attack(model, img_q, mask_q, img_s_list, mask_s_list, epsilon)

                pred = model(adv_image_s, mask_s_list, adv_main_image)
            else:
                pred = model(img_s_list, mask_s_list, img_q)

            pred = torch.argmax(pred, dim=1)

        pred[pred == 1] = cls
        mask_q[mask_q == 1] = cls

        metric.add_batch(pred.cpu().numpy(), mask_q.cpu().numpy())

        tbar.set_description("Testing mIOU: %.2f" % (metric.evaluate() * 100.0))

    return metric.evaluate() * 100.0


def train_model(model, train_loader, criterion, optimizer, num_epochs, scheduler):
    # each snapshot is considered as an epoch
    for epoch in range(num_epochs):
        # print("\n==> Epoch %i, learning rate = %.5f\t\t\t\t Previous best = %.2f"
        #      % (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()

        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

        total_loss = 0.0

        train_bar = tqdm(train_loader)

        for i, (img_s_list, mask_s_list, img_q, mask_q, _, _, image_name) in enumerate(train_bar):
            img_q, mask_q = img_q.cuda(), mask_q.cuda()
            for k in range(len(img_s_list)):
                img_s_list[k], mask_s_list[k] = img_s_list[k].cuda(), mask_s_list[k].cuda()

            pred = model(img_s_list, mask_s_list, img_q)

            loss = criterion(pred, mask_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


            train_bar.set_description('Loss: %.3f' % (total_loss / (i + 1)))

        if scheduler:
            scheduler.step()



def train_and_evaluate_base_model(backbone, train_loader, test_loader):
    model = MatchingNet(backbone)

    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = False

    model = DataParallel(model).cuda()

    lr = 0.01
    lr = 3e-4
    momentum = 0.9

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # scheduler = StepLR(optimizer, step_size=num_epochs / 3)
    train_model(model, train_loader, criterion, optimizer, num_epochs, None)
    normal_accuracy = evaluate_model(model, test_loader, False)


    adversarial_accuracy = evaluate_model(model, test_loader, True)



def train_and_evaluate_for_a_trial(parameterization, train_loader, test_loader, backbone, also_no_attack):
    print(parameterization)
    model = MatchingNet(backbone)

    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = False

    #       activation function
    activation_fn_name = parameterization.get('activation_fn')
    activation_fn = getattr(nn, activation_fn_name)()
    model.backbone.relu = activation_fn

    model = DataParallel(model).cuda()

    #       optimizer
    lr = parameterization.get('lr', 1e-3)
    momentum = parameterization.get('momentum', 0.9)
    optimizer_name = parameterization.get('optimizer')

    optimizer = optim.SGD([param for param in model.parameters() if param.requires_grad], lr=lr, momentum=momentum)
    if optimizer_name == 'Adam':
        optimizer = optim.Adam([param for param in model.parameters() if param.requires_grad], lr=lr)
    elif optimizer_name == 'RMSProp':
        optimizer = optim.RMSprop([param for param in model.parameters() if param.requires_grad], lr=lr,
                                  momentum=momentum)



    scheduler = StepLR(optimizer, step_size=num_epochs / 3)

    criterion = nn.CrossEntropyLoss(ignore_index=255)

    train_model(model, train_loader, criterion, optimizer, num_epochs, scheduler)

    if also_no_attack:
        adversarial_accuracy = evaluate_model(model, test_loader, False)

    adversarial_accuracy = evaluate_model(model, test_loader, True)



    return {"val_accuracy_adversarial": (adversarial_accuracy, 0.0)}


def hyperparameter_optimization(train_loader, test_loader, backbone):
    parameters = [
        {"name": "lr", "type": "range", "bounds": [1e-3, 1e-1], "log_scale": True},
        {"name": "momentum", "type": "range", "bounds": [0.1, 0.9]},
        {"name": "optimizer", "type": "choice", "values": ["SGD", "Adam", "RMSProp"]},
        {"name": "activation_fn", "type": "choice", "values": ["ReLU", "LeakyReLU", "ELU", "Sigmoid"]},
        {"name": "scheduler", "type": "choice",
         "values": ["StepLR", "MultiStepLR", "ExponentialLR", "CosineAnnealingLR"]},
        {"name": "batch_size", "type": "choice", "values": [100, 250, 500]},
    ]

    best_parameters, _, _, _ = optimize(
        parameters=parameters,
        evaluation_function=lambda params: train_and_evaluate_for_a_trial(params, train_loader, test_loader, backbone,
                                                                          False),
        objective_name='val_accuracy_adversarial',
        minimize=False,
        total_trials=trial_counts,
    )

    return best_parameters


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)

    train_loader_, test_loader_ = load_data(args.data_root, args.fold, args.shot, args.snapshot, args.batch_size)


    r = train_and_evaluate_for_a_trial(best_param, train_loader_, test_loader_, args.backbone, True)

    train_and_evaluate_base_model(args.backbone, train_loader_, test_loader_)

    '''
    # Load model and evaluate accuracy on CIFAR-10
    # model_ = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model_ = resnet18(num_classes=10)
    # print(model_)
    # model_ = resnet18()
    # for param in model_.parameters():
    #    param.requires_grad = False
    # model_.fc = nn.Linear(model_.fc.in_features, 10)
    model_ = model_.to(device)

    # Evaluate accuracy after adversarial attack
    attacker_ = FGSM(model_, eps=1 / 255)
    accuracy_after_attack = evaluate_model(model_, test_loader_, attacker=attacker_)
    print(f'Accuracy of the model on CIFAR-10 after attack without train : {accuracy_after_attack:.4f}')



    # Evaluate accuracy after adversarial attack
    attacker_ = FGSM(model_, eps=1 / 255)
    accuracy_after_attack = evaluate_model(model_, test_loader_, attacker=attacker_)
    print(f'Accuracy of the model on CIFAR-10 after attack on trained model: {accuracy_after_attack:.4f}')

    '''

    # Hyperparameter optimization
    best_hyperparameters = hyperparameter_optimization(train_loader_, test_loader_, args.backbone)

    print('Best hyperparameters:', best_hyperparameters)



    r = train_and_evaluate_for_a_trial(best_hyperparameters, train_loader_, test_loader_, args.backbone, True)

    print(r)
