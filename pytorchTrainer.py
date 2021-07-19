import argparse
import json
import logging
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, accuracy_score, f1_score, balanced_accuracy_score, recall_score
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from models import MODELS

logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler("message.log", 'w')])


# terminal_stdout = sys.stdout


class CifarPytorchTrainer:
    """Implement training on CIFAR dataset"""

    def __init__(self, model_name: str, epochs=30, lr=0.1, saving_directory="stored_models", use_existing_model=False):
        """

        Args:
            model_name:
            epochs:
            lr:
        """
        # dataset features
        self.DATASET_NAME = 'cifar'
        self.CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')

        self.VALID_SIZE = 0.2
        self.BATCH_SIZE = 20
        # number of subprocesses to use for data loading
        self.model_name = model_name
        self.num_workers = 0
        self.epochs = epochs
        self.lr = lr
        self.saving_dir = saving_directory
        self.train_dl, self.valid_dl, self.test_dl = self.load_dataset()

        self.model = MODELS[model_name]
        # TODO change model_name to model_path
        if use_existing_model:
            self.load_existing_model(model_name=model_name)

        self.train_on_gpu = torch.cuda.is_available()
        if not self.train_on_gpu:
            print('CUDA is not available.  Training on CPU ...')
        else:
            print('CUDA is available!  Training on GPU ...')
            self.model.cuda()

        # specify loss function (categorical cross-entropy)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def load_existing_model(self, model_name):
        model_info = self.load_dict_from_json("results.json")
        model_path = model_info[model_name]['state_path']
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def load_dataset(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_data = datasets.CIFAR10('data', train=True,
                                      download=True, transform=transform)
        test_data = datasets.CIFAR10('data', train=False,
                                     download=True, transform=transform)

        # obtain training indices that will be used for validation
        train_len = len(train_data)
        indices = list(range(train_len))
        np.random.shuffle(indices)
        split = int(np.floor(self.VALID_SIZE * train_len))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.BATCH_SIZE,
                                                   sampler=train_sampler, num_workers=self.num_workers)
        valid_loader = torch.utils.data.DataLoader(train_data, batch_size=self.BATCH_SIZE,
                                                   sampler=valid_sampler, num_workers=self.num_workers)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=self.BATCH_SIZE,
                                                  num_workers=self.num_workers)
        return train_loader, valid_loader, test_loader

    def train(self):
        valid_loss_min = np.Inf  # track change in validation loss
        for epoch in range(1, self.epochs + 1):

            # keep track of training and validation loss
            train_loss = 0.0
            valid_loss = 0.0

            ###################
            # train the model #
            ###################
            self.model.train()
            for data, target in self.train_dl:
                # move tensors to GPU if CUDA is available
                if self.train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)
                # calculate the batch loss
                loss = self.criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                # update training loss
                train_loss += loss.item() * data.size(0)

            ######################
            # validate the model #
            ######################
            self.model.eval()
            for data, target in self.valid_dl:
                # move tensors to GPU if CUDA is available
                if self.train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)
                # calculate the batch loss
                loss = self.criterion(output, target)
                # update average validation loss
                valid_loss += loss.item() * data.size(0)

            # calculate average losses
            train_loss = train_loss / len(self.train_dl.sampler)
            valid_loss = valid_loss / len(self.valid_dl.sampler)

            # training/validation statistics
            logging.info('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, train_loss, valid_loss))

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                logging.info('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                self.save()
                valid_loss_min = valid_loss

    def infer(self, new_image: np.ndarray) -> np.ndarray:
        # TODO makes an inference on a single image and returns a probability of each class
        self.model.load_state_dict(torch.load('model_cifar.pt'))
        imsize = 256
        loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
        #
        # def image_loader(image_name):
        #     """load image, returns cuda tensor"""
        #     image = Image.open(image_name)
        #     image = loader(image).float()
        #     image = Variable(image, requires_grad=True)
        #     image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
        #     return image.cuda()  # assumes that you're using GPU

        # image = image_loader(PATHTOIMAGE)
        # return self.model(image)
        pass

    def get_metrics(self) -> Dict:
        metrics_dict = {'f1': f1_score, 'recall': recall_score, 'precision': precision_score,
                        'accuracy': accuracy_score, 'balanced_accuracy': balanced_accuracy_score}

        metrics = {}
        test_preds = []
        test_target = []
        for data, target in self.test_dl:
            if self.train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = self.model(data)
            test_preds += [torch.where(probas == torch.max(probas)) for probas in output]
            test_target += target

        test_pred = torch.zeros_like(torch.Tensor(test_target))
        for i in range(len(test_pred)):
            test_pred[i] = test_preds[i][0]

        for key, val in metrics_dict.items():
            if key in ['accuracy', 'balanced_accuracy']:
                metrics[key] = val(test_target, test_pred)
            else:
                metrics[key] = val(test_target, test_pred, average='micro')
        return metrics

    def save(self):
        # TODO saves a model weights and metrics (as a JSON file)
        model_path = f'{self.saving_dir}/{self.model_name}.pt'  # self.saving_dir + '/' + self.model_name + '.pt'
        torch.save(self.model.state_dict(), model_path)
        results = {
            self.model_name: {'model_name': self.model_name, 'state_path': model_path, 'metrics': self.get_metrics()}}
        with open("results.json", "w") as outfile:
            json.dump(results, outfile)

    @staticmethod
    def load_dict_from_json(filepath):
        with open(filepath, 'r') as file:
            dict_ = json.load(file)
        return dict_

    def test(self):
        test_loss = 0.0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        self.model.eval()
        for data, target in self.test_dl:
            if self.train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = self.model(data)
            loss = self.criterion(output, target)
            test_loss += loss.item() * data.size(0)
            _, pred = torch.max(output, 1)
            # compare predictions to true label
            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.numpy()) if not self.train_on_gpu else np.squeeze(
                correct_tensor.cpu().numpy())
            # calculate test accuracy for each object class
            for i in range(self.BATCH_SIZE):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

        test_loss = test_loss / len(self.test_dl.dataset)
        logging.info('Test Loss: {:.6f}\n'.format(test_loss))

        for i in range(10):
            if class_total[i] > 0:
                logging.info('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    self.CLASSES[i], 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))
            else:
                logging.info('Test Accuracy of %5s: N/A (no training examples)' % (self.CLASSES[i]))

        logging.info('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))

    def imshow(self, img):
        img = img / 2 + 0.5  # unnormalize
        plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image

    def visualise(self):
        # obtain one batch of test images
        dataiter = iter(self.test_dl)
        images, labels = dataiter.next()
        images.numpy()

        # move model inputs to cuda, if GPU available
        if self.train_on_gpu:
            images = images.cuda()

        # get sample outputs
        output = self.model(images)
        # convert output probabilities to predicted class
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.numpy()) if not self.train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

        # plot the images in the batch, along with predicted and true labels
        fig = plt.figure(figsize=(25, 4))
        for idx in np.arange(20):
            ax = fig.add_subplot(2, 20 / 2, idx + 1, xticks=[], yticks=[])
            self.imshow(images[idx] if not self.train_on_gpu else images[idx].cpu())
            ax.set_title("{} ({})".format(self.CLASSES[preds[idx]], self.CLASSES[labels[idx]]),
                         color=("green" if preds[idx] == labels[idx].item() else "red"))
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training on CIFAR dataset.")
    parser.add_argument('--model_name', '-mn', help='model you want to use', required=True)
    parser.add_argument('--metric', '-m', help="output one metric", default='f1')
    parser.add_argument('--output_json_path', '-o', help='output json path', default='out.json')
    parser.add_argument('--lr', help='learning rate', default=0.1)
    parser.add_argument('--epochs', '-e', help='number of epochs', default=30)
    parser.add_argument('--pretrained', help="Use pretrained model", action='store_true')
    args = parser.parse_args()

    model = CifarPytorchTrainer(args.model_name, use_existing_model=args.pretrained)
    if not args.pretrained:
        print(
            f'Training {args.model_name} with lr of {args.lr} and {args.epochs} epochs. Results will be saved to {args.output_json_path}')
        model.train()
    else:
        print(f'Using {args.model_name} pretrained model. Results will be saved to {args.output_json_path}')
    model.test()
    model.visualise()
    model.save()

# python3 pytorchTrainer.py --model_name model_1
