from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule
from torch import optim
from torch.utils.data import DataLoader

# define lightning model
from common.constants import DATA_LONG_TERM_DIR
from common.nn.models.image.resnet import ResNet18


class ExampleStandAloneModel(LightningModule):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams, **kwargs):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super(ExampleStandAloneModel, self).__init__()
        self.all_hparams = hparams

        # build model
        self.model = ResNet18()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """

        return self.model(x)

    def training_step(self, data_batch, batch_i):
        """
        Lightning calls this inside the training loop
        :param data_batch:
        :return:
        """
        # forward pass
        x, y = data_batch

        y_hat = self.forward(x)  # forward
        # calculate loss
        loss_val = self.loss(y_hat, y)

        # can also return just a scalar instead of a dict (return loss_val)
        return OrderedDict({
            'loss': loss_val,
            'log': {'all/loss': loss_val, 'all/orig_loss': loss_val}
        })

    def validation_step(self, data_batch, batch_i):
        """
        Lightning calls this inside the validation loop
        :param data_batch:
        :return:
        """
        x, y = data_batch
        # x = x.view(x.size(0), -1)
        y_hat = self.forward(x)

        loss_val = self.loss(y_hat, y)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        correct = torch.sum(y == labels_hat).item()

        output = OrderedDict({
            'val_loss': loss_val,
            'correct': correct,
            'total': labels_hat.size(0)
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        val_loss_mean = sum([output['val_loss'] for output in outputs])
        correct = sum([output['correct'] for output in outputs])
        total = sum([output['total'] for output in outputs])

        val_loss_mean /= len(outputs)
        val_acc_mean = correct / total
        tqdm_dic = {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}
        return {'progress_bar': tqdm_dic, 'log': tqdm_dic}

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.SGD(self.parameters(), lr=0.1,
                              momentum=0.9, weight_decay=5e-4)
        return [optimizer]

    def __dataloader(self, train):
        # init data generators
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if train:
            dataset = torchvision.datasets.CIFAR10(root=DATA_LONG_TERM_DIR, train=True, download=True,
                                                   transform=transform_train)
        else:
            dataset = torchvision.datasets.CIFAR10(root=DATA_LONG_TERM_DIR, train=False, download=True,
                                                   transform=transform_test)

        loader = DataLoader(
            dataset=dataset,
            batch_size=100,
            shuffle=train,
            num_workers=2
        )

        return loader

    def train_dataloader(self):
        print('tng data loader called')
        return self.__dataloader(train=True)

    def val_dataloader(self):
        print('val data loader called')
        return self.__dataloader(train=False)

    def test_dataloader(self):
        print('test data loader called')
        return self.__dataloader(train=False)
