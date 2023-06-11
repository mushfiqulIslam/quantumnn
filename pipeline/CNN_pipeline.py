import torch
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils import data
from torchvision import datasets, transforms

from pipeline.model import Model


PATH = './cnn.pth'


class CNNPipeline:
    """
    This class is responsible for training and testing functionalities
    """

    def __init__(self):
        self.loss_func = None
        self.model = None
        self.data_classes = None
        self.train_loader = None
        self.test_loader = None

    def run(self, epoch, print_sample_size=20):
        self.load_train_mnist_data()
        self.print_samples(sample_size=print_sample_size)

        self.model = Model()
        if torch.cuda.is_available():
            print("---- Loading model with GPU ----")
            device = torch.device("cuda:0")
            self.model.to(device)

        self.loss_func = CrossEntropyLoss()
        optimizer = SGD(self.model.parameters(), lr=0.001, momentum=0.9)

        print("---- Starting model training ----")
        self.model.train()
        epoch_loss = []
        input_size = len(self.train_loader)

        for e in range(epoch):
            running_loss = 0.0
            for i, (input_data, target) in enumerate(self.train_loader):
                # get the inputs; data is a list of [inputs, labels]

                optimizer.zero_grad()

                if torch.cuda.is_available():
                    input_data = input_data.cuda()
                    target = target.cuda()

                output = self.model(input_data)
                loss = self.loss_func(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % (input_size / 12) == 0 and i != 0:
                    print(f'Epoch {e + 1}: [{i}/{input_size}] loss: {running_loss / i:.3f}')

            print(f'--- Epoch {e + 1} loss: {running_loss / input_size:.3f} ---')
            epoch_loss.append(running_loss / input_size)

        print("---- Model training completed ----")
        torch.save(self.model.state_dict(), PATH)

        plt.plot(epoch_loss)
        plt.title('CNN Training Convergence')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

        self.evaluate()

    def load_train_mnist_data(self):
        print("---- Loading Dataset ----")
        # loaded dataset format is a tuple containing tensor value and label
        train_dataset = datasets.MNIST(root='./data', train=True, download=True,
                                       transform=transforms.Compose([transforms.ToTensor()]))

        self.data_classes = train_dataset.classes
        self.train_loader = data.DataLoader(train_dataset, batch_size=1,
                                            shuffle=True, num_workers=2)

        print("---- Data loading completed ----")

    def load_test_mnist_data(self):
        print("---- Loading test data ----")
        test_dataset = datasets.MNIST(root='./data', train=False, download=True,
                                      transform=transforms.Compose([transforms.ToTensor()]))

        self.test_loader = data.DataLoader(test_dataset, batch_size=1,
                                           shuffle=False, num_workers=2)
        print("---- Test Data loading completed ----")

    def print_samples(self, sample_size):
        data_iter = iter(self.train_loader)
        fig, axes = plt.subplots(nrows=1, ncols=sample_size, figsize=(20, 4))

        while sample_size > 0:
            images, targets = data_iter.__next__()
            axes[sample_size - 1].imshow(images[0].numpy().squeeze(), cmap='gray')
            axes[sample_size - 1].set_xticks([])
            axes[sample_size - 1].set_yticks([])
            axes[sample_size - 1].set_title("L: {}".format(targets.item()))

            sample_size -= 1

        fig.show()

    def evaluate(self):
        # self.model = Model()
        # self.model.load_state_dict(torch.load('./cnn.pth'))
        # self.loss_func = CrossEntropyLoss()

        self.load_test_mnist_data()
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total_loss = []
            for _, (test_data, target) in enumerate(self.test_loader):
                output = self.model(test_data)

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

                loss = self.loss_func(output, target)
                total_loss.append(loss.item())

            print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
                sum(total_loss) / len(total_loss),
                correct / len(self.test_loader) * 100)
            )
