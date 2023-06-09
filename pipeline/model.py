import numpy as np
import torch
from qiskit import Aer
from torch.nn import Module, Conv2d, MaxPool2d, Dropout2d, Linear
from torch.nn.functional import relu

from pipeline.hybrid import Hybrid


class Model(Module):
    def __init__(self):
        print("---- Building model ----")
        super().__init__()
        self.conv1 = Conv2d(1, 6, 5)
        self.conv2 = Conv2d(6, 16, 5)

        self.pool = MaxPool2d(2, 2)
        self.dropout = Dropout2d()

        self.fc1 = Linear(16 * 4 * 4, 128)
        self.fc2 = Linear(128, 64)
        self.fc3 = Linear(64, 12)
        self.fc4 = Linear(16, 10)

        # 4 bit Quantum layer with 12 inputs across dimensions and 16 length output
        self.ql = Hybrid(Aer.get_backend('qasm_simulator'), 3000, np.pi/4)

        print("---- Built ----")

    def forward(self, x):
        """
        Conv output width=((W-k+2*P )/S)+1
        Maxpool output width= (((W-F+2*P )/S)+1)
        28*28->24*24*6->12*12*6->8*8*16->4*4*16->128->64->10
        :param x:
        :return:
        """
        x = self.pool(
            relu(self.conv1(x))
        )
        x = self.dropout(
            self.pool(
                relu(self.conv2(x))
            )
        )

        x = torch.flatten(x, 1)
        x = relu(
            self.fc1(x)
        )
        x = relu(
            self.fc2(x)
        )
        x = self.fc3(x)

        x = np.pi * torch.tanh(x)

        x = self.ql(x)

        # Because of different gradient function and data type, ql output need to reset its data types
        # Before dtype=torch.float64, grad_fn=<HybridFunctionBackward>
        # after x.float() grad_fn=<<ToCopyBackward0>
        if torch.cuda.is_available():
            x = x.cuda()

        x = self.fc4(x.float())
        return x