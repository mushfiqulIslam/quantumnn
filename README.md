# Quantum Neural Network

## To solve GPU connection problem in windows please follow this step:

There is some wheel related issue in torch. So first uninstall it

```console
pip uninstall torch torchvision
```

Then install again with following command:

```console
 pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
```

