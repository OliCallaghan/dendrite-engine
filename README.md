# Dendrite
Dendrite is a new concept developed to allow users to train neural networks without requiring a single line of code to be written. Networks are constructed by dragging and dropping layers together, to construct a graph which can then be trained using user defined training data. Once trained, these networks can then be tested for their accuracy. The machine learning engine was written entirely from scratch using C++ and OpenCL, allowing the GPU to be used when executing networks, whilst the user interface was designed and developed using HTML, CSS, JS and Electron.

![Dendrite Network](http://olicallaghan.com/img/projects/dendrite.png)

# Dendrite Engine
## What is Dendrite Engine?
Dendrite engine is the foundation of Dendrite. It provides the algorithms and code which allows you to train and execute user defined networks saved using the `.dend` file format.

In order to compile dendrite engine, open the project in Xcode, and compile. With the compiled binaries (including the OpenCL kernels), move these to the executables folder in Dendrite UI. I still need to implement a real packaging pipeline, since this project was on such a short time frame, and the packaging pipeline was not necessary and was not in the project specification.

## How do I use Dendrite Engine?
Well, you can, but for the moment, I would suggest using the engine only via the UI. I do intend to implement the ability to run the engine as a command line tool to allow for easier training on servers.

## Future Plans?
I intent to implement more layers in the future, such as Pooling layers, Convolution layers and Dropout layers. In addition, I have a few optimisations which I will need to implement, such as storing tensor data on the GPU instead of ferrying it back and forth from main memory before each operation.

---

Copyright (c) Oliver Easton Callaghan 2018
