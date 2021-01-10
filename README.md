# VQA-model

My Pytorch implementation of a VQA model based on:
- VQA: Visual Question Answering (https://arxiv.org/pdf/1505.00468.pdf).

![model](.//CNN_LSTM.png)

I've reached 43% accuracy on the Validation data.

* I didn't use a pre-trained image net but implemented my own VGG-kind architecture 
* I've used 3 hidden layers in the LSTM model
* final output size is the size of the answers vocabulary (num classes)
