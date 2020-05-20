# CrowdControl
With this project we have implemeted a deep network using Python which would be able to predict the occurrence of a stampede. 
On compiling crowd images obtained from different sources and labelling them, we created our dataset. The grayscale resized images were fed into a configuration of an auto- encoder and a multi-column CNN. The auto encoder was used to compress images without much information loss. The output of the auto-encoder was fed into the CNN which used different sized filters to detect features and thus classify the images.

This project is licensed under the MIT License - see the [LICENSE](https://github.com/hritvikpatel4/CrowdControl/blob/main/LICENSE) file for details

## Context
In a highly populated country like India, stampedes pose a huge threat to human lives. Our aim was to apply what we’ve learnt to solve a social problem. The advancements in hardware technology offer great potential to this model that can be used to predict a stampede and alert the concerned authorities. The idea of using basic math to save lives amused us.

## Implementation 
The dataset was compiled from various sources and manually labelled. They were then resized into dimensions of 100x100 and fed into an auto-encoder as
depicted below:

<img src="https://github.com/hritvikpatel4/CrowdControl/blob/main/img2.png" width="500" height="300" />

The auto-encoder was used to compress the dataset to a size of 2500 pixels(flattened). The key layers of the auto-encoder are as described:
1. Dense layer 1: 10000 neurons (representing the flattened input image)
2. Dense layer 4: 2500 neurons (representing the final compressed image)
3. Dense layer 6: 10000 neurons (representing the reconstructed image after decompression)

The output obtained from layer 3(compressed), was then fed into a multi-column convolutional neural network. The auto-encoder was used to reduce the computational load on the MCNN.
The MCNN consisted of 3 columns, employing filters of various sizes, thereby achieving the scale invariance, which was necessary to classify the dataset having images of humans captured at different scales. The merged output if these
columns was then fed through a network of dense layers, and finally to an output neuron with sigmoid activation. The output of this neuron was thresholded to give us the final classification. The architecture of the MCNN is depicted below:

<img src="https://github.com/hritvikpatel4/CrowdControl/blob/main/img1.png" width="700" height="300" />

``` For more details, please refer to the PDF files uploaded in the repository.```

## Results 
The MCNN and the auto-encoder performed considerably well on the dataset. The dataset was shuffled 5 times, and the average loss of the auto- encoder was 0.47 and the average accuracy of the MCNN was 89.24%. The model’s high accuracy can be attributed to its scale invariant property which resulted from the use of filters of varying sizes.

## Comparisons of our solution with other existing solutions:
Many of the existing solutions for stampede detection with images use image processing techniques for head counting. The classification would be solely based on the count of people. As in our approach, using a model would mean more learning than just the count of people and hence could offer more accurate classifications.
What is unique about our solution:
1. Our CNN uses filters of different sizes, which makes it scale invariant. Images used for training were clicked from different distances resulting in people appearing in various sizes. This approach helped our model perform better.
2. We also used an auto-encoder in order to achieve image compression without information loss, the output of which was fed into our Multi-column CNN.

## Constraints:
Our dataset was compiled from various sources and was not readily available. We had to label our images according to the class as stampede/non-stampede. We could find a limited amount of images which could be used for our application.
Assumptions: 
We had to label our images as stampede/non-stampede as there was no dataset with the label readily available. But we are sure that given a valid dataset, our model would still work.

## Authors

* **Archana Prakash** - [GitHub](https://github.com/ArchPrak)
* **Hritvik Patel**  - [GitHub](https://github.com/hritvikpatel4)
* **Shreyas BS** - [GitHub](https://github.com/bsshreyas99)
