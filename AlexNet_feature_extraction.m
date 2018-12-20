%% Feature Extraction using AlexNet
% This example shows how to extract learned image features from a pretrained
% convolutional neural network, and use those features to train an image classifier. 
% Feature extraction is the easiest and fastest way use the representational 
% power of pretrained deep networks. Because feature extraction only requires 
% a single pass through the data, it is a good starting point if you do not
% have a GPU to accelerate network training with.
%% Load Data
train_folder = 'f1_cxr\train\'; %load training data
test_folder = 'f1_cxr\test\'; %load test data
categories = {'abnormal', 'normal'};

% |imageDatastore| recursively scans the directory tree containing the
% images. Folder names are automatically used as labels for each image.

trainImages = imageDatastore(fullfile(train_folder, categories), 'LabelSource', 'foldernames'); 
testImages = imageDatastore(fullfile(test_folder, categories), 'LabelSource', 'foldernames'); 

% Extract the class labels from the training and test data.
YTrain = trainImages.Labels;
YTest = testImages.Labels;
%% Display some sample images.

numTrainImages = numel(YTrain);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(trainImages,idx(i));
    imshow(I)
end
%% Load Pretrained Network
% Load the pretrained Alexnet neural network. If Neural Network Toolbox(TM)
% Model _for AlexNet Network_ is not installed, then the software provides
% a download link. AlexNet is trained on more than one million images and
% can classify images into 1000 object categories, such as keyboard, mouse,
% pencil, and many animals. As a result, the model has learned rich feature
% representations for a wide range of images.
net = alexnet;
% Display the network architecture. The network has five convolutional
% layers and three fully connected layers.
net.Layers
inputSize = net.Layers(1).InputSize
%% Extract Image Features
% The network constructs a hierarchical representation of input images. 
% Deeper layers contain higher-level features, constructed using the 
% lower-level features of earlier layers. To get the feature representations
% of the training and test images, use activations on the fully connected 
% layer 'fc7'. To get a lower-level representation of the images, 
% use an earlier layer in the network. The network requires input images of 
% size 227-by-227-by-3 but the images in the image datastores have different sizes. 
% To automatically resize the training and test images before 
% they are input to the network, create augmented image datastores, 
% specify the desired image size, and use these datastores as 
% input arguments to activations.

augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainImages);
augimdsTest = augmentedImageDatastore(inputSize(1:2),testImages);

layer = 'fc7'; % varies for your data
featuresTrain = activations(net,augimdsTrain,layer,'OutputAs','rows');
featuresTest = activations(net,augimdsTest,layer,'OutputAs','rows');
%% Fit Image Classifier
% Use the features extracted from the training images as predictor variables
% and fit a multiclass support vector machine (SVM) using fitcecoc (
% Statistics and Machine Learning Toolbox).

classifier = fitcecoc(featuresTrain,YTrain);

%% Classify Test Images
% Classify the test images using the trained SVM model the features 
% extracted from the test images.

YPred = predict(classifier,featuresTest);

%% Display four sample test images with their predicted labels.

idx = [1 10 100 200];
figure
for i = 1:numel(idx)
    subplot(2,2,i)
    I = readimage(testImages,idx(i));
    label = YPred(idx(i));
    imshow(I)
    title(char(label))
end

%% Calculate the classification accuracy on the test set. 
% Accuracy is the fraction of labels that the network predicts correctly.

accuracy = mean(YPred == YTest)

%% This SVM has high accuracy. If the accuracy is not high enough 
% using feature extraction, then try transfer learning instead.