% Deep Learning using Convolutional Neural Network (CNN)
% Hyperparameters are automatically determined via Bayesian optimization
%%
% Kyle T. Peterson
% December, 2018
%%


clear all;clc; close all;

load data.mat;
% images.data contains 128-by-128 images in a 4-D array.
% images.labels contains the categorical array of the classes 0,1 .
Xdata = double(data.images.data) / 255; WG_score = data.images.labels;
classlabel = zeros(length(WG_score),1);
for i = 1:length(WG_score)
    if strncmp(WG_score(i),'Ab',2) | strncmp(WG_score(i),'ab',2)
        classlabel(i) = 0; % class 0.
    elseif strncmp(WG_score(i),'De',2)
        classlabel(i) = 0; % class 0.
    elseif strncmp(WG_score(i),'Nor',3)
        classlabel(i) = 1; % class 1.
    else
        classlabel(i) = -1;
    end
end
% class 0 for low quality class , 1 - high quality class 

classlabel = classlabel';
yclass = categorical(classlabel);
ind1 = find(classlabel==0); ind2 = find(classlabel==1);
n1 = length(ind1); n2 = length(ind2);
rand('state',0); % For reproducibility

%% partition dataset
temp1 = ind1(randperm(n1)); temp2 = ind2(randperm(n2));
train_portion = 0.7; validate_portion = 0.15;  test_portion = 0.15;
n1_train = temp1(1:ceil(n1*train_portion));
n1_validate = temp1(ceil(n1*train_portion)+1:ceil(n1*train_portion)+ceil(n1*validate_portion));
n1_test = temp1(ceil(n1*train_portion)+ceil(n1*validate_portion)+1:end);

n2_train = temp2(1:ceil(n2*train_portion));
n2_validate = temp2(ceil(n2*train_portion)+1:ceil(n2*train_portion)+ceil(n2*validate_portion));
n2_test = temp2(ceil(n2*train_portion)+ceil(n2*validate_portion)+1:end);

index_train = [n1_train';n2_train'];
index_validate = [n1_validate';n2_validate'];
index_test = [n1_test';n2_test'];

X_train = Xdata(:,:,:,index_train); y_train = (yclass(index_train))';
X_validate = Xdata(:,:,:,index_validate); y_validate = (yclass(index_validate))';
X_test = Xdata(:,:,:,index_test); y_test = (yclass(index_test))';

save train_validate_test_data index_train index_validate index_test;

%% augment training data with image rotations

X_train_augment = X_train; y_train_augment = y_train;
nboost = size(X_train,4);
for index_image = 1:nboost
    rotated_image = imrotate(X_train(:,:,:,index_image),ceil(rand*360),'bilinear','crop');
    X_train_augment(:,:,:,size(X_train_augment,4)+1) = rotated_image;
    y_train_augment = [y_train_augment;y_train(index_image)];
    
    rotated_image = imrotate(X_train(:,:,:,index_image),ceil(rand*360),'bilinear','crop');
    X_train_augment(:,:,:,size(X_train_augment,4)+1) = rotated_image;
    y_train_augment = [y_train_augment;y_train(index_image)];
    
    rotated_image = imrotate(X_train(:,:,:,index_image),ceil(rand*360),'bilinear','crop');
    X_train_augment(:,:,:,size(X_train_augment,4)+1) = rotated_image;
    y_train_augment = [y_train_augment;y_train(index_image)];
end
%% Display some of the images.
figure;
for j = 1:30
    subplot(6,5,j);
    selectImage = datasample(Xdata,1,4);
    imshow(selectImage,[]);
end

%% Bayesian optimization
optimVars = [
    optimizableVariable('NetworkDepth',[1 10],'Type','integer')
    optimizableVariable('InitialLearnRate',[1e-3 1e-1],'Transform','log')
    optimizableVariable('Momentum',[0.75 0.95])
    optimizableVariable('L2Regularization',[1e-10 1e-2],'Transform','log')];
ObjFcn = makeObjFcn(X_train,y_train,X_validate,y_validate);
BayesObject = bayesopt(ObjFcn,optimVars,...
    'MaxObj',30,...
    'MaxTime',8*60*60,...
    'IsObjectiveDeterministic',false,...
    'UseParallel',false);
bestIdx = BayesObject.IndexOfMinimumTrace(end);
fileName = BayesObject.UserDataTrace{bestIdx};
savedStruct = load(fileName);
valError = savedStruct.valError

[ypred_test,scores_test] = classify(savedStruct.trainedNet,XTest);

% scores_test outputs the probabilities for the classes, the 1st column for
% the bad class (0), and the 2nd column for the good class (1).

scores_test_class0 = scores_test(:,1);

%% ROC curve, test set data.
[x_axis,y_axis,T,AUC] = perfcurve(y_test, scores_test_class0, '0');
figure; plot(x_axis, y_axis); xlabel('False positive rate'); ylabel('True positive rate');
graph_string = {['Test data, ROC AUC: ', num2str(AUC)]};
text(0.1,0.1,graph_string);

%% PRC curve, precision-recall plot, test set data.
[x_axis1,y_axis1,T1,AUC1] = perfcurve(y_test, scores_test_class0, '0', 'XCrit', 'tpr', 'YCrit', 'ppv');
figure; plot(x_axis1, y_axis1); xlabel('Recall'); ylabel('Precision');
graph_string1 = {['Test data, PRC AUC: ', num2str(AUC1)]};
text(0.1,0.06,graph_string1);

%% Compute the confusion matrix.
targets(:,1)=(y_test=='0');
targets(:,2)=(y_test=='1');
outputs(:,1)=(ypred_test=='0');
outputs(:,2)=(ypred_test=='1');
figure; plotconfusion(double(targets'),double(outputs'))

%% display extracted network features
layer = 'conv1'; %layer to be visualized
channels = 1:30;
I = deepDreamImage(trainedNet,layer, channels,...
    'PyramidLevels',1,'Verbose',0);
figure;
for j = 1:30
    subplot(6,5,j);
    imshow(I,[]);
end

function ObjFcn = makeObjFcn(XTrain,YTrain,XValidation,YValidation)
ObjFcn = @valErrorFun;
    function [valError,cons,fileName] = valErrorFun(optVars)
    imageSize = [128 128 1];
        numClasses = numel(unique(YTrain));
        initialNumFilters = round(16/sqrt(optVars.NetworkDepth));
        layers = [
            imageInputLayer(imageSize)
            
            convBlock(3,initialNumFilters,optVars.NetworkDepth)
            maxPooling2dLayer(2,'Stride',2)
            
            convBlock(3,2*initialNumFilters,optVars.NetworkDepth)
            maxPooling2dLayer(2,'Stride',2)
            
            convBlock(3,2*initialNumFilters,optVars.NetworkDepth)
            maxPooling2dLayer(2,'Stride',2)
            
            convBlock(3,2*initialNumFilters,optVars.NetworkDepth)
            maxPooling2dLayer(2,'Stride',2)            
            
            convBlock(3,4*initialNumFilters,optVars.NetworkDepth)
            averagePooling2dLayer(4)
            
            % Add the fully connected layer and the final softmax and
            % classification layers.
            fullyConnectedLayer(2)
            softmaxLayer
            classificationLayer];
        % Model training parameters
        miniBatchSize = 64;
        validationFrequency = floor(numel(YTrain)/miniBatchSize);
        options = trainingOptions('sgdm',...
            'InitialLearnRate',optVars.InitialLearnRate,...
            'Momentum',optVars.Momentum,...
            'MaxEpochs',50, ...
            'LearnRateSchedule','piecewise',...
            'LearnRateDropPeriod',35,...
            'LearnRateDropFactor',0.1,...
            'MiniBatchSize',miniBatchSize,...
            'L2Regularization',optVars.L2Regularization,...
            'Shuffle','every-epoch',...
            'Verbose',false,...
            'Plots','training-progress',...
            'ValidationData',{XValidation,YValidation},...
            'ValidationPatience',Inf,...
            'ValidationFrequency',validationFrequency);
        
        % increase the variety of samples synthetically
        pixelRange = [-5 5]; %image shift scale
        imageAugmenter = imageDataAugmenter(...
            'RandXReflection', true,...
            'RandYTranslation',pixelRange,...
            'RandYTranslation',pixelRange);
        datasource = augmentedImageDatastore(imageSize,XTrain,YTrain,...
            'DataAugmentation',imageAugmenter,...
            'OutputSizeMode','randcrop');
        trainedNet = trainNetwork(datasource,layers,options);
        close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_FIGURE'))
        YPredicted = classify(trainedNet,XValidation);
        valError = 1 - mean(YPredicted == YValidation);
        [gmeanValid] = classAccuracy(YValidation,YPredicted);
        fileName = num2str(valError) + ".mat";
        save(fileName,'trainedNet','valError','gmeanValid','options')
        cons = [];
        
    end
end

function layers = convBlock(filterSize,numFilters,numConvLayers)
layers = [
    convolution2dLayer(filterSize,numFilters,'Padding','same')
    batchNormalizationLayer
    reluLayer];
layers = repmat(layers,numConvLayers,1);
end
