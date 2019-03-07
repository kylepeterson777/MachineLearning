
% DenseNet Binary Classification
% Created by: Kyle T. Peterson

% images.data contains 128-by-128 images in a 4-D array.
Xdata = double(data.images.data) / 255; score = data.images.labels;
Xdata = repmat(im, [1,1,3]);
classlabel = zeros(length(score),1);
% assign binary class label

classlabel = classlabel';
yclass = categorical(classlabel);

% Display some of the images.
figure;
for j = 1:30
    subplot(6,5,j);
    selectImage = datasample(Xdata,1,4);
    imshow(selectImage,[]);
end

% Freeze first 5 layers in training
layers = dense_ray.Layers;
connections = dense_ray.Connections;
layers(1:5) = freezeWeights(layers(1:5));
dense_ray = createLgraphUsingConnections(layers,connections);

ind1 = find(classlabel==0); ind2 = find(classlabel==1);
n1 = length(ind1); n2 = length(ind2);

rand('state',0); % For reproducibility

sensitivity_test_vector = [];
specificity_test_vector = [];
pred_power_positive_test_vector = [];
pred_power_negative_test_vector = [];
overall_accuracy_test_vector = [];
F_measure_test_vector = [];
G_mean_test_vector = [];

max_run = 100;
% for runi=1:max_run
% partition dataset
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

% augment training data with image rotations

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
% load network  
load denseNet.mat
% options settings for the stochastic gradient descent with momentum.
options = trainingOptions('sgdm',...
        'MaxEpochs',20,... 
        'InitialLearnRate',3e-4, ...
        'Shuffle','every-epoch', ...
        'MiniBatchSize',64,...
        'Plots','training-progress');
    
[net,traininfo] = trainNetwork(X_train_augment,y_train_augment,denseNet,options);

ypred_validate = classify(net,X_validate);

% Run the trained network on test set that was not used to train the
% network and predict the image labels.

[ypred_test,scores_test] = classify(net,X_test);
% scores_test outputs the probabilities for the classes, the 1st column for
% the bad class (0), and the 2nd column for the good class (1).

scores_test_class0 = scores_test(:,1);

% ROC curve, test set data.
[x_axis,y_axis,T,AUC] = perfcurve(y_test, scores_test_class0, '0');
figure; plot(x_axis, y_axis); xlabel('False positive rate'); ylabel('True positive rate');
graph_string = {['Test data, ROC AUC: ', num2str(AUC)]};
text(0.1,0.1,graph_string);

% PRC curve, precision-recall plot, test set data.
[x_axis1,y_axis1,T1,AUC1] = perfcurve(y_test, scores_test_class0, '0', 'XCrit', 'tpr', 'YCrit', 'ppv');
figure; plot(x_axis1, y_axis1); xlabel('Recall'); ylabel('Precision');
graph_string1 = {['Test data, PRC AUC: ', num2str(AUC1)]};
text(0.1,0.06,graph_string1);

% Compute the confusion matrix.
targets(:,1)=(y_test=='0');
targets(:,2)=(y_test=='1');
outputs(:,1)=(ypred_test=='0');
outputs(:,2)=(ypred_test=='1');

figure; plotconfusion(double(targets'),double(outputs'))
