% Deep Convolutional Neural Network (DCNN) with RUS Boosting

% Kyle T. Peterson
% December, 2018

load file.mat;
% images.data contains 2688 64-by-64 images in a 4-D array.
% images.labels contains the categorical array 
Xdata = images.data / 255; classlabel = images.labels;
yclass = categorical(classlabel);
% Display some of the images.
figure;
for j = 1:30
    subplot(6,5,j);
    selectImage = datasample(Xdata,1,4);
    imshow(selectImage,[]);
end
% Define the convolutional neural network architecture.
layers = [imageInputLayer([64 64 1],'DataAugmentation',{'randfliplr','randcrop'});
          convolution2dLayer(9,19);
          reluLayer();
          maxPooling2dLayer(2,'Stride',2);
          convolution2dLayer(6,16);
          reluLayer();
          maxPooling2dLayer(2,'Stride',2);         
          fullyConnectedLayer(2);
          softmaxLayer();
          classificationLayer()];
% options settings for the stochastic gradient descent with momentum.
options = trainingOptions('sgdm',...
      'MaxEpochs',25,... 
      'MiniBatchSize',60);

ind1 = find(classlabel==0); ind2 = find(classlabel==1);
n1 = length(ind1); n2 = length(ind2);
rng(1); % For reproducibility
sensitivity_test_vector = [];
specificity_test_vector = [];
pred_power_positive_test_vector = [];
pred_power_negative_test_vector = [];
overall_accuracy_test_vector = [];
F_measure_test_vector = [];
G_mean_test_vector = [];

max_run = 100;
for runi=1:max_run

temp1 = ind1(randperm(n1)); temp2 = ind2(randperm(n2));
train_portion = 0.8;

n1_train = temp1(1:ceil(n1*train_portion)); n1_test = temp1(ceil(n1*train_portion)+1:end);
n2_train = temp2(1:ceil(n2*train_portion)); n2_test = temp2(ceil(n2*train_portion)+1:end);

index_train = [n1_train';n2_train']; index_test = [n1_test';n2_test'];
X_train = Xdata(:,:,:,index_train); y_train = (yclass(index_train))';
X_test = Xdata(:,:,:,index_test); y_test = (yclass(index_test))';

ensemblesize = 11;
n = size(X_train,4);
weight = ones(n,1)/n;

beta_vector = [];
predClass_test_matrix = [];
for t = 1:ensemblesize
% training class1 and class2 have the same number of samples, that is, the
% number of minor class samples, n1_train.
% random undersampling from n2_train.
n1size = length(n1_train); n2size = length(n2_train); n2train_perm = randperm(n2size);
index_train_boost = [n1_train';(n2_train(n2train_perm(1:n1size)))'];
X_train_boost = Xdata(:,:,:,index_train_boost); y_train_boost = (yclass(index_train_boost))';

[net,traininfo] = trainNetwork(X_train_boost,y_train_boost,layers,options);

% run the trained network on all the training set samples prior to the RUSboosting.
[ypred_train,scores_train] = classify(net,X_train);
% calculate weighted error
ytrain = str2double(cellstr(y_train));
ypredtrain = str2double(cellstr(ypred_train));
delta = abs(ytrain - ypredtrain);
error_index = find(delta > 0);
error = zeros(n,1); error(error_index) = 1;

sum_error = weight' * error;
if sum_error == 0 sum_error = 0.005; end;
if sum_error > 0.5
    ensemblesize = t - 1;
    break;
end

beta = sum_error / (1 - sum_error);
weight = weight .* (beta.^(1 - error));
weight = weight / sum(weight); % normalization

beta_vector = [beta_vector beta];

% Run the trained network on test set that was not used to train the
% network and predict the image labels.

[ypred_test,scores_test] = classify(net,X_test);

% scores_test outputs the probabilities for the classes, the 1st column for
% the immature class (0), and the 2nd column for the mature class (1).

predClass_test = str2double(cellstr(ypred_test));
predClass_test_matrix = [predClass_test_matrix predClass_test];
end;

n_test = size(X_test,4);
n_class = 2;
weighted_voting = zeros(n_test, n_class);

for i = 1:n_test
    for j = 1:n_class        
        tempindex = find(predClass_test_matrix(i,:) == (j-1));
        if ~isempty(tempindex)
        weighted_voting(i,j) = sum(log(1 ./ beta_vector(tempindex)));
        end;
    end;
end;

[maxvalue, predclass_final] = max(weighted_voting,[],2);
predclass_final(predclass_final == 1) = 0;
predclass_final(predclass_final == 2) = 1;

ytest = str2double(cellstr(y_test));

pred_test = predclass_final;
test_label = ytest;
[C,order] = confusionmat(test_label,pred_test)

sensitivity_test = 100 * C(1,1)/sum(C(1,:))
specificity_test = 100 * C(2,2)/sum(C(2,:))
pred_power_positive_test = 100 * C(1,1)/sum(C(:,1));
pred_power_negative_test = 100 * C(2,2)/sum(C(:,2));
overall_accuracy_test = 100 * sum(diag(C))/sum(sum(C));
F_measure_test = (2*pred_power_positive_test*sensitivity_test)/(pred_power_positive_test+sensitivity_test);
G_mean_test = sqrt(sensitivity_test*specificity_test)
sensitivity_test_vector = [sensitivity_test_vector; sensitivity_test];
specificity_test_vector = [specificity_test_vector; specificity_test];
pred_power_positive_test_vector = [pred_power_positive_test_vector; pred_power_positive_test];
pred_power_negative_test_vector = [pred_power_negative_test_vector; pred_power_negative_test];
overall_accuracy_test_vector = [overall_accuracy_test_vector; overall_accuracy_test];
F_measure_test_vector = [F_measure_test_vector; F_measure_test];
G_mean_test_vector = [G_mean_test_vector; G_mean_test];

end;

percent_correct_all = [sensitivity_test_vector specificity_test_vector G_mean_test_vector];
group_variable = {'1';'2';'3'};
boxplot(percent_correct_all, group_variable); ylabel('performance measure (%) on test data');
disp('sensitivity, specificity, G_mean, median:'); median(percent_correct_all)
disp('sensitivity, specificity, G_mean, mean:'); mean(percent_correct_all)
disp('sensitivity, specificity, G_mean, std:'); std(percent_correct_all)
