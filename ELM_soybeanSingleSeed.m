% Single Seed Soybean Classification
% Extreme Learning Machine (ELM)
% Author: Kyle T. Peterson

% data loading and preprocessing

data = csvread('SingleSeedCleand_1.xlsx');

rand('state', 0)
data = data(randperm(size(data,1)),:);
% remove rows with NaNs;
ixToRemove = sum(isnan(data),2) > 0;
data(ixToRemove,:) = [];

% get number of inputs and patterns
[nPatterns, nInputs] = size(data);
nInputs = nInputs - 1; % last column is target data

% normalize inputs data between -1 and 1
for i = 1 : nInputs
    data(:,i) = -1 + 2.*(data(:,i) - min(data(:,i)))./(max(data(:,i)) - min(data(:,i)));
end

% verify target data (2 classes identified by 0 and 1)
if numel(unique(data(:,end))) > 2
    error('Not a binary classification problem!');
else
    classLabels = unique(data(:,end));
    data(data(:,end) == classLabels(1),end) = 0;
    data(data(:,end) == classLabels(2),end) = 1;
end

% divide datasets
percTraining = 0.75; % 75% data for training
endTraining  = ceil(percTraining * nPatterns);

trainData = data(1:endTraining,:); 
validData = data(endTraining+1:end,:);

%% creation and training of ELM model
% defined number of hidden neurons to use
nHidden = 300;

% create ELM for classification
ELM = ELM_MatlabClass('CLASSIFICATION',nInputs,nHidden);

% train ELM on the training dataset
ELM = train(ELM,trainData);

% compute and report accuracy on training dataset
Yhat = predict(ELM,trainData(:,1:end-1));
fprintf('TRAINING ACCURACY = %3.2f %%\n',computeAccuracy(trainData(:,end),Yhat)*100);


%% validation of ELM model
Yhat = predict(ELM,validData(:,1:end-1));
fprintf('VALIDATION ACCURACY = %3.2f %%\n',computeAccuracy(validData(:,end),Yhat)*100);

%% sensitivity analysis on number of hidden neurons
nHidden    = [50,100,200,300,305];
trainACC   = zeros(size(nHidden));
validACC   = zeros(size(nHidden));
for i = 1 : numel(nHidden)
    % create ELM for classification
    ELM = ELM_MatlabClass('CLASSIFICATION',nInputs,nHidden(i));
    % train ELM on the training dataset
    ELM = train(ELM,trainData);
    Yhat = predict(ELM,trainData(:,1:end-1));
    trainACC(i) = computeAccuracy(trainData(:,end),Yhat)*100;
    % validation of ELM model
    Yhat = predict(ELM,validData(:,1:end-1));
    validACC(i) = computeAccuracy(validData(:,end),Yhat)*100;
end

% plot results
plot(nHidden,[trainACC;validACC],'-o');
xlabel('Number of Hidden Neurons');
ylabel('Accuracy');
legend({'training','validation'},'Location','northwest')



