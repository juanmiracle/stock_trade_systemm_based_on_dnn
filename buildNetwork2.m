%% STEP 0: initiate parameters
inputSize = 80;
numClasses = 5;
hiddenSizeL1 = 100;    % Layer 1 Hidden Size
hiddenSizeL2 = 100;    % Layer 2 Hidden Size
lambda = 0.00001;
sparsityParam = 0.36;
beta = 3; 
alpha = 3;
%%======================================================================
%% STEP 1: Load data 
traindata = csvread('train.csv');
traindata = traindata';
trainLabels = traindata(81,:);
trainData =  1 ./ (1 + exp(-alpha.*traindata(1:80,:)));
clear traindata;
%%======================================================================
%% STEP 2: Train the first autoencoder
%  Randomly initialize the parameters
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);
addpath minFunc/
options.Method = 'lbfgs';
options.maxIter = 50;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

fprintf('Starting train Autoencoder Layer1!\n');
sae1OptTheta = sae1Theta;
[sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p,inputSize, hiddenSizeL1,lambda,sparsityParam,beta, trainData),sae1Theta, options);
% -------------------------------------------------------------------------
%%======================================================================
%% STEP 2: Train the second sparse autoencoder
[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, inputSize, trainData);

%  Randomly initialize the parameters
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);
fprintf('Starting train Autoencoder Layer2!\n');
sae2OptTheta = sae2Theta;
[sae2OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p,hiddenSizeL1, hiddenSizeL2,lambda,sparsityParam,beta, sae1Features),sae2Theta, options);
%%======================================================================
%% STEP 3: Train the softmax classifier
[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, hiddenSizeL1, sae1Features);
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);
fprintf('Starting train SoftmaxModel!\n');
softmaxModel = softmaxTrain(hiddenSizeL2, numClasses, lambda,sae2Features, trainLabels, options);
saeSoftmaxOptTheta = saeSoftmaxTheta;
saeSoftmaxOptTheta = softmaxModel.optTheta(:);
% -------------------------------------------------------------------------
%%======================================================================
%% STEP 5: Finetune softmax model
stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);
% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];
options.maxIter = 100;	  % Maximum number of iterations of L-BFGS to run 
fprintf('Starting  Fine-Tuning!\n');
[stackedAEOptTheta, cost] = minFunc( @(p) stackedAECost(p,inputSize, hiddenSizeL2, numClasses, netconfig,lambda, trainData,trainLabels), stackedAETheta, options);
                          
% -------------------------------------------------------------------------
%%======================================================================
%% STEP 6: Train single layer softmaxModel 
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, trainData, trainLabels, options);
                        
save parameter.mat  stackedAETheta stackedAEOptTheta softmaxModel netconfig inputSize hiddenSizeL2 alpha numClasses;
draw