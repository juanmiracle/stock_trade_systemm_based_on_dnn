function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);
groundTruth = full(sparse(labels, 1:numCases, 1));
%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
M = theta*data;
M = bsxfun(@minus,M,max(M,[],1));
M = exp(M);
M = bsxfun(@rdivide,M,sum(M));
cost = -sum(sum(log(M).*groundTruth),2)/size(data,2);
cost = cost + lambda*sum(sum(theta.*theta),2)/2;
thetagrad = -(groundTruth-M)*data'./size(data,2);
thetagrad = thetagrad + lambda.*theta;











% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

