function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%
z = cell(size(stack));
a = cell(size(stack));
delta = cell(size(stack));
z{1} = bsxfun(@plus,stack{1}.w*data,stack{1}.b);
a{1} = sigmoid(z{1});
if numel(stack)>=2
for d = 2:numel(stack)
    z{d} = bsxfun(@plus,stack{d}.w*a{d-1},stack{d}.b);
    a{d} = sigmoid(z{d});
end
end
M = softmaxTheta*a{numel(stack)};
M = bsxfun(@minus,M,max(M,[],1));
M = exp(M);
M = bsxfun(@rdivide,M,sum(M));
cost = -sum(sum(log(M).*groundTruth),2)/size(data,2);
cost = cost + lambda*sum(sum(softmaxTheta.*softmaxTheta),2)/2;
softmaxThetaGrad  = -(groundTruth-M)*a{numel(stack)}'./size(data,2);
softmaxThetaGrad = softmaxThetaGrad  + lambda.*softmaxTheta;
delta{numel(stack)} = -softmaxTheta'*(groundTruth-M).*(1-a{numel(stack)}).*a{numel(stack)};
if numel(stack)>=2
for d=1:numel(stack)-1
    delta{numel(stack)-d} = stack{numel(stack)-d+1}.w'*delta{numel(stack)-d+1}.*(1-a{numel(stack)-d}).*a{numel(stack)-d};
end
end
stackgrad{1}.w = delta{1}*data'/size(data,2);
stackgrad{1}.b = sum(delta{1},2)/size(data,2);
if numel(stack)>=2
for d=2:numel(stack)
    stackgrad{d}.w = delta{d}*a{d-1}'/size(data,2);
    stackgrad{d}.b = sum(delta{d},2)/size(data,2);
end
end


% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
