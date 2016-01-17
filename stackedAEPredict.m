function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.


z = cell(size(stack));
a = cell(size(stack));
delta = cell(size(stack));
z{1} = bsxfun(@plus,stack{1}.w*data,stack{1}.b);
a{1} = sigmoid(z{1});
for d = 2:numel(stack)
    z{d} = bsxfun(@plus,stack{d}.w*a{d-1},stack{d}.b);
    a{d} = sigmoid(z{d});
end
M = softmaxTheta*a{numel(stack)};
M = bsxfun(@minus,M,max(M,[],1));
M = exp(M);
M = bsxfun(@rdivide,M,sum(M));
pred = 1.5.*M(1,:)+1.*M(2,:)-1.5.*M(3,:)-M(4,:);








% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
