function [J grad] = nnCostFunction(nn_params, ...
    input_layer_size, ...
    hidden_layer_size, ...
    num_labels, ...
    X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Neural Network Forward Propagation
z1 = X;
a1 = [ones(m, 1) z1]; % 5000 X 401  %Add Bias
z2 = a1 * Theta1'; % (5000 X 401) X (401 X 25) = 5000 X 25
a2 = [ones(m, 1) sigmoid(z2)]; % 5000 X 26   %Add Bias
z3 = a2 * Theta2'; % (5000 X 26) X (26 X 10)
a3 = sigmoid(z3); % 5000 X 10
h = a3; % h is prediction

%Convert y to binary such as if y = 5, it turns to y = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
new_y = zeros(size(y,1), num_labels);
[max_index_y, max_val_y] = max(y, [], 2);

for k = 1:m
    for i = 1:num_labels
        if max_index_y(k) == i
            new_y(k,i) = 1;
        end
    end
end

for i = 1:num_labels
    J = J + sum((1/m).*(-new_y(:,i) .* log(h(:,i)) - (1-new_y(:,i)) .* log(1-h(:,i)))); % Calculates cost
end

%Add regularization term of Theta2
sum_Theta2 = 0;
for k = 2:size(Theta2,2)
    for j = 1:size(Theta2,1)
        sum_Theta2 = sum_Theta2 + sum(Theta2(j,k) .* Theta2(j,k));
    end
end

%Add regularization term of Theta1
sum_Theta1 = 0;
for k = 2:size(Theta1,2)
    for j = 1:size(Theta1,1)
        sum_Theta1 = sum_Theta1 + sum(Theta1(j,k) .* Theta1(j,k));
    end
end

%Regularized Cost
J = J + (lambda/(2*m))*(sum_Theta1 + sum_Theta2);

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.

% Neural Network Back Propagation
d3 = h - new_y; % 5000 X 10
d2 = (d3 * Theta2) .* [ones(size(z2,1), 1) sigmoidGradient(z2)]; % (5000 X 10) X (10 X 26) = 5000 X 26

D2 = d3' * a2; % (10 X 5000) X (5000 X 26) = 10 X 26
D1 = (d2(:, 2:end))' * a1; % (25 X 5000) X (5000 X 401) = 25 X 401

Theta2_grad = (1 / m) * D2; % 10 X 26
Theta1_grad = (1 / m) * D1; % 25 X 401

% Regularization for Back Propagation
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end); % 10 X 26
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end); % 25 X 401

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
