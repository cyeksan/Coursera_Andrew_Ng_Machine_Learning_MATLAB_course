function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%Linear regression
y_pred = zeros(size(y));
y_pred = X * theta;


%Cost calc.
J = sum((1/(2*m)).*(y_pred-y).^2) + sum((lambda/(2*m))*theta.*theta) - sum((lambda/(2*m))*theta(1).*theta(1)); %% Add theta0 term back.

%Grad calc.
grad = (1/m)*(X'*(y_pred-y));

temp = theta;
temp(1) = 0;

grad = grad + (lambda/m)*temp;












% =========================================================================

grad = grad(:);

end
