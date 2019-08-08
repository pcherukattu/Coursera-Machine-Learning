function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y),; % number of training examples

% You need to return the following variables correctly 
J = 0;

X;y;theta;

grad = zeros(size(theta));
temp1= log(sigmoid([theta'*X']'));
temp1=-y.*temp1;
temp2=log(1-sigmoid([theta'*X']'));
temp2=(1-y).*temp2;
J=sum((temp1-temp2))/m;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
tempg1=sigmoid([theta'*X']')-y;
tempg2=tempg1'*X,
grad=tempg2'/m,

  





% =============================================================

end
