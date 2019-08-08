function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y),% number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
H = theta'*X';
T = H'-y;
T = sum(T.^2);
theta_sq=theta.^2;
theta_sq(1)=0;
theta_sqsum=sum(theta_sq);
reg_compj=lambda*theta_sqsum;
J = (T+reg_compj)/(2*m);

temp_1=(H'-y)'*X;
grad= temp_1'/m;
reg_grad=(lambda/m)*theta;
reg_grad(1)=0;
grad=grad+reg_grad;











% =========================================================================

grad = grad(:);

end
