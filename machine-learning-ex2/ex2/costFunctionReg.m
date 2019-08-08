function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
temp1= log(sigmoid(X*theta));
temp1=-y.*temp1;
temp2=log(1-sigmoid(X*theta));
temp2=(1-y).*temp2;
theta_sq=theta.^2;
theta_sq(1)=0;
theta_sqsum=sum(theta_sq);
lam_temp1=lambda/(2*m);
reg_compj=lam_temp1*theta_sqsum;
reg_compj,
temp1=temp1-temp2;
J=sum((temp1+reg_compj))/m;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

tempg1=sigmoid(X*theta)-y;
tempg2=tempg1'*X,
grad=tempg2'/m;
reg_grad=(lambda/m)*theta;
reg_grad(1)=0;
grad=grad+reg_grad;
grad = grad(:);







% =============================================================

end
