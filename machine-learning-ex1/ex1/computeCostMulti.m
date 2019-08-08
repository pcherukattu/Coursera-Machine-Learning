function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

H = theta'*X';
T = H'-y;
T = sum(T.^2);
theta_sq(1)=0;
theta_sqsum=sum(theta_sq);
lam_temp1=lambda/(2*m);
reg_compj=lam_temp1*theta_sqsum;
J = (T+reg_compj)/(2*m);

temp_1=T'X;
reg_grad=(lambda/m)*theta;
grad=reg_grad'/m;
reg_grad(1)=0;
grad=grad+reg_grad;


% =========================================================================

end
