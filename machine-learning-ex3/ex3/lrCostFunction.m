function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

sTheta = size(theta);
yMatrix = [y 1-y]';
xTheta = X*theta;
hx = -log(sigmoid(xTheta));
oneMinusHx = -log(1-sigmoid(xTheta));
hxMatrix = [hx oneMinusHx];
productMatrix = hxMatrix*yMatrix;

J = sum(diag(productMatrix))/m + (lambda/(2.0*m))*theta(2:sTheta)'*theta(2:sTheta);

grad = X'*(sigmoid(xTheta)-y)/m;
grad2 = theta*lambda/(m*1.0);
grad2(1) = 0;
grad = grad + grad2;







% =============================================================

grad = grad(:);

end
