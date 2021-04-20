function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

	A=X(:,2);
	B=-(y-(X*theta));
	C=B'*A;


	DefSSEb = sum(B);
	DefSSEa = sum(C);

	bold=theta(1,1);
	aold=theta(2,1);

	theta(1,1) = (bold - (alpha*(DefSSEb/m)));
	theta(2,1) = (aold - (alpha*(DefSSEa/m)));




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
theta = theta(:);
end
