function y = sin2D_high(x)
% 
% Branin function 
% Matlab Code by A. Hedar (Sep. 29, 2005).
% The number of variables n = 2.
% 

	bounds_branin = [0,1; 0,1];
	x(1:2) = bounds_branin(1:2, 1)' + ...
	        ((x(1:2)+1)/2).*(bounds_branin(1:2, 2) - bounds_branin(1:2, 1))';

    y = (sin(13 .* x(1)) .* sin(27 .* x(1)) ./ 2.0 + 0.5) .* (sin(13 .* x(2)) .* sin(27 .* x(2)) ./ 2.0 + 0.5);
end