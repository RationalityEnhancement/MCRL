function y = OrdinalRankings(x)
%
%   Calculate the ORDINAL RANKINGS of vector x (in ascending order)
%
%   For details regarding ranking methodologies: http://en.wikipedia.org/wiki/Ranking
%
%
% INPUT
%   The user supplies the data vector x.
%
%
% EXAMPLE 1:
%   x = [32 73 46 32 95 73 87 73 22 69 13 57];
%   y = OrdinalRankings(x);
%   sortrows([x', y], 1)
%   ans =
%            13     1
%            22     2
%            32     4
%            32     3
%            46     5
%            57     6
%            69     7
%            73    10
%            73     9
%            73     8
%            87    11
%            95    12
%
%
% EXAMPLE 2:
%   x = ceil(10000 * rand(10000000, 1));
%   tic;
%   y = OrdinalRankings(x);
%   toc
%   Elapsed time is 21.538000 seconds.
%
%
% EXAMPLE 3:
%   x = rand(10000000, 1);
%   tic;
%   y = OrdinalRankings(x);
%   toc
%   Elapsed time is 20.535000 seconds.
%
%
%   Mike Paduada has provided very useful suggestions which I woul like
%   to acknowledge.
%
%
%-*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-*%
%                                                                                               %
%            Author: Liber Eleutherios                                             %
%            E-Mail: libereleutherios@gmail.com                             %
%            Date: 10 April 2008                                                      %
%                                                                                               %
%-*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-*%
% 
% 
% 

% Prepare data
ctrl = isvector(x) & isnumeric(x);
if ctrl
  x = x(:);
  x = x(~isnan(x) & ~isinf(x));
else
  error('x is not a vector of numbers! The Ordinal Rankings could not be calculated')
end

i = length(x);
j = randsample(i, i);       % let's shuffle the original data
y = x(j);

i = (1:i)';
j = [j, i];
j = sortrows(j, 1);
j(:, 1) = [];               % j is such that all(x == y(j)) ---> 1

[k, k] = sort(y);
y(k) = i;

y = y(j);
