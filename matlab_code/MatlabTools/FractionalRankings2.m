function y = FractionalRankings2(x)
%   This is an example of an extremely inefficient Matlab code.
%
%   Calculate the FRACTIONAL RANKINGS of vector x (in ascending order)
%   The Fractional Ranking y is such that sum(y) is equal to sum([1:length(x)])
%
%   For details regarding ranking methodologies: http://en.wikipedia.org/wiki/Ranking
%
% INPUT
%   The user supplies the data vector x.
%
%
% EXAMPLE 1:
%   x = [32 73 46 32 95 73 87 73 22 69 13 57];
%   y = FractionalRankings2(x);
%   sortrows([x, y], 1)
%   ans =
%           13.0000    1.0000
%           22.0000    2.0000
%           32.0000    3.5000
%           32.0000    3.5000
%           46.0000    5.0000
%           57.0000    6.0000
%           69.0000    7.0000
%           73.0000    9.0000
%           73.0000    9.0000
%           73.0000    9.0000
%           87.0000   11.0000
%           95.0000   12.0000
%
%
% EXAMPLE 2:
%   x = ceil(10000 * rand(10000000, 1));
%   tic;
%   y = FractionalRankings2(x);
%   toc
%   Elapsed time is 1009.786000 seconds.
%
%
% EXAMPLE 3:
%   x = rand(10000000, 1);
%   tic;
%   y = FractionalRankings2(x);
%   toc
%   Elapsed time is 10.250000 seconds.
%
%
% EXAMPLE 4:
%   x = rand(10000000, 1);
%   x(1:2) = 0;
%   tic;
%   y = FractionalRankings2(x);
%   toc                         % It takes hours: don't do that.
%
%
%-*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-*%
%                                                                                               %
%            Author: Liber Eleutherios                                             %
%            E-Mail: libereleutherios@gmail.com                             %
%            Date: 8 April 2008                                                       %
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
  error('x is not a vector of numbers! The Fractional Rankings could not be calculated')
end

N = length(x);

if length(unique(x)) == N
  [y, ind] = sort(x);
  y(ind) = 1:N;
  return
end

[temp, ind] = sort(x);
point1 = 1;
point2 = 1;
y = 1;
for i = 2:N
  if (temp(i) > temp(i - 1))
    if point2 == 1
      y(i - 1) = i - 1;
      point1 = i;
    else
      y(point1:(i - 1)) = sum(point1:(i - 1)) / point2;
      point1 = i;
      point2 = 1;
    end
  else
    point2 = point2 + 1;
  end
end
y(point1:i) = sum(point1:i) / point2;
y = sortrows([y', ind], 2);
y(:, 2) = [];
