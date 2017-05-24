function y = Rankings(x, meth)
%
%   y = Rankings(x)
%       Calculate the Fractional Ranking as default
%       If x is [5 9 9 23 67 67 67] it returns [1 2.5 2.5 4 6 6 6]
%       If x is [67 9 67 5 67 23 9] it returns [6 2.5 6 1 6 4 2.5]
%
%   y = Rankings(x, 'fractional')
%       Calculate the Fractional Ranking
%       If x is [5 9 9 23 67 67 67] it returns [1 2.5 2.5 4 6 6 6]
%       If x is [67 9 67 5 67 23 9] it returns [6 2.5 6 1 6 4 2.5]
%
%   y = Rankings(x, 'dense')
%       Calculate the Dense Ranking
%       If x is [5 9 9 23 67 67 67] it returns [1 2 2 3 4 4 4]
%       If x is [67 9 67 5 67 23 9] it returns [4 2 4 1 4 3 2]
%
%   y = Rankings(x, 'competition1')
%       Calculate the Standard Competition Rankings
%       If x is [5 9 9 23 67 67 67] it returns [1 2 2 4 5 5 5]
%       If x is [67 9 67 5 67 23 9] it returns [5 2 5 1 5 4 2]
%
%   y = Rankings(x, 'competition2')
%       Calculate the Modified Competition Rankings
%       If x is [5 9 9 23 67 67 67] it returns [1 3 3 4 7 7 7]
%       If x is [67 9 67 5 67 23 9] it returns [7 3 7 1 7 4 3]
%
%   y = Rankings(x, 'ordinal')
%       Calculate the Ordinal Rankings
%       If x is [5 9 9 23 67 67 67] it returns
%                   [1 2 3 4 5 6 7] OR [1 2 3 4 5 7 6] OR [1 2 3 4 6 5 7] OR [1 2 3 4 6 7 5] OR [1 2 3 4 7 5 6] OR [1 2 3 4 7 6 5] OR
%                   [1 3 2 4 5 6 7] OR [1 3 2 4 5 7 6] OR [1 3 2 4 6 5 7] OR [1 3 2 4 6 7 5] OR [1 3 2 4 7 5 6] OR [1 3 2 4 7 6 5].
%       with uniform probabilities
%       If x is [67 9 67 5 67 23 9] it returns
%                   [5 2 6 1 7 4 3] OR [5 2 7 1 6 4 3] OR [6 2 5 1 7 4 3] OR [7 2 5 1 6 4 3] OR [6 2 7 1 5 4 3] OR [7 2 6 1 5 4 3] OR
%                   [5 3 6 2 7 4 3] OR [5 3 7 2 6 4 3] OR [6 3 5 2 7 4 3] OR [7 3 5 2 6 4 3] OR [6 3 7 2 5 4 3] OR [7 3 6 2 5 4 3].
%       with uniform probabilities
%
%
%
%   EXAMPLE 1: LOOK AT THE DIFFERENCES!
% 
%        x = ceil(rand(20, 1) * 10);
%        y1 = FractionalRankings(x);
%        y2 = DenseRankings(x);
%        y3 = StandardCompetitionRankings(x);
%        y4 = ModifiedCompetitionRankings(x);
%        y5 = OrdinalRankings(x);
%
%   OR, EQUIVALENTLY:
%
%        y1 = Rankings(x, 'fractional');
%        y2 = Rankings(x, 'dense');
%        y3 = Rankings(x, 'competition1');
%        y4 = Rankings(x, 'competition2');
%        y5 = Rankings(x, 'ordinal');
%
%        sortrows([x, y1, y2, y3, y4, y5], 1)
%
%         ans =
%             1.0000    1.0000    1.0000    1.0000    1.0000    1.0000
%             3.0000    2.5000    2.0000    2.0000    3.0000    3.0000
%             3.0000    2.5000    2.0000    2.0000    3.0000    2.0000
%             4.0000    4.5000    3.0000    4.0000    5.0000    5.0000
%             4.0000    4.5000    3.0000    4.0000    5.0000    4.0000
%             5.0000    8.0000    4.0000    6.0000   10.0000    6.0000
%             5.0000    8.0000    4.0000    6.0000   10.0000    7.0000
%             5.0000    8.0000    4.0000    6.0000   10.0000   10.0000
%             5.0000    8.0000    4.0000    6.0000   10.0000    8.0000
%             5.0000    8.0000    4.0000    6.0000   10.0000    9.0000
%             7.0000   11.0000    5.0000   11.0000   11.0000   11.0000
%             8.0000   13.5000    6.0000   12.0000   15.0000   12.0000
%             8.0000   13.5000    6.0000   12.0000   15.0000   13.0000
%             8.0000   13.5000    6.0000   12.0000   15.0000   15.0000
%             8.0000   13.5000    6.0000   12.0000   15.0000   14.0000
%             9.0000   16.0000    7.0000   16.0000   16.0000   16.0000
%            10.0000   18.5000    8.0000   17.0000   20.0000   17.0000
%            10.0000   18.5000    8.0000   17.0000   20.0000   20.0000
%            10.0000   18.5000    8.0000   17.0000   20.0000   19.0000
%            10.0000   18.5000    8.0000   17.0000   20.0000   18.0000
%
%
%   EXAMPLE 2: RANK IN DESCENDING ORDER
% 
%        x = ceil(rand(20, 1) * 10);
%        y1 = FractionalRankings(-x);
%        y2 = DenseRankings(-x);
%        y3 = StandardCompetitionRankings(-x);
%        y4 = ModifiedCompetitionRankings(-x);
%        y5 = OrdinalRankings(-x);
%
%   OR, EQUIVALENTLY:
%
%        y1 = Rankings(-x, 'fractional');
%        y2 = Rankings(-x, 'dense');
%        y3 = Rankings(-x, 'competition1');
%        y4 = Rankings(-x, 'competition2');
%        y5 = Rankings(-x, 'ordinal');
%
%        sortrows([x, y1, y2, y3, y4, y5], 1)
%
%         ans =
%             1.0000   19.5000    9.0000   19.0000   20.0000   20.0000
%             1.0000   19.5000    9.0000   19.0000   20.0000   19.0000
%             2.0000   17.5000    8.0000   17.0000   18.0000   17.0000
%             2.0000   17.5000    8.0000   17.0000   18.0000   18.0000
%             3.0000   14.0000    7.0000   12.0000   16.0000   16.0000
%             3.0000   14.0000    7.0000   12.0000   16.0000   15.0000
%             3.0000   14.0000    7.0000   12.0000   16.0000   12.0000
%             3.0000   14.0000    7.0000   12.0000   16.0000   13.0000
%             3.0000   14.0000    7.0000   12.0000   16.0000   14.0000
%             5.0000   10.5000    6.0000   10.0000   11.0000   11.0000
%             5.0000   10.5000    6.0000   10.0000   11.0000   10.0000
%             6.0000    9.0000    5.0000    9.0000    9.0000    9.0000
%             7.0000    7.5000    4.0000    7.0000    8.0000    8.0000
%             7.0000    7.5000    4.0000    7.0000    8.0000    7.0000
%             8.0000    5.5000    3.0000    5.0000    6.0000    6.0000
%             8.0000    5.5000    3.0000    5.0000    6.0000    5.0000
%             9.0000    3.0000    2.0000    2.0000    4.0000    3.0000
%             9.0000    3.0000    2.0000    2.0000    4.0000    2.0000
%             9.0000    3.0000    2.0000    2.0000    4.0000    4.0000
%            10.0000    1.0000    1.0000    1.0000    1.0000    1.0000
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

if nargin == 1;
  y = FractionalRankings(x);
  return;
end;

switch meth
    case 'fractional'
        y = FractionalRankings(x);
    case 'dense'
        y = DenseRankings(x);
    case 'competition1'
        y = StandardCompetitionRankings(x);
    case 'competition2'
        y = ModifiedCompetitionRankings(x);
    case 'ordinal'
        y = OrdinalRankings(x);
end
