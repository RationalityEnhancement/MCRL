The archive contains the following files:

Rankings.m
FractionalRankings.m
FractionalRankings2.m
DenseRankings.m
StandardCompetitionRankings.m
ModifiedCompetitionRankings.m
OrdinalRankings.m
OrdinalRankings2.m

*****************************************

The principal function is Rankings.m which
can be used to calculate five different kinds
of rankings (in ascending order):

	FRACTIONAL RANKINGS ([1 2.5 2.5 4])
	DENSE RANKINGS ([1 2 2 3])
	STANDARD COMPETITION RANKINGS ([1 2 2 4])
	MODIFIED COMPETITION RANKINGS ([1 3 3 4])
	ORDINAL RANKINGS ([1 2 3 4] OR [1 3 2 4])

Due to its special statistical properties, the
default method is the Fractional Ranking. In fact,
the Fractional Ranking is such that the sum of N
ranks is equal to sum([1:N]), so that the average
rank of N items is always the same.

For details regarding ranking methodologies:
http://en.wikipedia.org/wiki/Ranking

*****************************************

The file FractionalRankings2.m contains an alternative
function for the computation of Fractional Rankings but
it's extremely slow and inefficient and, actually, it
is not supposed to be used but for general consultation.

I chose to keep it because it is the most intuitive
code, though it cannot be implemented efficiently in
Matlab.


The file OrdinalRankings2 contains an alternative
function for the computation of Ordinal Rankings that
seems to be slightly less efficient than the one
contained in the file OrdinalRankings.m, and also
much more complicated and less readable.

I chose to leave it in the archive for general
consultation.

*****************************************

%-*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-*%
%                                                                                               %
%            Author: Liber Eleutherios                                             %
%            E-Mail: libereleutherios@gmail.com                             %
%            Date: 8 April 2008                                                       %
%                                                                                               %
%-*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-* -*-*%
