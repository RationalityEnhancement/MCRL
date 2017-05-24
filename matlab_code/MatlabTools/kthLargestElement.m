function [element,pos] = kthLargestElement( elements,k )
% kthLargestElement( elements,k ) returns the k-th largest element of the
% array elements.

[sorted_elements,indices]=sort(elements,'descend');

element=sorted_elements(k);
pos=indices(k);

end

