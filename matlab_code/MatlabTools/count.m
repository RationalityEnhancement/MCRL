function [counts,values]=count(elements,values)
%counts how often each element in values occurs in elements

if not(exist('values','var'))
    values=unique(elements(:));
end
counts=zeros(size(values));
for v=1:length(values)
   counts(v)=sum(elements==values(v)); 
end

end