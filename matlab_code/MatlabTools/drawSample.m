function [ sample ] = drawSample( values )
%draws a sample uniformly from the set of provided values

n=numel(values);
index=randi(n);
sample=values(index);

end

