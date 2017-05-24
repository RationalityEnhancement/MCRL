function [p_delta,values]=deltaDistribution(delta_value,values)

p_delta=zeros(size(values));
p_delta(values==delta_value)=1;

end