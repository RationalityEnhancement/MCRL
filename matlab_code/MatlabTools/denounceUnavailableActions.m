function [T_prime,R_prime]=denounceUnavailableActions(T,R)
%T(s_from,s_to,a): transition probability of getting from the s_from
%to s_to by taking the action a.
%R(s_from,s_to,a): reward associated with that transition

nr_states=size(T,1);
nr_actions=size(T,3);

R_prime=R;
T_prime=T;
R_min=min(R(:));

for from=1:nr_states
    for a=1:nr_actions
        if all(T(from,:,a)==0) %action unavailable
            R_prime(from,:,a)=-1000*abs(R_min);
            T_prime(from,:,a)=1/nr_states*ones(1,nr_states,1);
        end        
    end
end

end