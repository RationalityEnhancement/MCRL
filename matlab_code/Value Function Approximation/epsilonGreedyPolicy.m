function action=epsilonGreedyPolicy(state,actions,features,weights,epsilon)

if rand()<epsilon
    %choose randomly
    action=draw(actions);
else
    %choose greedily
    nr_actions=numel(actions);
    q_hat=zeros(nr_actions,1);
    for a=1:nr_actions
        f=features(state,actions(a));
        q_hat(a)=dot(f,weights);
    end
    
    action=actions(argmax(q_hat));
end

end