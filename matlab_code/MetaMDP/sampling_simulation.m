nr_simulations=1000;
nr_trials=100;

a=NaN(nr_trials,nr_simulations);
b=NaN(nr_trials,nr_simulations);

a(1,:)=1; b(1,:)=1;
for sim=1:nr_simulations
    for t=1:nr_trials
        p(t,sim)=a(t,sim)/(a(t,sim)+b(t,sim));
        
        if rand<p(t,sim)
            a(t+1,sim)=a(t,sim)+1;
            b(t+1,sim)=b(t,sim);
        else
            a(t+1,sim)=a(t,sim);
            b(t+1,sim)=b(t,sim)+1;
        end
    end
end

figure(),hist(p(nr_trials,:))
figure(),plot(mean(max(p,1-p),2))