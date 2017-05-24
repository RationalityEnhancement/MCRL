function probabilities=stickBreaking(nr_outcomes,alpha)
            %generate outcome probabilities by a stick-breaking process
            probabilities=zeros(nr_outcomes,1);
            
            cumsum=0;
            for p=1:nr_outcomes-1
                if (alpha==1) %sample stick proportions from uniform distribution
                    probabilities(p)=rand()*(1-cumsum);
                else
                   probabilities(p)=betarnd(1,alpha)*(1-cumsum);
                end
                
                cumsum=cumsum+probabilities(p);
            end
            
            probabilities(nr_outcomes)=1-cumsum;
                                        
end