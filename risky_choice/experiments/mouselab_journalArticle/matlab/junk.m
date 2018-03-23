    psumsum = 0;
    cont = true;
    while (psumsum != 1 || cont){
        cont = true;
        psumsum = 0;
        psum = 0;
        for (o=0;o<nr_outcomes;o++){
            prob = 0;
            while (prob==0){
                prob = Math.round(Math.random()*100)/100;
            }
            probabilities[o]=prob;
            psum+=probabilities[o];
        }
        psumsum = 0;
        for (o=0;o<nr_outcomes;o++){
            if (probabilities[o]<0.01){
                cont = true;
            }
        }
        for (o=0;o<nr_outcomes;o++){   
            probabilities[o] = Math.round(probabilities[o]/psum*100)/100;
            psumsum+=probabilities[o];
        }
        if (isHighCompensatory[block_nr-1][trial_nr-1]){
            for (o=0;o<nr_outcomes;o++){
                if (probabilities[o]>=0.85){
                    cont = false
                }
            }
        }
        else{
            cont = false
            for (o=0;o<nr_outcomes;o++){
                if (probabilities[o]>=0.4 || probabilities[o]<=0.1){
                    cont = true
                }
            }
        }
    }