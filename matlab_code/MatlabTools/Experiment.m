classdef Experiment
    %Parent Class for simulating experiments
    %   specific experiments inherit from this class.
    
    properties
        blocks %struct array where the i-th entry specifies the parameters of the i-th block        
        nr_blocks
        stimuli
        p_stimulus_given_trial_type
        correct_response
    end
    
    methods (Access = public)
        
        function experiment=Experiment(blocks,p_stimulus_given_trial_type,stimuli,correct_response)
            experiment.blocks=blocks;
            experiment.nr_blocks = numel(blocks);
            experiment.stimuli = stimuli;
            experiment.p_stimulus_given_trial_type = p_stimulus_given_trial_type;
            experiment.correct_response = correct_response;
        end
        
        function experiment=addBlock(experiment,block)
            experiment.blocks(end+1)=block;
            experiment.nr_blocks=experiment.nr_blocks+1;
        end
        
        function results=simulateExperiment(experiment,model,nr_simulations,nr_subjects)
            
            switch nargin
                case 2
                    nr_simulations = 1;
                    nr_subjects = 1;
                case 3
                    nr_subjects = 1;
            end                    
            
            results=struct('trials',[],'responses',[]);            
            
            for sim=1:nr_simulations
                for sub=1:nr_subjects
                    
                    model=model.newSubject();
                    
                    for b=1:experiment.nr_blocks
                        results(b).trials=struct('stimulus',[],'task',[],'type',[]);
                        
                        trials=experiment.generateTrials(b);
                        results(b).trials(1:experiment.blocks(b).nr_trials,sub,sim) = trials;
                        
                        
                        for t=1:experiment.blocks(b).nr_trials
                            
                            results(b).trial_type_nrs(t,sub,sim)=trials(t).type;
                            
                            %TODO: refactor because control_signal is
                            %specific to Stroop experiment
                            [results(b).responses{t,sub,sim},results(b).RT(t,sub,sim),...
                                results(b).control_signal(t,sub,sim)] = ...
                                model.simulateResponse(trials(t));
                            
                            %trials(t).stimulus
                            %experiment.correct_response(trials(t))
                            
                            results(b).correct(t,sub,sim)=strcmpi(results(b).responses{t,sub,sim},...
                                experiment.correct_response(trials(t)));
                            results(b).controlled_response(t,sub,sim)=strcmpi(...
                                results(b).responses{t,sub,sim},trials(t).stimulus{1});
                            
                            if results(b).correct(t,sub,sim)
                                results(b).reward(t,sub,sim) = experiment.blocks(b).task.reward_by_trial_type(trials(t).type);
                            else
                                results(b).reward(t,sub,sim) = 0;%-experiment.blocks(b).task.reward_by_trial_type(trials(t).type);
                            end                            
                            
                            [model,results(b).VOC_model(t,sub,sim)]=model.learn(trials(t).stimulus,...
                                results(b).reward(t,sub,sim),results(b).control_signal(t,sub,sim),...
                                results(b).RT(t,sub,sim),results(b).correct(t,sub,sim));

                            results(b).feature_weights(:,t,sub,sim)=model.metalevel_model.glm_EVOC.mu_n;
                            results(b).feature_names=model.feature_names; 
                            
                        end
                        
                        results(b).avg_reward_rate = mean(results(b).reward(:)./results(b).RT(:));
                        results(b).sem_reward_rate = sem(results(b).reward(:)./results(b).RT(:));                        
                    end
                end
            end
            
        end
    end
    
    methods (Access = protected)
        function trial_types=generateTrialTypes(experiment,block_nr)
            
            p_trial_type=reshape(experiment.blocks(block_nr).p_trial_type,...
                [1,experiment.blocks(block_nr).nr_trial_types]);
            trial_types = sampleDiscreteDistributions(p_trial_type,experiment.blocks(block_nr).nr_trials);
        end
        
        function trials=generateTrials(experiment,block_nr)
            
            nr_trials = experiment.blocks(block_nr).nr_trials;
            nr_features = size(experiment.stimuli,2);            
            stimulus = zeros(nr_trials,nr_features);
            
            trial_types=experiment.generateTrialTypes(block_nr);
                                    
            for t=1:experiment.blocks(block_nr).nr_trials
                p_stimulus=experiment.p_stimulus_given_trial_type(trial_types(t),:);
                stimulus_type = sampleDiscreteDistributions(p_stimulus,1);
                trials(t).stimulus=experiment.stimuli{stimulus_type};
                trials(t).task=experiment.blocks(block_nr).task;
                trials(t).type=trial_types(t);                
            end
        end
       
    end
    
    methods (Abstract)
    end
    
end