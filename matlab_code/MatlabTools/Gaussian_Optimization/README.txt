 The Infinite-Metric Gaussian Process Optimization (IMGPO) algorithm
    
 This MATLAB code implements IMGPO 
            presented in Bayesian Optimization With Exponential Convergence

          
root
 |
 |--IMGPO.m                             IMGPO main source                        | necessary to run IMGPO
 |                                                                               |
 |--supplementary                       IMGPO supplementary source               |
 |             |--draw_function.m                                                |
 |             |--draw_function2.m                                               |
 |             |--draw_function2_GP.m                                            |
 |             |--gp_call.m                                                      |
 |                                                                               |
 |--gpml-matlab-v3.6-2015-07-07         gpml library folder                      | 
 |
 |--IMGPO_default_run.m                 Run IMGPO with default setting           | not necessary to run IMGPO
 |                                                                               | 
 |--IMGPO_run_paper_experiments.m       Script used for experiments in the paper | 
 |                                                                               |
 |--test_function                       test functions folder                    |
 

For a quick use, one would use and modify the script IMGPO_run_paper_experiments.m 
For a quick use with more flexibility, one would call IMGPO_default_run
For a use with detailed settings, one would call IMGPO
 
