# Webppl mouselab
This package contains code for simulating the mouselab-mdp meta-MDP, and for computing a feature-based approximation to the optimal meta-Q function.

## Files

### main.wppl
The main script, executed by `make main`

### mouselab.wppl
Defines the meta-MDP and policies for the MDP.

### features.wppl
Defines the features (VOC_1, VPI_action, VPI_full, etc..) used to approximate the optimal meta-Q function.

### utils.coffee
Utilities. Any code that is easier to write with mutable state lives here, including the code that builds the mouselab environment.

# Webppl resources
The very basics: http://dippl.org/chapters/02-webppl.html
More detailed introduction: http://agentmodels.org/chapters/2-webppl.html
Modeling agents: http://agentmodels.org/