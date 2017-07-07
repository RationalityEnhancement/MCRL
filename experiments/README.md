
# Developing the experiment

## Compile experiment
In this directory execute `make`. This will generate experimental stimuli and compile the experiment coffeescript into javascript. You will need to install coffeescript.

http://coffeescript.org/#installation

To continuously recompile the coffeescript, execute `make watch` in the `experiment/` directory. This is handy when developing. If you don't notice a change in the experiment, check that the compiler hasn't found any syntax errors.

## View experiment
The experiment runs with PsiTurk. You can read documentation here: https://psiturk.org/quick_start/ 

You'll need to install psiturk. Try `pip install psiturk`. You may have better luck if you use a virtual environment. Once you have psiturk installed, run the following commands to view the experiment in a browser.

```
cd experiment/
psiturk -c  # -c means offline mode, starts up faster
# we should now be in the psiturk shell
server on
debug  # opens a browser window
```

Once this window is open, any changes you make to the javascript will take effect as soon as you reload the page.


# Deploying the experiment

In this directory, run
```
make deploy
```

This will upload all your experiment files to the webserver. If you see an error about the file `tmp/restart.txt` not existing, make sure there is a directory called `tmp` existing on the server:

```
ssh cocosci@cocosci.berkeley.edu
cd cocosci-mcrl.dreamhosters.com
mkdir tmp
```

**Note:** Generally it is a good idea to make changes to your experiment locally, and then upload them to the webserver using the `deploy_experiment.py` script. If you do want to make changes from the webserver, you must tell the server to restart the application each time you make a change. To do this, SSH into the webserver, change into the directory from your subdomain, and recreate the `tmp/restart.txt` file:

```
ssh cocosci@cocosci.berkeley.edu
cd cocosci-mcrl.dreamhosters.com
rm tmp/restart.txt && touch tmp/restart.txt
```

The deploy script already does this for you, so if you are changing files with that then you don't need to worry about doing this by hand.

## Creating HITs
Before you create a HIT, make sure you have updated `experiment_code_version` in `remote_config.txt`. Psiturk uses this value to determine how to allocate participants to different conditions, and it is how we differentiate between different runs of the experiment when we preprocess the data.

### Open Psiturk shell
Once your experiment is ready to go, you will next need to actually create the HIT for your experiment. SSH to the webserver, cd to your subdomain directory, activate the virtual environment, cd to your experiment directory, and finally run psiTurk:

```
ssh cocosci@cocosci.berkeley.edu
cd cocosci-mcrl.dreamhosters.com
source bin/activate
cd experiment
psiturk
```

From here you can use the psiTurk command line to create your HITs. Note that you do not need to keep the psiTurk shell running (though you would normally need to do so). This is because we have configured the Apache server (running on dreamhost) to run our psiTurk experiment for us. You'll notice that when you run psiTurk, you'll get a prompt that looks like this:

```
[psiTurk server:blocked mode:live #HITs:0]$
```

Notice that it says `server:blocked`. This is because the server is already running---just not through psiTurk (it is running through Apache). That's ok, and is to be expected. 

### Post the hit

Post a hit with the following command:
```
hit create [NUM_HITS] [PAYMENT] [TIME_LIMIT]
```

For example, `hit create 12 1.00 0.5` creates 12 hits paying $1 each with a time limit of 30 minutes. When you create a HIT, you'll get the following warning message:

```
*****************************
  Your psiTurk server is currently not running but you are trying to create
  an Ad/HIT.  This can cause problems for worker trying to access your
  hit.  Please start the server by first typing 'server on' then try this
  command again.

  If you are using an external server process, press `y` to continue.
  Otherwise, press `n` to cancel:
```

You can go ahead and press `y` to continue.

## Downloading the data

After you've run your experiment, you can download your data to your local computer with the following command:

```
bin/fetch_data.py -v [EXPERIMENT_CODE_VERSION]
```

where you should replace `[experiment_code_version]` with the version of the data you want to download (corresponding to the `experiment_code_version` parameter in your psiTurk config file when you ran the experiment). This will prompt you for the username and password to download the data. You provided this earlier when you set up the project. Once you specify the correct username and password, it will download the data to the path you specified when you set up the project.

This downloads raw data to `data/1/[EXPERIMENT_CODE_VERSION]/human_raw`. The data is pretty hard to use in this form, so you'll probably want to preprocess it:

```
bin/reformat_data.py -v [EXPERIMENT_CODE_VERSION]
```

This reformats the data into separate CSVs for each type of trial. They will be written to `data/1/[EXPERIMENT_CODE_VERSION]/human`.
