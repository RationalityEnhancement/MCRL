# Managing the experiment

## Accessing the server

You can SSH to the server with:

```
ssh cocosci@cocosci.berkeley.edu
```

You will need to get the password from the lab manager. You may also want to [setup a SSH key](https://www.debian.org/devel/passwordlessssh).

You will see a folder with the same name as the subdomain you created when you first set up your project. This is the folder you will put all your experiment files, though you will typically not need to do this manually.

### Set up a python virtual environment

Before setting up your experiment, you need to install a python [virtual environment](http://docs.python-guide.org/en/latest/dev/virtualenvs/) so that you can install psiturk. First, change to the directory corresponding to your subdomain (replace `[your-subdomain]` with your acual subdomain name):

```
cd [your-subdomain]
```

Next, create the virtual environment:

```
virtualenv --no-site-packages
```

You should see that this has created a few folders in the current directory, e.g. `bin`, `lib`, and so on. To activate the virtual environment, you need to run:

```
source bin/activate
```

**You will need to run this command every time you want to launch psiTurk**.

### Install psiTurk

To install psiTurk, make sure you have completed the previous steps and then run:

```
pip install psiturk mysql-python python-levenshtein
```

There is currently a bug in psiTurk that prevents it from working out-of-the-box on the Dreamhost server. Luckily, the fix is pretty simple. After you have installed psiTurk, edit the file located at ` /home/cocosci/[your-subdomain]/lib/python2.7/site-packages/psiturk/psiturk_shell.py`. Find the following line:

```
ip_address = str(self.web_services.get_my_ip())
```

and change it to:

```
ip_address = str(self.config.get('Server Parameters', 'host'))
```

## Developing your experiment

The cookiecutter command that you ran already created an example experiment for you in the `experiment` directory (based on the Stroop task example from psiTurk). You can edit the files in the `experiment` to develop your own task. See the [psiTurk documentation](http://psiturk.org/docs/) for more details on this.

One important thing to note: psiTurk uses a `config.txt` file to configure the experiment. You'll see this file exists in the `experiment` directory, but there is *also* a file called `remote-config.txt`. These two files are essentially the same, but `remote-config.txt` will be used as the config file on the remote server. So, **if you make any changes to `config.txt`, make sure you also make them to `remote-config.txt`, and vice versa!**

## Deploying your experiment

From your local computer, run the command:

```
./bin/deploy_experiment.py
```

This will upload all your experiment files to the webserver. If you see an error about the file `tmp/restart.txt` not existing, make sure there is a directory called `tmp` existing on the server:

```
ssh cocosci@cocosci.berkeley.edu
cd [your-subdomain]
mkdir tmp
```

**Note:** Generally it is a good idea to make changes to your experiment locally, and then upload them to the webserver using the `deploy_experiment.py` script. If you do want to make changes from the webserver, you must tell the server to restart the application each time you make a change. To do this, SSH into the webserver, change into the directory from your subdomain, and recreate the `tmp/restart.txt` file:

```
ssh cocosci@cocosci.berkeley.edu
cd [your-subdomain]
rm tmp/restart.txt && touch tmp/restart.txt
```

The deploy script already does this for you, so if you are changing files with that then you don't need to worry about doing this by hand.

## Previewing your experiment

After you have run the deploy script, you can preview your experiment from the following URL:

```
http://[your-subdomain]/
```

If you are using the template for `default.html` provided by this cookiecutter project, then you should be able to click on the link to the ad and it will randomly generate IDs for the worker ID, assignment ID, and HIT ID, allowing you to preview the experiment (and also allowing you to send the link to collaborators).

Note that if you are using the `default.html` file from psiTurk and not the one from this cookiecutter project, then the link to the ad will just redirect you to `http://[your-subdomain/ad` which is not a valid psiTurk URL. You can construct a preview link manually as follows:

```
http://[your-subdomain]/ad?workerId=debug1234&assignmentId=debug1234&hitId=debug1234&mode=debug
```

where you will need to replace `debug1234` with something different each time you want to run through the experiment. Alternately, you can insert the following into your `default.html` template (make sure you also include the `jquery` library in the header), which will create the link for you automatically:

```html
<p>Begin by viewing the <a href="#" id="ad-href">ad</a>.</p>
<script type="text/javascript">
    function makeid()
    {
        var text = "";
        var possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

        for( var i=0; i < 5; i++ )
            text += possible.charAt(Math.floor(Math.random() * possible.length));

        return text;
    }

    var assignmentId = "debug" + makeid();
    var hitId = "debug" + makeid();
    var workerId = "debug" + makeid();
    var href = "/ad?assignmentId=" + assignmentId + "&hitId=" + hitId + "&workerId=" + workerId + "&mode=debug"

    $("#ad-href").attr("href", href);
</script>
```

## Creating HITs

Once your experiment is ready to go, you will next need to actually create the HIT for your experiment. SSH to the webserver, cd to your subdomain directory, activate the virtual environment, cd to your experiment directory, and finally run psiTurk:

```
ssh cocosci@cocosci.berkeley.edu
cd [your-subdomain]
source bin/activate
cd experiment
psiturk
```

From here you can use the psiTurk command line to create your HITs. Note that you do not need to keep the psiTurk shell running (though you would normally need to do so). This is because we have configured the Apache server (running on dreamhost) to run our psiTurk experiment for us. You'll notice that when you run psiTurk, you'll get a prompt that looks like this:

```
[psiTurk server:blocked mode:sdbx #HITs:0]$
```

Notice that it says `server:blocked`. This is because the server is already running---just not through psiTurk (it is running through Apache). That's ok, and is to be expected. Additionally, when you create a HIT, you'll get the following warning message:

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
./bin/fetch_data.py -e [experiment_code_version]
```

where you should replace `[experiment_code_version]` with the version of the data you want to download (corresponding to the `experiment_code_version` parameter in your psiTurk config file when you ran the experiment). This will prompt you for the username and password to download the data. You provided this earlier when you set up the project. Once you specify the correct username and password, it will download the data to the path you specified when you set up the project.
