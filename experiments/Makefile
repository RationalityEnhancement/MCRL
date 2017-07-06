all:
	bin/make_stimuli.py
	bin/convert_json.py
deploy:
	bin/templater.py
	bin/deploy_experiment.py deploy
clean:
	rm experiment/static/json/*
	rm -r deploy