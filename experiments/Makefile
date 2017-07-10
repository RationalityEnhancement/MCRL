all:
	bin/make_stimuli.py
	bin/convert_json.py
deploy: exp1
	bin/templater.py exp1
	bin/deploy_experiment.py deploy
clean:
	rm experiment/static/json/*
	rm -r deploy