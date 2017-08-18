all:
	bin/make_stimuli.py
	bin/convert_json.py

deploy: exp1
	bin/templater.py exp1 -v variables.yml
	# bin/deploy_experiment.py deploy
	# rm -r deploy

e2a: exp1
	bin/templater.py exp1 -v stage1.yml
	bin/deploy_experiment.py deploy
	rm -r deploy

e2b: exp1
	bin/templater.py exp1 -v stage2.yml
	bin/deploy_experiment.py deploy
	rm -r deploy


clean:
	rm experiment/static/json/*
	rm -r deploy
