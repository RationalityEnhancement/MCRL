# The main script.
main:
	coffee -c src/utils.coffee
	webppl main.wppl --require . --require webppl-timeit

# Same as main but with more memory allocated to Node.
memory:
	coffee -c src/utils.coffee
	webppl main.wppl --require . --require webppl-timeit --compile
	node --max_old_space_size=8096 tmp.js 

# Compile webppl.js for the experiment.
bundle:
	coffee -c src/utils.coffee
	cd webppl && grunt bundle:../.
	mv webppl/bundle/webppl.js ../experiments/exp1/static/js/webppl.js

scratch:
	coffee -c src/utils.coffee
	coffee test.coffee