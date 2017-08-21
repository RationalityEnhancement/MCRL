scratch:
	coffee test.coffee

test:
	coffee -c src/utils.coffee
	webppl test.wppl --require . --require webppl-timeit
test2:
	coffee -c src/utils.coffee
	webppl test.wppl --require . --require webppl-timeit --compile
	node --max_old_space_size=8096 tmp.js 

bundle:
	coffee -c src/utils.coffee
	cd webppl && grunt bundle:../.
	mv webppl/bundle/webppl.js ../experiments/exp1/static/js/webppl.js