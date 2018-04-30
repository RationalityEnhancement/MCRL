local:
	cp templates/exp.html index.html
	coffee -o static/js -cb src/*

all: 
	coffee -o static/js -cb src/*

watch:
	coffee -o static/js -cbw src/*

clean:
	rm static/json/*

demo:
	cp templates/demo.html index.html
	rsync -av --delete-after --copy-links . cocosci@cocosci-mcrl.dreamhosters.com:/home/cocosci/cocosci.dreamhosters.com/webexpt/cognitive-tutor-demo