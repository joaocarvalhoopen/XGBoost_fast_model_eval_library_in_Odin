all:
	odin build . -out:xgboost_fast_model_eval.exe -o:speed -no-bounds-check -disable-assert
#	odin build . -out:xgboost_fast_model_eval.exe -o:aggressive -no-bounds-check -disable-assert

clean:
	rm xgboost_fast_model_eval.exe

run:
	./xgboost_fast_model_eval.exe