***Implementation of Neural Optimal Control***
<<<<<<< HEAD
NOTE: We rely on cuda enabled jax library. This does is not included in the Pipfile.
- run pipenv shell
- Depending on preference either install 
=======
- We rely on cuda enabled jax library. This is not included in the Pipfile.
- Depending on preference after initialising pipenv shell and running pipenv install, either install 
>>>>>>> 1047f3688b3becfb85fad9b09d843afb9f04b0be
	- pipenv run pip install --upgrade "jax[cuda12_local]" # for local CUDA toolkit
	- pipenv run pip install pip install -U "jax[cuda12]" # for using provided bins
- run pipenv install for the rest of the deps
