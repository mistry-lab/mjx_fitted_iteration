***Implementation of Neural Optimal Control***
- We rely on cuda enabled jax library. This does is not included in the Pipfile.
- Depending on preference after initialising pipenv shell and running pip install, either install 
	- pipenv run pip install --upgrade "jax[cuda12_local]" # for local CUDA toolkit
	- pipenv run pip install pip install -U "jax[cuda12]" # for using provided bins
