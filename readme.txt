***Implementation of Neural Optimal Control***

NOTE: We rely on cuda enabled jax library. This does is not included in the Pipfile.
- run pipenv shell
- Depending on preference either install 
	- pipenv run pip install --upgrade "jax[cuda12_local]" # for local CUDA toolkit
	- pipenv run pip install -U "jax[cuda12]" # for using provided bins
- run pipenv install for the rest of the deps

If absolute import error occurs, run the following command:
- pipenv install -e /path_to/mjx_fitted_iteration/

On MacOS mujoco can not handle launch passive, you need to use mjpython instead of python which is bundled with mujoco install for example

- mjpython runner.py instead of python/python3 runner
