***Implementation of Neural Optimal Control***

- run pipenv shell
- run pipenv install for the rest of the deps

If absolute import error occurs, run the following command:
- pipenv install -e /path_to/mjx_fitted_iteration/

On MacOS mujoco can not handle launch passive, you need to use mjpython instead of python which is bundled with mujoco install for example

- mjpython runner.py instead of python/python3 runner.py

