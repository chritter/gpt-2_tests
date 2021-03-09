local:
	#pip install -r environment.yml
	#conda env create -f environment.yml --name news_dashboard_env
	conda env create -f environment.yml --prefix /home/jovyan/conda_envs/gpt python=3.7
	source activate /home/jovyan/conda_envs/gpt
	#python -m ipykernel install --user --name=news_dashboard_env
save:
	conda env export > environment. yml.
