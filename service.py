import bentoml
import numpy as np
from bentoml.io import NumpyNdarray
from bentoml.io import JSON

model_ref = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5")
#model_ref = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5")

model_runner = model_ref.to_runner()

svc = bentoml.Service("cool_model_classifier", runners=[model_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(vector):
	prediction = model_runner.predict.run(vector)
	return prediction
