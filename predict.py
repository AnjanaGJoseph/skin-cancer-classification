from fastai.vision import *



learn = load_learner('./')

img = "./test.jpg"

learn.predict(img)
