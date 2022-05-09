from fastapi import FastAPI
from ml import predict
from preprocess import preprocess
from sample import Sample
import nest_asyncio
nest_asyncio.apply()

app = FastAPI()


@app.post("/predict/")
def read_items(sample: Sample) -> int:
    sample_dict = sample.__dict__
    preprocessed_sample = preprocess(sample_dict)
    prediction = predict(preprocessed_sample)

    return prediction


@app.get("/whoami")
def whoami() -> str:
    # TODO
    isim = "Muhammed Eyup"
    soyisim = "Basak"
    mail = "eyupbsk06@hotmail.com"
    
    person_card = {
        "isim": isim,
        "soyisim": soyisim,
        "mail": mail
    }

    return person_card


@app.get("/model_card")
def model_card() -> str:
    # TODO

    model_card = {
        'model_name': 'Who will leave the job',
        'model_description': 'Logistic Regression',
        'model_version': 'Version 1.0.2',
        'model_author': 'Muhammed Eyup Basak',
        'model_author_mail': 'eyupbsk06@hotmail.com',
        'model_creation_date': '20.03.2022',
        'model_last_update_date': '31.03.2022',
        'required_parameters_list': '''max_iter=5000,penalty='l2', C=4.281332398719396, random_state = 42,
                                        class_weight={1:5},intercept_scaling = 1, solver='liblinear', multi_class='ovr', n_jobs=1''',
        'required_parameters_descriptions': 'A new feature was created with the ratio of the total number of years worked to the total number of companies. Model is aimed to learn the model better.',
    }

    return model_card


import uvicorn
if __name__ == "__main__":
    uvicorn.run("main:app", host = "127.0.0.1", port= 5000, log_level="info")





