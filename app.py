import uvicorn
import tensorflow as tf
import pandas as pd
import numpy as np
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from transformers import BertTokenizer
from processor import DataLoader
from fastapi import FastAPI, Request, Query, Form

app = FastAPI(version='0.0.0.',
              title='Test')

templates = Jinja2Templates(directory="template/")

tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
with tf.device('/cpu:0'):
    roberta = tf.saved_model.load('roberta/back_translate_roberta_model/')
print('Finishing Loading')


@app.get('/{text}')
async def main(request: Request, text: str = Query(None, description='text data')):
    try:
        train_BTL = DataLoader(data=preprocess_data(text).iloc[:],
                               MAX_SEQUENCE_LENGTH=200,
                               batch_size=1,
                               imbalance_sample=False,
                               tokenizer=tokenizer,
                               mode='on-line')
        feature = next(train_BTL.__iter__())
        res, score = predict(feature)
        result = {'predictions': res, 'score': score}
        return result

    except Exception as e:
        return e


@app.get("/")
async def form_post(request: Request):
    return templates.TemplateResponse('index.html', context={'request': request})


@app.post("/")
async def form_post(request: Request, content=Form(...)):
    train_BTL = DataLoader(data=preprocess_data(content).iloc[:],
                           MAX_SEQUENCE_LENGTH=200,
                           batch_size=1,
                           imbalance_sample=False,
                           tokenizer=tokenizer,
                           mode='on-line')
    feature = next(train_BTL.__iter__())
    res, score = predict(feature)
    result = {'predictions': res, 'score': score}
    return templates.TemplateResponse('index.html', context={'request': request, 'result': result})


def preprocess_data(context):
    import re

    context = context.strip()

    context = re.sub(r'[\s]+|[0-9]+|http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F]))', "", context)
    context = re.sub(r'[a-zA-Z]?|$', "", context)

    return pd.DataFrame([context], columns=['content'])


def predict(feature):
    with tf.device('/cpu:0'):
        d = {0: '惡意', 1: '正常'}
        output = roberta(feature)

        res = d[np.argmax(output, -1)[0]]
        score = tf.math.reduce_max(output).numpy()
        return res, str(score)

# if __name__ == '__main__':
#     uvicorn.run("fastapp:app ", host="0.0.0.0", port=5003, reload=True, debug=True)