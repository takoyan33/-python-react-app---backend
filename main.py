from fastapi import FastAPI
from chat_function import ChatFunction
from pydantic import BaseModel
#リクエストBodyの検証
from fastapi.middleware.cors import CORSMiddleware
#推論api
# from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegression
# import pickle
# from fastapi import FastAPI
# import pickle

origins = [
    'http://127.0.0.1:5173',
]



# iris_datasets = load_iris()
# model = LogisticRegression(C=100)
# # 推論時に使用するためデータを1つだけ学習対象から除去
# model.fit(iris_datasets.data[:-1], iris_datasets.target[:-1])
# pickle.dump(model, open('iris_model.pkl','wb'))

# app = FastAPI()
# #chat(推論モジュール)のセッティング
# chat = ChatFunction('rinna/japanese-gpt2-medium')


# model = pickle.load(open("iris_model.pkl", "rb"))
# app = FastAPI()

# class PredictRequestBody(BaseModel):
#  input: List[float]

# @app.post("/predict")
# async def iris(req: PredictRequestBody):
#  to_predict = np.array(req.input)
#  return { "output": int(model.predict([to_predict])[0]) }

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)
#chat(推論モジュール)のセッティング
chat = ChatFunction('rinna/japanese-gpt2-medium')
#rinna株式会社様がHuggingFaceで公開されている日本語のGPT-2のテキスト生成モデル
#GPT-2は、OpenAIが開発した言語モデル（テキスト生成モデル）です。

#POSTで受け取るデータフォーマットを定義
class Message(BaseModel):
    message: str
    max_length: int = 30 #デフォルト値
    num_return_sequences: int = 1 #デフォルト値

@app.get("/")
#/にアクセスした時
def root():
    return {"message":"こんにちは"}
    #returnを返す

@app.post("/msg")
#/msgにアクセスした時
def message(msg: Message):
    output = chat.generate_msg(
        msg.message,
        #続きを生成してほしいテキスト
        msg.max_length,
        #生成するテキストの長さ
        msg.num_return_sequences
        #生成するテキストの個数
    )
    return {"message": output}
    #結果を返す