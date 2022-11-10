from transformers import T5Tokenizer, AutoModelForCausalLM
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class ChatFunction:
    def __init__(self, model_name):
        #初期化
        self.model_name = model_name
        #モデルの名前=rinna/japanese-gpt2-medium
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        #トークナイザーの準備をする
        self.tokenizer.do_lower_case = True
        #小文字化する
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
    
    def generate_msg(self, text, max_length=30, num_return_sequences=1):
        #文章生成
        input = self.tokenizer.encode(text, return_tensors="pt")
        print(input)
        output = self.model.generate(
            input, 
            do_sample=True, 
            max_length=max_length, 
            num_return_sequences=num_return_sequences
        )
        msg_list = self.tokenizer.batch_decode(output,skip_special_tokens=True)
        return [m[len(text):] for m in msg_list]
        #for分でリストに入れる


if __name__ == '__main__':
    chat = ChatFunction('rinna/japanese-gpt2-medium')
    text = 'テストだよ。'
    msg = chat.generate_msg(text)
    print(f'input:{text}')
    #python側で表示
    print(f'response:{msg}')
    #python側で表示