import pandas as pd
from tqdm import tqdm
from googletrans import Translator


def back_translation(data: str):
    translator = Translator()
    chinese_to_en = translator.translate(data, dest='en', src='zh-cn').text
    back_to_chinese = translator.translate(chinese_to_en, dest='zh-cn', src='en').text
    return back_to_chinese


if __name__ == '__main__':

    res_df = pd.DataFrame()
    content = pd.read_pickle('./data/train_data/train.pickle').reset_index(drop=True)["content"]
    res = []

    for i in tqdm(range(len(content))):
        try:
            res.append(back_translation(content[i]))
        except Exception as e:
            print(i, e.args)
            res.append('None')

    res_df['content'] = content
    res_df['back_translate'] = res
    res_df.to_pickle('./data/augment_data/back_translate_data.pkl')
