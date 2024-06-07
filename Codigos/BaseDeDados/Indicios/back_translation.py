from deep_translator import GoogleTranslator
import pandas as pd
from tqdm.auto import tqdm
from unidecode import unidecode
import re


def back_translate(texts, language_src, language_dst):
  """Implements back translation"""
  # Translate from source to target language
  translated = GoogleTranslator(source='auto', target=language_dst).translate(texts)

  # Translate from target language back to source language
  back_translated = GoogleTranslator(source='auto', target=language_src).translate(translated)

  return back_translated


def main() -> None:
    df_weak_sel_cleaned= pd.read_csv('/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Datasets/Indicios/DatasetsWeak/DataAugmentation/data_for_augmentation_n_min_max_idoneidade_financeira.csv')
    df_translated=pd.DataFrame(columns=df_weak_sel_cleaned.columns)
    for i, row in tqdm(df_weak_sel_cleaned.iterrows(), total=df_weak_sel_cleaned.shape[0], colour='blue'):
        df_translated.loc[i]=row
        if len(df_translated.loc[i,'text'])<5000:
            result=back_translate(df_translated.loc[i,'text'],'pt','ja')
        else:
            result=[]
            sentences = re.split(r'\.\s+', df_translated.loc[i,'text'])
            for sentence in tqdm(sentences,total=len(sentences), colour='green'):
                if len(sentence) < 5000:
                    result.append(back_translate(sentence,'pt','ja'))
                else:
                    sentences_virgula = []
                    for sentence_virgula in tqdm(sentence.split(','), colour='yellow'):
                        sentences_virgula.append(back_translate(sentence_virgula,'pt','ja'))
                    complete_sentence = ', '.join(sentences_virgula)
                    complete_sentence = re.sub(r'\,+', ',', complete_sentence)
                    result.append(complete_sentence)              
            result = '. '.join(result)
        result = re.sub(r'\.+', '.', result)
        result = unidecode(result)
        result = result.lower()
        df_translated.loc[i,'text']=result
    df_translated.to_csv('/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Datasets/Indicios/DatasetsWeak/DataAugmentation/data_back_translated_n_min_max_idoneidade_financeira.csv')

if __name__ == "__main__":
    main()