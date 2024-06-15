from deep_translator import GoogleTranslator
import pandas as pd
from pandarallel import pandarallel
from tqdm.auto import tqdm
from unidecode import unidecode
import re
import numpy as np


def back_translate(texts, language_src, language_dst):
  """Implements back translation"""
  # Translate from source to target language
  translated = GoogleTranslator(source='auto', target=language_dst).translate(texts)
  # Translate from target language back to source language
  back_translated = GoogleTranslator(source='auto', target=language_src).translate(translated)
  return back_translated


def back_translation_(sentence: str):
    if len(sentence) < 5000:
        result = back_translate(sentence,'pt','ja')
    else:
        result=[]
        sentences_ponto = re.split(r'\.\s+', sentence)
        for sentence_ponto in sentences_ponto:
            if len(sentence_ponto) < 5000:
                try:
                    result.append(back_translate(sentence_ponto,'pt','ja'))
                except:
                    result.append(sentence_ponto)
            else:
                sentences_virgula = []
                for sentence_virgula in sentence_ponto.split(','):
                    try:
                        sentences_virgula.append(back_translate(sentence_virgula,'pt','ja'))
                    # Exceção: No translation was found using the current translator.
                    except:
                        sentences_virgula.append(sentence_virgula)
                complete_sentence = ', '.join(sentences_virgula)
                complete_sentence = re.sub(r'\,+', ',', complete_sentence)
                result.append(complete_sentence)
        result = '. '.join(result)
    result = re.sub(r'\.+', '.', result)
    result = unidecode(result.lower())
    return result         
            
            
def main() -> None:
    pandarallel.initialize(progress_bar=True, nb_workers=2)
    dataframe_total = pd.read_csv('/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Datasets/Indicios/DatasetsWeak/somente_indicios_acima_100_14_indicios_cleaned.csv')

    splits = np.array_split(dataframe_total, 5)
    for indice, dados in tqdm(enumerate(splits), total=len(splits), desc='Realizando back translation'):
        dados['back_translated'] = dados['text'].parallel_apply(lambda x: back_translation_(x))
        dados.to_csv(f'/var/projetos/Jupyterhubstorage/victor.silva/HelBERTModel/Datasets/Indicios/DatasetsWeak/DataAugmentation/BackTranslateNovosIndicios/data_back_translated_acima_100_14_indicios-{indice+1}.csv', index=False)    


if __name__ == "__main__":
    main()