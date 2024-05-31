from copy import deepcopy
import re
from unidecode import unidecode


class Cleaner:
        
    def clear(self, paragraphs):
        if not paragraphs:
            return paragraphs
        texts = deepcopy(paragraphs)
        
        texts = self._removeParenteses(texts)
        texts = self._remove_undesired_chars(texts)
        texts = self._remove_barraNInvertido(texts)
        texts = self._remove_multiplos_hifens(texts)
        texts = self._remove_multiples_dots(texts)
        texts = self._remove_multiples_x(texts)
        texts = self._corrigeNumPalavra(texts)
        texts = self._corrigePalavraNum(texts)
        texts = self._d_ataToData(texts)
        texts = self._removeAnigap(texts)
        texts = self._removeNumSecoesInicio(texts)
        texts = self._removeNumSecoesInicio(texts)
        texts = self._removeNumSecoes(texts)
        texts = self._removeNumSecoes(texts)
        texts = self._remove_alternativas(texts)    
        return texts
    
    @staticmethod
    def _removeParenteses(paragraphs):
        """
        Remove padrões do tipo (conteudo).
        """
        return re.sub(r'\(([^()]+)\)', ' ', paragraphs)
    
    @staticmethod
    def _remove_undesired_chars(paragraphs):
        """
        Remove caracteres indesejados.
        """
        return re.sub(r'([”“\\\\•●▪•_·□»«#£¢¿&!;*{}^~´`=|[\]²¹\+¨\t\'\"\)\(]|(0xx))', ' ', paragraphs)

    @staticmethod
    def _remove_barraNInvertido(paragraphs):
        """
        Remove /n invertido.
        """
        return re.sub(r'((\/n(\s))|(\/n$))', ' ', paragraphs)

    @staticmethod
    def _remove_multiplos_hifens(paragraphs):
        """
        Remove multiplos hifens.
        O espaço antes e depois do token é por conta do padrão que retira os espaços antes e depois do texto.
        """
        paragraphs = re.sub(r'((\s)-)+', ' - ', paragraphs)
        return re.sub(r'-+', '-', paragraphs)
    
    @staticmethod
    def _remove_multiples_dots(paragraphs):
        """
        Remove multiplos pontos.
        """
        paragraphs = re.sub(r'\.+', '.', paragraphs)
        return re.sub(r'^\.\s', '', paragraphs)
    
    @staticmethod
    def _remove_multiples_x(paragraphs):
        """
        Reduz multiplos x para apenas um x.
        """
        return re.sub(r'(([xX]){2,})', 'x', paragraphs)
    
    @staticmethod
    def _corrigeNumPalavra(paragraphs):
        """
        Corrige sentenças do tipo 1palavra para 1 palavra.
        """
        return re.sub(r'((\d+)([a-zA-Z]{2,}))', r'\2 \3', paragraphs)
    
    @staticmethod
    def _corrigePalavraNum(paragraphs):
        """
        Corrige sentenças do tipo palavra1 para palavra 1.
        """
        return re.sub(r'(([a-zA-Z]{2,})(\d+))', r'\2 \3', paragraphs)
    
    @staticmethod
    def _d_ataToData(paragraphs):
        """
        Corrige d ata para data.
        """
        return re.sub(r'(([Dd] ata)|(D ATA))', 'data', paragraphs)
    
    @staticmethod
    def _removeAnigap(paragraphs):
        """
        Remove padrões do tipo a n i g á p.
        """
        return re.sub(r'(([aA] [Nn] [Ii] [Gg] [Áá] [Pp])|([aA] [Nn] [Ii] [Gg] [Áá]))', '', paragraphs)
    
    @staticmethod
    def _removeNumSecoesInicio(paragraphs):
        """
        Remove padrões de numeros de secao no início.
        """
        return re.sub(r'(^((\s)*(((\d{1,2}\.)(\d{1,2}(\.)?)*)|(\d)+)(\s)?[\-–]?))', ' ', paragraphs)
    
    @staticmethod
    def _removeNumSecoes(paragraphs):
        """
        Remove padrões de numeros de secao.
        """
        return re.sub(r'(\d{1,2}\.){1,}(\d{1,2}\s[\-–])?', ' ', paragraphs)
    
    @staticmethod
    def _remove_alternativas(paragraphs):
        """
        Remove alternativas de questões.
        """
        return re.sub(r'(^[bcdfghijklmnpqrstuvwxyz\:\/\,\.\;@-]([\.)-])?(\s)|(^[aeo][\.)-](\s)))', '', paragraphs, flags=re.IGNORECASE)
    

class Corretor:
    def __init__(self, cased: bool, accents: bool) -> None:
        self.accents = accents
        self.cased = cased
        
    def corrige_termos(self, paragraphs):
        if not paragraphs:
            return paragraphs
        texts = deepcopy(paragraphs)

        texts = self._padraoEmail(texts)
        texts = self._corrigeCapitulo(texts)
        texts = self._telToTelefone(texts)
        texts = self._NtoNumero(texts)
        texts = self._artToArtigo(texts)
        texts = self._ccToCumulativamenteCom(texts)
        texts = self._remove_SbarraN(texts)
        texts = self._c_ontratacaoToContratacao(texts)
        texts = self._memoToMemorando(texts)
        texts = self._corrigeParagrafo(texts)
        texts = self._corrigeLicitacao(texts)
        texts = self._corrigeContratacao(texts)
        texts = self._corrigePregao(texts)
        texts = self._corrigeFiscal(texts)
        texts = self._corrigeObjeto(texts)
        texts = self._corrigeValor(texts)
        texts = self._corrigeCertidao(texts)
        texts = self._corrigeEmpenho(texts)
        texts = self._corrigeQuantidade(texts)
        texts = self._corrigeAditivo(texts)
        texts = self._corrigeSancao(texts)
        texts = self._corrigeEdital(texts)
        texts = self._corrigeGarantia(texts)
        texts = self._corrigeOnus(texts)
        texts = self._corrigeReajuste(texts)
        texts = self._corrigeDigital(texts)
        texts = self._incToInciso(texts)
        texts = self._padronizaCNPJ(texts)
        texts = self._padronizaSiglas(texts)
        texts = self._tokenizaURL(texts)
        texts = self._tokenizaEmail(texts)
        texts = self._tokenizaData(texts)
        texts = self._tokenizaHora(texts)
        texts = self._tokenizaNumero(texts)
        texts = self._tokenizaNumeroRomano(texts)
        texts = self._reduzNumeros(texts)
        texts = self._removeHifenInicial(texts)
        texts = self._corrigePontuacao(texts)
        texts = self._remove_characters_inicial(texts)
        texts = self._remove_characters_final(texts)
        texts = self._remove_multiples_spaces(texts)
        texts = self._remove_space_in_last_period(texts)
        texts = self._strip_spaces(texts)

        if not self.accents:
            texts = self._removeAcentos(texts)
        
        if not self.cased:
            texts = self._toLowerCase(texts)
        
        return texts
    
    @staticmethod
    def _padraoEmail(paragraphs):
        """
        Padroniza a palavra email.
        """
        return re.sub(r'(([eE][\-–]?mail([.:])*)|(E[\-–]?MAIL([.:])*))', 'email', paragraphs)
        
    @staticmethod
    def _corrigeCapitulo(paragraphs):
        """
        Corrige c apitulo para capitulo.
        """
        return re.sub(r'(c(\s)ap[ií]tulo)', 'capítulo', paragraphs, flags=re.IGNORECASE)

    @staticmethod
    def _telToTelefone(paragraphs):
        """
        Corrige tel para telefone.
        """
        return re.sub(r'(\b(tel)\b)', 'telefone', paragraphs, flags=re.IGNORECASE)

    @staticmethod
    def _NtoNumero(paragraphs):
        """
        Convert nº para numero.
        """
        paragraphs = re.sub(r'((\s)((([nN][º°])|(n[uú]mero)\.)|((([nN])|(([Nn][uÚ]mero)|(N[UÚ]MERO)))\.[º°])|([nN][º°])|([nN]\.)|([Nn][rR]))(\s)?)', ' número ', paragraphs)
        return re.sub(r'([º°])', ' ', paragraphs)
    
    @staticmethod
    def _artToArtigo(paragraphs):
        """
        Corrige art. para artigo.
        """
        return re.sub(r'(\b([Aa]rt\.)|(ART\.)|(([Aa]rt)|(ART))\b)', 'artigo', paragraphs)
    
    @staticmethod
    def _ccToCumulativamenteCom(paragraphs):
        """
        Corrige c/c para cumulativamente com.
        """
        return re.sub(r'(\b[cC]\/[cC]\b)', 'cumulativamente com', paragraphs)

    @staticmethod
    def _remove_SbarraN(paragraphs):
        """
        Corrige padrões do tipo s/n para sem numero.
        """
        return re.sub(r'(((\b([sS]\/(([nN]º)|([nN]))[,.]?))|([sS]\/(([nN]º)|([nN]))[,.]))(\s|\,|\.|\:|\;))', 'sem número ', paragraphs)
    
    @staticmethod
    def _c_ontratacaoToContratacao(paragraphs):
        """
        Corrige c ontratação para contratação.
        """
        return re.sub(r'(\b([cC] ontrata[cç][aã]o)|(C ONTRATA[CÇ][AÃ]O)\b)', 'contratação', paragraphs)
    
    @staticmethod
    def _memoToMemorando(paragraphs):
        """
        Corrige memo para memorando.
        O espaço antes e depois do token é por conta do padrão que retira os espaços antes e depois do texto.
        """
        return re.sub(r'(\b(([mM]emo)|(MEMO))\b)', 'memorando', paragraphs)

    @staticmethod
    def _corrigeParagrafo(paragraphs):
        """
        Corrige formas de parágrafo para parágrafo.
        O espaço antes e depois do token é por conta do padrão que retira os espaços antes e depois do texto.
        """
        paragraphs = re.sub(r'((\s|\.|\,|\;|\:|^)(§{2,})º?(\s|\.|\,|\;|\:|$))', ' parágrafos ', paragraphs)
        return re.sub(r'((\s|\.|\,|\;|\:|^)(§)º?(\s|\.|\,|\;|\:|$))', ' parágrafo ', paragraphs)
    
    @staticmethod
    def _corrigeLicitacao(paragraphs):
        """
        Corrige ocorrências da palavra ou Licitaçã o" para "licitação".
        Corrige ocorrências da palavra ou "Licitaçã" para "licitação".
        """
        paragraphs = re.sub(r'(\s|\.|\,|\;|\:)(iicitante)(\s|\.|\,|\;|\:)', ' licitante ', paragraphs, flags=re.IGNORECASE)
        paragraphs = re.sub(r'((\s|\.|\,|\;|\:)(licita[çc][aã](\s)o)(\s|\.|\,|\;|\:))', ' licitação ', paragraphs, flags=re.IGNORECASE)
        return re.sub(r'((\s|\.|\,|\;|\:)(licita[çc][aã])(\s|\.|\,|\;|\:))', ' licitação ', paragraphs, flags=re.IGNORECASE)
    
    @staticmethod
    def _corrigeContratacao(paragraphs):
        """
        Corrige ocorrências da palavra Contrataçã o" para "contratação".
        Corrige ocorrências da palavra "Contrataçã" para "contratação".
        """
        paragraphs = re.sub(r'((\s|\.|\,|\;|\:)(contrata[çc][aã](\s)o)(\s|\.|\,|\;|\:))', ' contratação ', paragraphs, flags=re.IGNORECASE)
        paragraphs = re.sub(r'(\s|\.|\,|\;|\:)(contrante)(\s|\.|\,|\;|\:)', ' contratante ', paragraphs, flags=re.IGNORECASE)
        return re.sub(r'((\s|\.|\,|\;|\:)(contrata[çc][aã])(\s|\.|\,|\;|\:))', ' contratação ', paragraphs, flags=re.IGNORECASE)
    
    @staticmethod
    def _corrigePregao(paragraphs):
        """
        Corrige ocorrências da palavra "P REGÃO" para "pregão".
        Corrige ocorrências da palavra "regão" para "pregão".
        """
        paragraphs = re.sub(r'((\s|\.|\,|\;|\:)(p(\s)reg[ãa]o)(\s|\.|\,|\;|\:))', ' pregão ', paragraphs, flags=re.IGNORECASE)
        paragraphs = re.sub(r'(\s|\.|\,|\;|\:)(pregoelro)(\s|\.|\,|\;|\:)', ' pregoeiro ', paragraphs, flags=re.IGNORECASE)
        paragraphs = re.sub(r'(\s|\.|\,|\;|\:)(opregoeiro)(\s|\.|\,|\;|\:)', ' o pregoeiro ', paragraphs, flags=re.IGNORECASE)
        paragraphs = re.sub(r'(\s|\.|\,|\;|\:)(apregoeira)(\s|\.|\,|\;|\:)', ' a pregoeira ', paragraphs, flags=re.IGNORECASE)
        paragraphs = re.sub(r'(\s|\.|\,|\;|\:)(pelopregoeiro)(\s|\.|\,|\;|\:)', ' pelo pregoeiro ', paragraphs, flags=re.IGNORECASE)
        paragraphs = re.sub(r'(\s|\.|\,|\;|\:)(pelapregoeira)(\s|\.|\,|\;|\:)', ' pela pregoeira ', paragraphs, flags=re.IGNORECASE)
        paragraphs = re.sub(r'(\s|\.|\,|\;|\:)(pregoeir a)(\s|\.|\,|\;|\:)', ' pregoeira ', paragraphs, flags=re.IGNORECASE)
        paragraphs = re.sub(r'(\s|\.|\,|\;|\:)(pregoeir o)(\s|\.|\,|\;|\:)', ' pregoeiro ', paragraphs, flags=re.IGNORECASE)
        paragraphs = re.sub(r'(\s|\.|\,|\;|\:)(pregoeir)(\s|\.|\,|\;|\:)', ' pregoeiro ', paragraphs, flags=re.IGNORECASE)
        paragraphs = re.sub(r'((\s|\.|\,|\;|\:)(reg[ãa]o)(\s|\.|\,|\;|\:))', ' pregão ', paragraphs, flags=re.IGNORECASE)
        paragraphs = re.sub(r'((\s|\.|\,|\;|\:)((preg)(\s)(pregoeira)|(pregoeira(\s)preg))(\s|\.|\,|\;|\:))', ' pregoeira ', paragraphs, flags=re.IGNORECASE)
        paragraphs = re.sub(r'((\s|\.|\,|\;|\:)((preg)(\s)(pregoeiro)|(pregoeiro(\s)preg))(\s|\.|\,|\;|\:))', ' pregoeiro ', paragraphs, flags=re.IGNORECASE)
        paragraphs = re.sub(r'((\s|\.|\,|\;|\:)(preg(\s)[aã]o)(\s|\.|\,|\;|\:))', ' pregão ', paragraphs, flags=re.IGNORECASE)
        return re.sub(r'((\s|\.|\,|\;|\:)(preg)(\s|\.|\,|\;|\:))', ' pregão ', paragraphs, flags=re.IGNORECASE)
    
    @staticmethod
    def _corrigeFiscal(paragraphs):
        """
        Corrige ocorrências da palavra "F i scal" para "fiscal".
        Corrige ocorrências da palavra "scal" para "fiscal".
        """
        paragraphs = re.sub(r'(\s|\.|\,|\;|\:|^)(f(\s)i(\s)scal)(\s|\.|\,|\;|\:)', ' fiscal ', paragraphs, flags=re.IGNORECASE)
        return re.sub(r'(\s|\.|\,|\;|\:|^)(scal)(\s|\.|\,|\;|\:)', ' fiscal ', paragraphs, flags=re.IGNORECASE)
    
    @staticmethod
    def _corrigeObjeto(paragraphs):
        """
        Corrige ocorrências da palavra "O bjeto" para "objeto".
        Corrige ocorrências da palavra "bjeto" para "objeto".
        """
        paragraphs = re.sub(r'(\s|\.|\,|\;|\:|^)(o?bjeto(\s)(o(\s))?bjeto)(\s|\.|\,|\;|\:)', ' objeto ', paragraphs, flags=re.IGNORECASE)
        paragraphs = re.sub(r'(\s|\.|\,|\;|\:|^)(abjeto)(\s|\.|\,|\;|\:)', ' objeto ', paragraphs, flags=re.IGNORECASE)
        return re.sub(r'(\s|\.|\,|\;|\:|^)(bjeto)(\s|\.|\,|\;|\:)', ' objeto ', paragraphs, flags=re.IGNORECASE)

    @staticmethod
    def _corrigeValor(paragraphs):
        """
        Corrige ocorrências da palavra "valo r" para "valor".
        Corrige ocorrências da palavra "valo" para "valor".
        """
        paragraphs = re.sub(r'(\s|\.|\,|\;|\:|^)(valo(\s)r)(\s|\.|\,|\;|\:)', ' valor ', paragraphs, flags=re.IGNORECASE)
        return re.sub(r'(\s|\.|\,|\;|\:|^)(valo)(\s|\.|\,|\;|\:)', ' valor ', paragraphs, flags=re.IGNORECASE)
    
    @staticmethod
    def _corrigeCertidao(paragraphs):
        """
        Corrige ocorrências da palavra "cer dão" para "certidão".
        """
        return re.sub(r'(\s|\.|\,|\;|\:|^)(cer(\s)d[ãa]o)(\s|\.|\,|\;|\:)', ' certidão ', paragraphs, flags=re.IGNORECASE)
    
    @staticmethod
    def _corrigeEmpenho(paragraphs):
        """
        Corrige ocorrências da palavra "emprenho" para "empenho".
        """
        return re.sub(r'(\s|\.|\,|\;|\:|^)(emprenho)(\s|\.|\,|\;|\:)', ' empenho ', paragraphs, flags=re.IGNORECASE)
    
    @staticmethod
    def _corrigeQuantidade(paragraphs):
        """
        Corrige ocorrências da palavra "quantidad" para "quantidade".
        """
        return re.sub(r'(\s|\.|\,|\;|\:|^)(quantidad)(\s|\.|\,|\;|\:)', ' quantidade ', paragraphs, flags=re.IGNORECASE)
    
    @staticmethod
    def _corrigeAditivo(paragraphs):
        """
        Corrige ocorrências da palavra "adi vo" para "aditivo".
        """
        return re.sub(r'(\s|\.|\,|\;|\:|^)(adi(\s|\.|\,|\;|\:)vo[s]?)(\s|\.|\,|\;|\:)', ' aditivo ', paragraphs, flags=re.IGNORECASE)
    
    @staticmethod
    def _corrigeSancao(paragraphs):
        """
        Corrige ocorrências da palavra "sansão" para "sanção".
        Corrige ocorrências da palavra "sansões" para "sanções".
        """
        paragraphs = re.sub(r'(\s|\.|\,|\;|\:|^)(sans[ãa]o)(\s|\.|\,|\;|\:)', ' sanção ', paragraphs, flags=re.IGNORECASE)
        return re.sub(r'(\s|\.|\,|\;|\:|^)(sans[oõ]es)(\s|\.|\,|\;|\:)', ' sanções ', paragraphs, flags=re.IGNORECASE)
    
    @staticmethod
    def _corrigeEdital(paragraphs):
        """
        Corrige ocorrências da palavra "editai" para "edital".
        """
        paragraphs = re.sub(r'(\s|\.|\,|\;|\:|^)(edita(\s)l)(\s|\.|\,|\;|\:)', ' edital ', paragraphs, flags=re.IGNORECASE)
        paragraphs = re.sub(r'(\s|\.|\,|\;|\:|^)(editai)(\s|\.|\,|\;|\:)', ' edital ', paragraphs, flags=re.IGNORECASE)
        return re.sub(r'(\s|\.|\,|\;|\:|^)(edita)(\s|\.|\,|\;|\:)', ' edital ', paragraphs, flags=re.IGNORECASE)
    
    @staticmethod
    def _corrigeGarantia(paragraphs):
        """
        Corrige ocorrências da palavra "garana" para "garantia".
        """
        paragraphs = re.sub(r'(\s|\.|\,|\;|\:|^)(garanti(\s|\.|\,|\;|\:)a)(\s|\.|\,|\;|\:)', ' garantia ', paragraphs, flags=re.IGNORECASE)
        return re.sub(r'(\s|\.|\,|\;|\:|^)((garana)|(garania))(\s|\.|\,|\;|\:)', ' garantia ', paragraphs, flags=re.IGNORECASE)
    
    @staticmethod
    def _corrigeOnus(paragraphs):
        """
        Corrige ocorrências da palavra "anus" para "ônus".
        """
        return re.sub(r'(\s|\.|\,|\;|\:|^)(anus)(\s|\.|\,|\;|\:)', ' ônus ', paragraphs, flags=re.IGNORECASE)
    
    @staticmethod
    def _corrigeReajuste(paragraphs):
        """
        Corrige ocorrências da palavra "parareajuste" para "para reajuste".
        """
        return re.sub(r'(\s|\.|\,|\;|\:|^)(parareajuste)(\s|\.|\,|\;|\:)', ' para reajuste ', paragraphs, flags=re.IGNORECASE)
    
    @staticmethod
    def _corrigeDigital(paragraphs):
        """
        Corrige ocorrências da palavra "digitai" para "digital".
        """
        return re.sub(r'(\s|\.|\,|\;|\:|^)(digitai)(\s|\.|\,|\;|\:|$)', ' digital ', paragraphs, flags=re.IGNORECASE)

    @staticmethod
    def _incToInciso(paragraphs):
        """
        Corrige inc. para inciso.
        O espaço antes e depois do token é por conta do padrão que retira os espaços antes e depois do texto.
        """
        return re.sub(r'(\b(inc)\b)', 'inciso', paragraphs, flags=re.IGNORECASE)
    
    @staticmethod
    def _padronizaCNPJ(paragraphs):
        """
        Padroniza CNPJ.
        """
        return re.sub(r'(\b(c\.n\.p\.j\.?)\b)', 'cnpj', paragraphs, flags=re.IGNORECASE)
    
    @staticmethod
    def _padronizaSiglas(paragraphs):
        """
        Corrige lc para lei complementar.
        O espaço antes e depois do token é por conta do padrão que retira os espaços antes e depois do texto.
        """
        paragraphs = re.sub(r'(\b(cf)\b)', 'Constituição Federal', paragraphs, flags=re.IGNORECASE)
        paragraphs = re.sub(r'(\b(n?cpc)(\/15)?\b)', 'Código de Processo Civil', paragraphs, flags=re.IGNORECASE)
        paragraphs = re.sub(r'(\b(clt)\b)', 'Consolidação das Leis do Trabalho', paragraphs, flags=re.IGNORECASE)
        paragraphs = re.sub(r'(\b(cdc)\b)', 'Código de Defesa do Consumidor', paragraphs, flags=re.IGNORECASE)
        return re.sub(r'(\b(lc)\b)', 'lei complementar', paragraphs, flags=re.IGNORECASE)
    
    @staticmethod
    def _tokenizaURL(paragraphs):
        """
        Tokeniza URL.
        """
        return re.sub(r'(((https?:\/\/)(www\.))|(www\.)|(https?:\/\/))[-a-zA-Z0-9@:%.\+~#=]{1,256}\.[a-zA-Z@0-9()]{1,6}\b([-a-zA-Z0-9()@:%\+.~#?&\/=]*)', 'url', paragraphs)
    
    @staticmethod
    def _tokenizaEmail(paragraphs):
        """
        Tokeniza email.
        """
        return re.sub(r'(\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b)', 'mail', paragraphs)

    @staticmethod
    def _tokenizaData(paragraphs):
        """
        Tokeniza data.
        """
        return re.sub(r'(\b([0-3][0-9]\/[0-1][0-9]\/(([0-9]{2})|([0-2][0-9]{3})))\b)', 'date', paragraphs)

    @staticmethod
    def _tokenizaHora(paragraphs):
        """
        Tokeniza hora.
        """
        return re.sub(r'(\b(([0-1][0-9])|(2[0-3]))(\:|h)([0-5][0-9])?\b)', 'hour', paragraphs)
    
    @staticmethod
    def _tokenizaNumero(paragraphs):
        """
        Tokeniza número.
        """
        return re.sub(r'(\b([0-9]+)\b)', 'number', paragraphs)
    
    @staticmethod
    def _tokenizaNumeroRomano(paragraphs):
        """
        Tokeniza número romano.
        """
        return re.sub(r"(\s|\.|\,|\;|\:|^)(?=[XVIΙ])(XC|XL|L?X{0,3})([IΙ]X|[IΙ]V|V?[IΙ]{0,3})(\s|\.|\,|\;|\:|$)", 'number', paragraphs, flags=re.IGNORECASE)
    
    @staticmethod
    def _reduzNumeros(paragraphs):
        """
        Reduz números.
        """
        return re.sub(r'(([\.\\\/\;\:\s])*(<NUMERO>([\-–\.\\\/\;\:\,\s])*)+)', ' number ', paragraphs)

    @staticmethod
    def _removeHifenInicial(paragraphs):
        """
        Remove hifen inicial.
        """
        return re.sub(r'^\s*[\-–]\s*', '', paragraphs)
    
    @staticmethod
    def _corrigePontuacao(paragraphs):
        """
        Corrige má formatação nas vírgulas (nome , outro nome).
        Corrige má formatação nos pontos (nome . Outro nome).
        Corrige má formatação na pontuação (.,:).
        """
        paragraphs = re.sub(r'((\s)+(\,))', ',', paragraphs)
        paragraphs = re.sub(r'((\s)+(\.))', '.', paragraphs)
        paragraphs = re.sub(r'((\s)+(\:))', ':', paragraphs)
        # Mais de uma pontuação
        paragraphs = re.sub(r'((\,)+)', ',', paragraphs)
        paragraphs = re.sub(r'((\.)+)', '.', paragraphs)
        paragraphs = re.sub(r'((\:)+)', ':', paragraphs)
        return re.sub(r'((\.\,)|(\,\.))', '.', paragraphs)
    
    @staticmethod
    def _remove_characters_inicial(paragraphs):
        """
        Remove caracteres finais.
        """
        return re.sub(r'(^((\s)*[ª\%\·\.\:\\\/\,\;@\-–]){1,})', '', paragraphs)
        
    @staticmethod
    def _remove_characters_final(paragraphs):
        """
        Remove caracteres finais.
        """
        return re.sub(r'(((\s)*[ª\·\:\\\/\,\;@\-–]){1,}$)', '', paragraphs)
    
    @staticmethod
    def _remove_multiples_spaces(paragraphs):
        """
        Remove multiplos espaços nas sentenças.
        """
        return re.sub(r'\s+', ' ', paragraphs)
    
    @staticmethod
    def _remove_space_in_last_period(paragraphs):
        """
        Remove espaços finais na sentença e coloca o ponto final.
        """
        return re.sub(r'\s\.$', '.', paragraphs)
    
    @staticmethod
    def _strip_spaces(paragraphs):
        """
        Remove espaços antes e depois do texto.
        """
        return paragraphs.strip()
    
    @staticmethod
    def _removeAcentos(paragraphs):
        """
        Remove acentos das palavras.
        """
        return unidecode(paragraphs)
    
    @staticmethod
    def _toLowerCase(paragraphs):
        """
        Converte o texto para caixa baixa.
        """
        return paragraphs.lower()
    

class Remover:
    def removeSentences(self, paragraphs):
        if not paragraphs:
            return paragraphs
        texts = deepcopy(paragraphs)

        texts = self._removeSentencasComCaracteresIndesejados(texts)
        texts = self._removePalavrasQuebradas(texts)

        return texts
        
    @staticmethod
    def _removeSentencasComCaracteresIndesejados(paragraphs):
        """
        Remove sentenças que têm interrogação.
        """
        if re.search(r'([\@\?ʼˢ˫βδθιμξοσφωώасфֹֻ־׀׃ׄ؛؟عُ٭ڊڎڡଋ฀ขงจฉซญฏทนผภวษฬᴏᴓᴼᵌᵒᶟᶲᶾḉṕầẽễ‐‑‒–—―‖‗‘’‚‛„‟†‡‥…‰′″‹›‾⁄⁒⁰⁴⁵⁶⁻⁽⁾ⁿ₀₂₄ₒ€₹⃝⃰℃℉ℎℓ№™⅓⅔⅜⅝ⅱ（）＋，－：＝ｪｺﾁﾉﾍﾤ￼𝐀𝐂𝐋𝐍𝐎𝐏𝐐𝐑𝐒𝐕𝐚𝐜𝐞𝐠𝐢𝐥𝐦𝐧𝐨𝐫𝐬𝐭𝐮𝐯𝐳𝐴𝐵𝐶𝐷𝐸𝐹𝐺𝐻𝐼𝐿𝑀𝑁𝑂𝑃𝑄𝑅𝑆𝑇𝑈𝑉𝑋𝑍𝑎𝑏𝑐𝑑𝑒𝑓𝑔𝑖𝑗𝑘𝑙𝑚𝑛𝑜𝑝𝑞𝑟𝑠𝑡𝑢𝑣𝑥𝑧𝑨𝑩𝑪𝑫𝑭𝑮𝑰𝑲𝑳𝑴𝑵𝑶𝑷𝑸𝑹𝑺𝑽𝒂𝒄𝒅𝒆𝒇𝒈𝒊𝒌𝒍𝒎𝒏𝒐𝒑𝒒𝒓𝒔𝒕𝒖𝒗𝒛🌞←↑→↓↔↩⇁⇒⇛∅∆∑−∕∗∘∙√∝∞∫∬∴≃≅≈≠≤≥≦≺≼⊠⋅⋯⌧〈〉⎓⎛⎜⎝⎞⎟⎠⎡⎢⎣⎤⎯①②③④⑤⑥⑦⑧⑨⑩⓭⓮⓯─│┤═▒■▲▶▷►▼◄◆◊○◐◗◙◦◯◻◼◾☐☒☺♀♠♣♥♦♪♯⚫✇✈✉✐✓✔✗✦✱✲✳✴✵✶✷✹✺✻✼✽✾✿❀❁❂❄❆❇❈❉❊❋❍❏❑❒❖❘❙❚❜❤❥❦❧❯❱❲❳❴❶❷❸❹❺❻❼❽❾❿➀➁➂➃➄➅➋➔➙➛➢➷⦁⩾⮚。〃「」】〔〕〖ꞏꟷ갇갭떷퐀ﬀﬁﬂﬃﬄε¸äåæèëìîïð¯§¶ñòöøùûüýþÿăęőœŕřšžǧ˝あえがきくさしすつでなぬねふまむめもやゆるれわゑをイセタトブプミムヰン・㊦㌧㌫㌱㍉㍍㘷丁七上且丘乞亀云互五亘亜亡亨京亭什仰仲伍会佃低何使來倉健側儀億允克冊出函切別到則剛副劇加劣助勘勾匂匝匤印卲厄厨句叩叫可台叶吃各合同名吠含呑呼商喜喝喧嗣嘩嘱噌噴回国圄址坤垂埠場士壬壷奮如姫孤宜宣寄富寒対尊尤屯山岨岳嵩嶋川州工左巳巴巾幼庇廿当心急意慧成手押抽担拙持挿捕捻揃描損撃整旧晦暮曲書最朋楓檲櫨歡止正毎毛治注浮灣為焔煎照爪独獄王班璯生田申町畦番疵的皿盃盆直着瞳確磨祖禽秘種竜章競筋簡簿紅純細紺組締縊缶置罵翻耽聞聾肋肌肥肴胞脆脇脚膏臆自至舟舶艇色芯荷萱葦蘭虹蜜蝿融血裏西观言訃訊討証詰誇誓語説読誼調論諾謀謹譜議豆象豹貂負蹄車軋軍軸輩辞迎迫造都醜重量金釦鈍鈎鈬鋏鋤鋳録鍋鐘関閥閲阀階随雀雄雪電震霜露面音韸餌館馘香馨馰馴髄魯鯖鰹鳶鴫鵬鶴ǿ˙ɛɵʃśʺʽ˂ˇ˃ˆː˗˘˛˜\d])', paragraphs, flags=re.IGNORECASE):
            return ''
        else:
            return paragraphs
    
    @staticmethod
    def _removePalavrasQuebradas(paragraphs):
        """
        Remove palavras quebradas.
        """
        if re.search(r'([A-Za-záàâãéèêíïóôõöúçñÁÀÂÃÉÈÍÏÓÔÕÖÚÇÑ\?\,\.\:\;@\/\<\>]\s){4,}', paragraphs):
            return ''
        else:
            return paragraphs